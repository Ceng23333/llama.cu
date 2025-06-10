mod exec;
mod handle;
mod load;
mod memory;
mod model;
mod op;
mod utils;

use crate::{
    exec::{Command, Output, engine},
    model::{ChatTemplate, GGufModel, map_files},
    utils::meta,
};
use exec::Request;
use ggus::GGufMetaMapExt;
use log::{debug, info};
use nn::Tensor;
use operators::cuda::{self, Device};
use std::{
    collections::BTreeMap,
    ffi::c_int,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
        Mutex,
    },
    time::{Duration, Instant},
};
use tokeneer::{Bpe, Tokeneer};

pub use crate::op::random_sample::SampleArgs;
pub use exec::{DistKVCache, Session, SessionId};
pub use model::Message;
pub use tokeneer::{TextBuf, utok};

pub type ModelConfig = (PathBuf, Box<[c_int]>, usize);

pub struct Service {
    handles: Vec<(String, Sender<Command>, std::thread::JoinHandle<()>)>,
    terminal: Terminal,
    global_receiver: Receiver<Output>,
}

#[derive(Clone)]
pub struct Terminal {
    senders: Arc<BTreeMap<String, Sender<Command>>>,
    cache_parts: Arc<BTreeMap<String, Box<[(Device, usize)]>>>,
    components: Arc<OnceLock<ModelComponents>>,
    session_models: Arc<Mutex<BTreeMap<SessionId, String>>>,
}

pub enum ReturnReason {
    Finish,
    Overflow,
}

#[derive(Default)]
pub struct Received {
    pub sessions: Vec<(Session, ReturnReason)>,
    pub outputs: BTreeMap<SessionId, Vec<utok>>,
}

struct ModelComponents {
    tokenizer: Tokeneer<Bpe>,
    chat_template: Option<ChatTemplate>,
    cache_template: Tensor<usize, 2>,
    eos: utok,
}

impl Service {
    pub fn new_with_configs(configs: Vec<(String, ModelConfig)>, use_cuda_graph: bool) -> Self {
        info!("start inference with {} models, configs: {:?}", configs.len(), configs);
        let mut handles = Vec::with_capacity(configs.len());
        let mut senders = BTreeMap::new();
        let mut cache_parts = BTreeMap::new();
        let once = Arc::new(OnceLock::new());
        let session_models = Arc::new(Mutex::new(BTreeMap::new()));
        
        // Create a single global receiver and multiple senders
        let (global_sender, global_receiver) = mpsc::channel();
        
        for (name, (model, gpus, _)) in &configs {
            let (sender, commands) = mpsc::channel();
            senders.insert(name.clone(), sender.clone());
            
            let maps = map_files(model);
            let gpus = gpus.to_vec();
            let gpus_ = gpus.clone();
            let once_ = once.clone();
            let global_sender = global_sender.clone();
            
            assert!(cuda::init().is_ok());
            debug!("cache_parts insert: name: {:?}, gpus: {:?}", name, gpus);
            cache_parts.insert(name.clone(), gpus.iter().map(|&i| (Device::new(i), 1)).collect());
            
            let handle = std::thread::spawn(move || {
                let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
                gguf.insert_sin_cos();

                let tokenizer = Bpe::from_gguf(&gguf);
                let chat_template = gguf.chat_template(&tokenizer);
                let cache_template = gguf.kv_cache();
                let eos = meta![gguf => tokenizer_ggml_eos_token_id];

                once_.get_or_init(|| ModelComponents {
                    tokenizer,
                    chat_template,
                    cache_template,
                    eos,
                });
                drop(once_);

                let llama = gguf.llama();
                engine(llama, &gpus_, commands, global_sender, use_cuda_graph)
            });
            
            handles.push((name.clone(), sender, handle));
        }
        
        once.wait();
        
        // Wait for all models to be ready
        for _ in 0..configs.len() {
            assert!(matches!(global_receiver.recv().unwrap(), Output::Ready));
        }
        
        info!("all models ready for inference");
        
        Self {
            handles,
            terminal: Terminal {
                senders: Arc::new(senders),
                cache_parts: Arc::new(cache_parts),
                components: once,
                session_models,
            },
            global_receiver,
        }
    }

    pub fn new(model: impl AsRef<Path>, gpus: &[c_int], use_cuda_graph: bool) -> Self {
        Self::new_with_configs(vec![("default".to_string(), (model.as_ref().to_path_buf(), gpus.to_vec().into_boxed_slice(), 0))], use_cuda_graph)
    }

    pub const fn terminal(&self) -> &Terminal {
        &self.terminal
    }

    pub fn recv(&self, timeout: Duration) -> Received {
        let time = Instant::now();
        let mut received = Received::default();
        
        match self.global_receiver.recv_timeout(timeout) {
            Ok(output) => {
                self.handle_output(output, &mut received);
            },
            Err(RecvTimeoutError::Timeout) => (),
            Err(RecvTimeoutError::Disconnected) => unreachable!(),
        }
        
        self.recv_all(timeout.saturating_sub(time.elapsed()), &mut received);
        received
    }

    pub fn try_recv(&self) -> Received {
        let mut received = Received::default();
        self.recv_all(Duration::MAX, &mut received);
        received
    }

    fn recv_all(&self, timeout: Duration, received: &mut Received) {
        let time = Instant::now();
        loop {
            match self.global_receiver.try_recv() {
                Ok(output) => {
                    self.handle_output(output, received);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => unreachable!(),
            }
            if time.elapsed() >= timeout {
                break;
            }
        }
    }

    fn handle_output(&self, output: Output, received: &mut Received) {
        match output {
            Output::Overflow(sessions) => received
                .sessions
                .extend(sessions.into_iter().map(|s| (s, ReturnReason::Overflow))),
            Output::Removed(session) => {
                self.terminal.session_models.lock().unwrap().remove(&session.id);
                received.sessions.push((session, ReturnReason::Finish))
            },
            Output::Complete {
                output,
                kv_pair,
                event,
                finished: no_decode,
            } => {
                let session_models = self.terminal.session_models.lock().unwrap();
                let device = if let Some((session_id, _)) = output.first() {
                    if let Some(model_name) = session_models.get(session_id) {
                        if let Some(cache_parts) = self.terminal.cache_parts.get(model_name) {
                            cache_parts.as_ref()[0].0
                        } else {
                            unreachable!("Model should have cache parts")
                        }
                    } else {
                        unreachable!("Session should have a model")
                    }
                } else {
                    unreachable!("Output should have at least one session")
                };

                let mut outputs = device
                    .retain_primary()
                    .apply(|ctx| exec::decode(output, kv_pair, event, &ctx.stream()));
                let components = self.terminal.components.wait();
                for (&id, toks) in &mut outputs {
                    if toks.contains(&components.eos) {
                        if let Some(model_name) = session_models.get(&id) {
                            if let Some(sender) = self.terminal.senders.get(model_name) {
                                sender.send(Command::Remove(id)).unwrap();
                            }
                        }
                    }
                }
                received
                    .sessions
                    .extend(no_decode.into_iter().map(|s| (s, ReturnReason::Finish)));
                for (id, mut tokens) in outputs {
                    if let Some((len, _)) = tokens
                        .iter()
                        .enumerate()
                        .find(|(_, t)| **t == components.eos)
                    {
                        tokens.truncate(len)
                    }
                    received.outputs.entry(id).or_default().extend(tokens)
                }
            }
            Output::Ready => unreachable!(),
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let Terminal {
            senders,
            cache_parts,
            ..
        } = &self.terminal;
        
        // Send shutdown command to all models
        for sender in senders.values() {
            sender.send(Command::ShutDown).unwrap();
        }
        
        for (model_name, _, handle) in self.handles.drain(..) {
            handle.join().unwrap();
            cache_parts.get(&model_name).unwrap()[0].0.retain_primary().apply(|ctx| {
                while let Ok(output) = self.global_receiver.try_recv() {
                    output.drop_on(ctx);
                }
            });
            info!("model {} dropped", model_name);
        }
    }
}

impl Terminal {
    pub fn get_session_model(&self, session_id: SessionId) -> Option<String> {
        self.session_models.lock().unwrap().get(&session_id).cloned()
    }

    pub fn set_session_model(&self, session_id: SessionId, model_name: String) {
        self.session_models.lock().unwrap().insert(session_id, model_name);
    }

    pub fn new_cache(&self) -> DistKVCache {
        self.new_cache_with_model("default").unwrap()
    }

    pub fn new_cache_with_model(&self, model_name: &str) -> Option<DistKVCache> {
        if let Some(cache_parts) = self.cache_parts.get(model_name) {
            Some(DistKVCache::new(&self.components.wait().cache_template, cache_parts))
        } else {
            None
        }
    }

    pub fn render(&self, msgs: &[Message]) -> String {
        self.components
            .wait()
            .chat_template
            .as_ref()
            .unwrap()
            .render(msgs, true)
            .unwrap()
    }

    pub fn tokenize(&self, text: &str) -> Vec<utok> {
        self.components.wait().tokenizer.encode(text)
    }

    pub fn start(&self, session: Session, tokens: &[utok], max_steps: usize) -> bool {
        assert_ne!(max_steps, 0, "Cannot decode 0 step");
        let model_name = session.model.clone();
        self.set_session_model(session.id, model_name.clone());
        debug!("start terminal with session_models: {:?}", self.session_models.lock().unwrap());
        if let Some(sender) = self.senders.get(&model_name) {
            debug!("start terminal with sender: {:?}", sender);
            sender
                .send(Command::Insert(Request {
                    session,
                    prompt: tokens.to_vec().into(),
                    out: 1,
                    max_steps,
                }))
                .is_ok()
        } else {
            debug!("start terminal with sender not found: {:?}", model_name);
            false
        }
    }

    pub fn stop(&self, id: SessionId) -> bool {
        if let Some(model_name) = self.get_session_model(id) {
            if let Some(sender) = self.senders.get(&model_name) {
                sender.send(Command::Remove(id)).is_ok()
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn decode(&self, tokens: &[utok], buf: &mut TextBuf) -> String {
        self.components.wait().tokenizer.decode(tokens, buf)
    }
}
