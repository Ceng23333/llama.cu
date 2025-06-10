mod bench;
mod chat;
mod generate;
mod logger;
mod service;

use clap::Parser;
use regex::Regex;
use std::{ffi::c_int, path::PathBuf, sync::LazyLock, collections::HashMap};

#[macro_use]
extern crate clap;

fn main() {
    logger::init();
    use Commands::*;
    match Cli::parse().command {
        Generate(args) => args.generate(),
        Chat(args) => args.chat(),
        Service(args) => args.service(),
        Bench(args) => args.bench(),
    }
}

#[derive(Parser)]
#[clap(name = "InfiniLM")]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// text generation
    Generate(generate::GenerateArgs),
    /// chat in console
    Chat(chat::ChatArgs),
    /// web service
    Service(service::ServiceArgs),
    /// batched benchmark
    Bench(bench::BenchArgs),
}

#[derive(Args)]
struct BaseArgs {
    #[clap(long)]
    model_path: Vec<PathBuf>,
    #[clap(long)]
    model_name: Vec<String>,
    #[clap(long)]
    gpus: Vec<String>,
    #[clap(long)]
    max_steps: Vec<usize>,
    #[clap(long)]
    no_cuda_graph: bool,
}

#[derive(Clone)]
struct ModelConfig {
    path: PathBuf,
    gpus: Box<[c_int]>,
    max_steps: usize,
}

impl BaseArgs {
    fn get_model_configs(&self) -> HashMap<String, ModelConfig> {
        let mut configs = HashMap::new();
        
        // Generate default names if needed
        let model_names: Vec<String> = if self.model_name.is_empty() {
            self.model_path.iter()
                .enumerate()
                .map(|(i, _)| format!("model_{}", i))
                .collect()
        } else {
            self.model_name.clone()
        };

        // Ensure we have enough names
        let model_names = if model_names.len() < self.model_path.len() {
            let mut names = model_names;
            for i in names.len()..self.model_path.len() {
                names.push(format!("model_{}", i));
            }
            names
        } else {
            model_names
        };

        // Create configs for each model
        for (i, (path, name)) in self.model_path.iter().zip(model_names).enumerate() {
            let gpus = if i < self.gpus.len() {
                self.parse_gpus(&self.gpus[i])
            } else if !self.gpus.is_empty() {
                self.parse_gpus(&self.gpus[0])
            } else {
                [0].into()
            };

            let max_steps = if i < self.max_steps.len() {
                self.max_steps[i]
            } else if !self.max_steps.is_empty() {
                self.max_steps[0]
            } else {
                1000
            };

            configs.insert(name.clone(), ModelConfig {
                path: path.clone(),
                gpus,
                max_steps,
            });
        }

        configs
    }

    fn parse_gpus(&self, devices: &str) -> Box<[c_int]> {
        static NUM_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
        NUM_REGEX
            .find_iter(devices)
            .map(|c| c.as_str().parse().unwrap())
            .collect()
    }

    fn get_default_model(&self) -> Option<ModelConfig> {
        let configs = self.get_model_configs();
        configs.values().next().cloned()
    }
}

mod macros {
    macro_rules! print_now {
        ($($arg:tt)*) => {{
            use std::io::Write;

            print!($($arg)*);
            std::io::stdout().flush().unwrap();
        }};
    }

    pub(crate) use print_now;
}
