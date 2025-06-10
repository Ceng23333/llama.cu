mod bench;
mod chat;
mod generate;
mod logger;
mod service;

use clap::Parser;
use std::{ffi::c_int, path::PathBuf, collections::HashMap};
use llama_cu::ModelConfig;

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

impl BaseArgs {
    pub fn get_model_configs(&self) -> HashMap<String, ModelConfig> {
        let mut configs = HashMap::new();
        let model_name = if self.model_name.len() < self.model_path.len() {
            let mut names = self.model_name.clone();
            names.extend((names.len()..self.model_path.len()).map(|i| format!("model_{i}")));
            names
        } else {
            self.model_name[..self.model_path.len()].to_vec()
        };

        let gpus = if self.gpus.len() < self.model_path.len() {
            let mut gpus = self.gpus.clone();
            gpus.extend(std::iter::repeat("0".to_string()).take(self.model_path.len() - gpus.len()));
            gpus
        } else {
            self.gpus[..self.model_path.len()].to_vec()
        };

        let max_steps = if self.max_steps.len() < self.model_path.len() {
            let mut steps = self.max_steps.clone();
            steps.extend(std::iter::repeat(512).take(self.model_path.len() - steps.len()));
            steps
        } else {
            self.max_steps[..self.model_path.len()].to_vec()
        };
        for (((path, name), gpus), max_steps) in self.model_path.iter()
            .zip(model_name.iter())
            .zip(gpus.iter())
            .zip(max_steps.iter()) {
            let name = name.clone();
            let gpus: Box<[c_int]> = gpus.split(',')
                .map(|s| s.parse().unwrap())
                .collect();
            configs.insert(name, (path.clone(), gpus, *max_steps));
        }
        configs
    }

    fn get_default_model(&self) -> Option<(PathBuf, Box<[c_int]>, usize)> {
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
