use clap::{Arg, ArgAction, Command};
use fasttext::FastText;
use std::env;
use std::path::Path;

#[cfg(feature = "grpc")]
mod grpc;
#[cfg(feature = "http")]
mod http;

#[cfg(all(unix, not(target_env = "musl"), not(target_arch = "aarch64")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(windows)]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[inline]
pub fn predict_one(
    model: &FastText,
    text: &str,
    k: u32,
    threshold: f32,
) -> (Vec<String>, Vec<f32>) {
    // Ensure k >= 1
    let k = if k > 0 { k } else { 1 };
    // NOTE: text needs to end in a newline
    // to exactly mimic the behavior of the cli
    let preds = if text.ends_with('\n') {
        model
            .predict(text, k as i32, threshold)
            .expect("predict failed")
    } else {
        let mut text = text.to_string();
        text.push('\n');
        model
            .predict(&text, k as i32, threshold)
            .expect("predict failed")
    };
    let mut labels = Vec::with_capacity(preds.len());
    let mut probs = Vec::with_capacity(preds.len());
    for pred in &preds {
        labels.push(pred.label.trim_start_matches("__label__").to_string());
        probs.push(pred.prob);
    }
    (labels, probs)
}

fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "fasttext_serving=info");
    }
    pretty_env_logger::init();

    let num_threads = num_cpus::get().to_string();
    let matches = Command::new("fasttext-serving")
        .version(env!("CARGO_PKG_VERSION"))
        .about("fastText model serving service")
        .author("Messense Lv <messense@icloud.com>")
        .arg(
            Arg::new("model")
                .required(true)
                .short('m')
                .long("model")
                .value_name("model")
                .num_args(1)
                .help("Model path"),
        )
        .arg(
            Arg::new("address")
                .short('a')
                .long("address")
                .default_value("127.0.0.1")
                .num_args(1)
                .help("Listen address"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .default_value("8000")
                .num_args(1)
                .help("Listen port"),
        )
        .arg(
            Arg::new("workers")
                .short('w')
                .long("workers")
                .alias("concurrency")
                .alias("threads")
                .default_value(&num_threads)
                .num_args(1)
                .help("Worker thread count, defaults to CPU count"),
        )
        .arg(
            Arg::new("grpc")
                .long("grpc")
                .action(ArgAction::SetTrue)
                .help("Serving gRPC API instead of HTTP API"),
        )
        .get_matches();
    let model_path = matches.get_one::<String>("model").unwrap();
    if !Path::new(model_path).exists() {
        panic!("Error: model {} does not exists", model_path);
    }
    let address = matches
        .get_one::<String>("address")
        .expect("missing address");
    let port = matches.get_one::<String>("port").expect("missing port");
    let workers = matches
        .get_one::<String>("workers")
        .expect("missing workers");
    let mut model = FastText::new();
    model
        .load_model(model_path)
        .expect("Failed to load fastText model");
    let port: u16 = port.parse().expect("invalid port");
    let workers: usize = workers.parse().expect("invalid workers");
    if matches.get_flag("grpc") {
        #[cfg(feature = "grpc")]
        crate::grpc::runserver(model, address, port, workers);
        #[cfg(not(feature = "grpc"))]
        panic!("gRPC support is not enabled!")
    } else {
        #[cfg(feature = "http")]
        crate::http::runserver(model, address, port, workers);
        #[cfg(not(feature = "http"))]
        panic!("HTTP support is not enabled!")
    }
}
