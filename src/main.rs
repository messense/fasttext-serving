#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use] extern crate rocket;
extern crate rocket_contrib;
extern crate clap;
extern crate rayon;
extern crate fasttext;
extern crate futures;
extern crate protobuf;
extern crate grpcio;
extern crate grpcio_proto;
extern crate pretty_env_logger;
#[macro_use] extern crate log;
extern crate ctrlc;
extern crate num_cpus;

use std::env;
use std::path::Path;
use clap::{App, Arg};
use fasttext::FastText;

mod http;
mod grpc;

#[inline]
pub fn predict_one(model: &FastText, text: &str, k: u32, threshold: f32) -> (Vec<String>, Vec<f32>) {
    // Ensure k >= 1
    let k = if k > 0 { k } else { 1 };
    // NOTE: text needs to end in a newline
    // to exactly mimic the behavior of the cli
    let preds = if text.ends_with('\n') {
        model.predict(text, k as i32, threshold).expect("predict failed")
    } else {
        let mut text = text.to_string();
        text.push('\n');
        model.predict(&text, k as i32, threshold).expect("predict failed")
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
    let matches = App::new("fasttext-serving")
        .version(env!("CARGO_PKG_VERSION"))
        .about("fastText model serving service")
        .author("Messense Lv <messense@icloud.com>")
        .arg(Arg::with_name("model")
                 .required(true)
                 .short("m")
                 .long("model")
                 .value_name("model")
                 .takes_value(true)
                 .help("Model path"))
        .arg(Arg::with_name("address")
                 .short("a")
                 .long("address")
                 .default_value("127.0.0.1")
                 .takes_value(true)
                 .help("Listen address"))
        .arg(Arg::with_name("port")
                 .short("p")
                 .long("port")
                 .default_value("8000")
                 .takes_value(true)
                 .help("Listen port"))
        .arg(Arg::with_name("workers")
                 .short("w")
                 .long("workers")
                 .alias("concurrency")
                 .alias("threads")
                 .default_value(&num_threads)
                 .takes_value(true)
                 .help("Worker thread count, defaults to CPU count"))
        .arg(Arg::with_name("grpc")
                 .long("grpc")
                 .help("Serving gRPC API instead of HTTP API"))
        .get_matches();
    let model_path = matches.value_of("model").unwrap();
    if !Path::new(model_path).exists() {
        panic!(format!("Error: model {} does not exists", model_path));
    }
    let address = matches.value_of("address").expect("missing address");
    let port = matches.value_of("port").expect("missing port");
    let workers = matches.value_of("workers").expect("missing workers");
    let mut model = FastText::new();
    model.load_model(model_path).expect("Failed to load fastText model");
    if !matches.is_present("grpc") {
        if env::var("ROCKET_ENV").is_err() {
            env::set_var("ROCKET_ENV", "prod");
        }
        env::set_var("ROCKET_ADDRESS", address);
        env::set_var("ROCKET_PORT", port);
        env::set_var("ROCKET_WORKERS", workers);
        crate::http::server(model).launch();
    } else {
        let port: u16 = port.parse().expect("invalid port");
        let workers: usize = workers.parse().expect("invalid workers");
        crate::grpc::runserver(model, address, port, workers);
    }
}
