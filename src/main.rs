#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use] extern crate rocket;
extern crate rocket_contrib;
extern crate clap;
extern crate rayon;
extern crate fasttext;

use std::env;
use std::path::Path;
use rayon::prelude::*;
use rocket::State;
use rocket::request::Form;
use rocket::fairing::AdHoc;
use rocket_contrib::json::Json;
use clap::{App, Arg};
use fasttext::FastText;

#[derive(FromForm, Debug, Default)]
struct PredictOptions {
    k: Option<u32>,
    threshold: Option<f32>,
}

struct RocketWorkerCount(u16);

#[inline]
fn predict_one(model: &FastText, text: &str, k: u32, threshold: f32) -> (Vec<String>, Vec<f32>) {
    // Ensure k >= 1
    let k = if k > 0 { k } else { 1 };
    // NOTE: text needs to end in a newline
    // to exactly mimic the behavior of the cli
    let preds = if text.ends_with('\n') {
        model.predict(text, k as i32, threshold)
    } else {
        let mut text = text.to_string();
        text.push('\n');
        model.predict(&text, k as i32, threshold)
    };
    let mut labels = Vec::with_capacity(preds.len());
    let mut probs = Vec::with_capacity(preds.len());
    for pred in &preds {
        labels.push(pred.label.trim_left_matches("__label__").to_string());
        probs.push(pred.prob);
    }
    (labels, probs)
}

#[post("/predict?<options..>", data = "<texts>")]
fn predict(worker_count: State<RocketWorkerCount>, model: State<FastText>, texts: Json<Vec<String>>, options: Form<PredictOptions>)
    -> Json<Vec<(Vec<String>, Vec<f32>)>>
{
    let k = options.k.unwrap_or(1);
    let threshold = options.threshold.unwrap_or(0.0);
    let text_count = texts.len();
    let worker_count = worker_count.inner().0 as usize;
    let ret: Vec<(Vec<String>, Vec<f32>)> = match text_count {
        0 => Vec::new(),
        1 => vec![predict_one(model.inner(), &texts[0], k, threshold)],
        n if n > 1 && n <= worker_count => {
            texts.par_iter().map(|txt| {
                predict_one(model.inner(), txt, k, threshold)
            }).collect()
        },
        _ => {
            texts.iter().map(|txt| {
                predict_one(model.inner(), txt, k, threshold)
            }).collect()
        }
    };
    Json(ret)
}

fn main() {
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
                 .takes_value(true)
                 .help("Worker thread count, defaults to CPU count"))
        .get_matches();
    let model_path = matches.value_of("model").unwrap();
    if !Path::new(model_path).exists() {
        panic!(format!("Error: model {} does not exists", model_path));
    }
    if env::var("ROCKET_ENV").is_err() {
        env::set_var("ROCKET_ENV", "prod");
    }
    if let Some(address) = matches.value_of("address") {
        env::set_var("ROCKET_ADDRESS", address);
    }
    if let Some(port) = matches.value_of("port") {
        env::set_var("ROCKET_PORT", port);
    }
    if let Some(workers) = matches.value_of("workers") {
        env::set_var("ROCKET_WORKERS", workers);
    }
    let mut fasttext = FastText::new();
    fasttext.load_model(model_path);
    rocket::ignite()
        .manage(fasttext)
        .attach(AdHoc::on_attach("rocket-worker-count", |rocket| {
            let workers = rocket.config().workers;
            Ok(rocket.manage(RocketWorkerCount(workers)))
        }))
        .mount("/", routes![predict])
        .launch();
}
