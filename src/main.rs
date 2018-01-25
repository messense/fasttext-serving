#![feature(plugin)]
#![feature(custom_derive)]
#![plugin(rocket_codegen)]
extern crate rocket;
extern crate rocket_contrib;
extern crate clap;
extern crate rayon;
extern crate fasttext;

use std::path::Path;
use rayon::prelude::*;
use rocket::State;
use rocket::fairing::AdHoc;
use rocket_contrib::Json;
use clap::{App, Arg};
use fasttext::FastText;

#[derive(FromForm, Debug, Default)]
struct PredictOptions {
    k: Option<i32>,
    threshold: Option<f32>,
}

struct RocketWorkerCount(u16);

#[inline]
fn predict_one(model: &FastText, text: &str, k: i32, threshold: f32) -> (Vec<String>, Vec<f32>) {
    // NOTE: text needs to end in a newline
    // to exactly mimic the behavior of the cli
    let preds = if text.ends_with('\n') {
        model.predict(text, k, threshold)
    } else {
        let mut text = text.to_string();
        text.push('\n');
        model.predict(&text, k, threshold)
    };
    let mut labels = Vec::with_capacity(preds.len());
    let mut probs = Vec::with_capacity(preds.len());
    for pred in &preds {
        labels.push(pred.label.trim_left_matches("__label__").to_string());
        probs.push(pred.prob);
    }
    (labels, probs)
}

#[post("/predict", format = "application/json", data = "<texts>")]
fn predict_without_option(worker_count: State<RocketWorkerCount>, model: State<FastText>, texts: Json<Vec<String>>)
    -> Json<Vec<(Vec<String>, Vec<f32>)>>
{
    // XXX: https://github.com/SergioBenitez/Rocket/issues/376
    predict(worker_count, model, texts, Default::default())
}

#[post("/predict?<options>", format = "application/json", data = "<texts>")]
fn predict(worker_count: State<RocketWorkerCount>, model: State<FastText>, texts: Json<Vec<String>>, options: PredictOptions)
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
        .get_matches();
    let model_path = matches.value_of("model").unwrap();
    if !Path::new(model_path).exists() {
        panic!(format!("Error: model {} does not exists", model_path));
    }
    let mut fasttext = FastText::new();
    fasttext.load_model(model_path);
    rocket::ignite()
        .manage(fasttext)
        .attach(AdHoc::on_attach(|rocket| {
            let workers = rocket.config().workers;
            Ok(rocket.manage(RocketWorkerCount(workers)))
        }))
        .mount("/", routes![predict, predict_without_option])
        .launch();
}
