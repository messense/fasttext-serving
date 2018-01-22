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
use rocket_contrib::Json;
use clap::{App, Arg};
use fasttext::FastText;

#[derive(FromForm, Debug, Default)]
struct PredictOptions {
    k: Option<i32>,
    threshold: Option<f32>,
}

#[post("/predict", format = "application/json", data = "<texts>")]
fn predict_without_option(model: State<FastText>, texts: Json<Vec<String>>) -> Json<Vec<(Vec<String>, Vec<f32>)>> {
    // XXX: https://github.com/SergioBenitez/Rocket/issues/376
    predict(model, texts, Default::default())
}

#[post("/predict?<options>", format = "application/json", data = "<texts>")]
fn predict(model: State<FastText>, texts: Json<Vec<String>>, options: PredictOptions) -> Json<Vec<(Vec<String>, Vec<f32>)>> {
    let k = options.k.unwrap_or(1);
    let threshold = options.threshold.unwrap_or(0.0);
    let ret: Vec<(Vec<String>, Vec<f32>)> = texts.par_iter().map(|txt| {
        let preds = model.predict(txt, k, threshold);
        let mut labels = Vec::with_capacity(preds.len());
        let mut probs = Vec::with_capacity(preds.len());
        for pred in &preds {
            labels.push(pred.label.trim_left_matches("__label__").to_string());
            probs.push(pred.prob);
        }
        (labels, probs)
    }).collect();
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
        .mount("/", routes![predict, predict_without_option])
        .launch();
}
