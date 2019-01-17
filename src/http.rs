use rocket::{State, post, routes, FromForm};
use rocket::request::Form;
use rocket::fairing::AdHoc;
use rocket_contrib::json::Json;
use rayon::prelude::*;
use fasttext::FastText;

use crate::predict_one;

#[derive(FromForm, Debug, Default)]
struct PredictOptions {
    k: Option<u32>,
    threshold: Option<f32>,
}

struct RocketWorkerCount(u16);

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

pub(crate) fn server(model: FastText) -> rocket::Rocket {
    rocket::ignite()
        .manage(model)
        .attach(AdHoc::on_attach("rocket-worker-count", |rocket| {
            let workers = rocket.config().workers;
            Ok(rocket.manage(RocketWorkerCount(workers)))
        }))
        .mount("/", routes![predict])
}

#[cfg(test)]
mod test {
    use rocket::local::Client;
    use rocket::http::{Status, ContentType};
    use fasttext::FastText;
    use super::server;

    #[test]
    fn test_predict_empty_input() {
        let mut fasttext = FastText::new();
        fasttext.load_model("models/cooking.model.bin").expect("Failed to load fastText model");
        let client = Client::new(server(fasttext)).unwrap();
        let mut res = client.post("/predict")
            .header(ContentType::JSON)
            .body(r#"[]"#)
            .dispatch();
        assert_eq!(res.status(), Status::Ok);
        let body = res.body().unwrap().into_string().unwrap();
        assert_eq!("[]", body);
    }

    #[test]
    fn test_predict() {
        let mut fasttext = FastText::new();
        fasttext.load_model("models/cooking.model.bin").expect("Failed to load fastText model");
        let client = Client::new(server(fasttext)).unwrap();
        let mut res = client.post("/predict")
            .header(ContentType::JSON)
            .body(r#"["Which baking dish is best to bake a banana bread?"]"#)
            .dispatch();
        assert_eq!(res.status(), Status::Ok);
        let body = res.body().unwrap().into_string().unwrap();
        assert!(body.contains("baking"));
    }
}
