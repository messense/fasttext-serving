use fasttext::FastText;
use serde::Deserialize;
use actix_web::{web, App, HttpServer, FromRequest};

use crate::predict_one;

#[derive(Deserialize, Debug, Default)]
struct PredictOptions {
    k: Option<u32>,
    threshold: Option<f32>,
}

fn predict(model: web::Data<FastText>, texts: web::Json<Vec<String>>, options: web::Query<PredictOptions>)
    -> web::Json<Vec<(Vec<String>, Vec<f32>)>>
{
    let k = options.k.unwrap_or(1);
    let threshold = options.threshold.unwrap_or(0.0);
    let text_count = texts.len();
    let ret: Vec<(Vec<String>, Vec<f32>)> = match text_count {
        0 => Vec::new(),
        _ => {
            texts.iter().map(|txt| {
                predict_one(model.get_ref(), txt, k, threshold)
            }).collect()
        }
    };
    web::Json(ret)
}

pub(crate) fn runserver(model: FastText, address: &str, port: u16, workers: usize) {
    let model_data = web::Data::new(model);
    HttpServer::new(move || {
        App::new()
            .register_data(model_data.clone())
            .service(
                web::resource("/predict")
                    .data(web::Json::<Vec<String>>::configure(|cfg| {
                        // Accpet any content type
                        cfg.content_type(|_mime| true)
                    }))
                    .route(web::post().to(predict))
            )
        })
        .workers(workers)
        .bind((address, port))
        .expect("bind failed")
        .run()
        .expect("run failed");
}

#[cfg(test)]
mod test {
    use fasttext::FastText;
    use actix_web::{web, App};
    use actix_web::http::StatusCode;
    use actix_web::test::{call_service, init_service, TestRequest};
    use super::predict;

    #[test]
    fn test_predict_empty_input() {
        let mut fasttext = FastText::new();
        fasttext.load_model("models/cooking.model.bin").expect("Failed to load fastText model");
        let model_data = web::Data::new(fasttext);
        let mut srv = init_service(
            App::new()
                .register_data(model_data)
                .service(web::resource("/predict").route(web::post().to(predict))
            )
        );
        let data: Vec<String> = Vec::new();
        let req = TestRequest::post().uri("/predict").set_json(&data).to_request();
        let resp = call_service(&mut srv, req);
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn test_predict() {
        let mut fasttext = FastText::new();
        fasttext.load_model("models/cooking.model.bin").expect("Failed to load fastText model");
        let model_data = web::Data::new(fasttext);
        let mut srv = init_service(
            App::new()
                .register_data(model_data)
                .service(web::resource("/predict").route(web::post().to(predict))
            )
        );
        let data = vec!["Which baking dish is best to bake a banana bread?"];
        let req = TestRequest::post().uri("/predict").set_json(&data).to_request();
        let resp = call_service(&mut srv, req);
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
