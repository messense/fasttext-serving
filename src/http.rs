use std::fmt;
use std::io;
use std::str::FromStr;

use actix_web::rt::System;
use actix_web::{web, App, FromRequest, HttpServer};
use fasttext::FastText;
use serde::Deserialize;

use crate::predict_one;

const UNIX_PREFIX: &'static str = "unix:";

enum Address {
    IpPort(String, u16),
    Unix(String),
}

impl From<(&str, u16)> for Address {
    fn from(addr: (&str, u16)) -> Self {
        addr.0
            .parse::<Address>()
            .unwrap_or_else(|_| Address::IpPort(addr.0.to_string(), addr.1))
    }
}

impl FromStr for Address {
    type Err = io::Error;

    fn from_str(string: &str) -> io::Result<Self> {
        #[cfg(unix)]
        {
            if string.starts_with(UNIX_PREFIX) {
                let address = &string[UNIX_PREFIX.len()..];
                return Ok(Address::Unix(address.into()));
            }
        }
        Err(io::Error::new(
            io::ErrorKind::Other,
            "failed to resolve TCP address",
        ))
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Address::IpPort(ip, port) => write!(f, "{}:{}", ip, port),
            Address::Unix(path) => write!(f, "{}{}", UNIX_PREFIX, path),
        }
    }
}

#[derive(Deserialize, Debug, Default)]
struct PredictOptions {
    k: Option<u32>,
    threshold: Option<f32>,
}

async fn predict(
    model: web::Data<FastText>,
    texts: web::Json<Vec<String>>,
    options: web::Query<PredictOptions>,
) -> web::Json<Vec<(Vec<String>, Vec<f32>)>> {
    let k = options.k.unwrap_or(1);
    let threshold = options.threshold.unwrap_or(0.0);
    let text_count = texts.len();
    let ret: Vec<(Vec<String>, Vec<f32>)> = match text_count {
        0 => Vec::new(),
        _ => texts
            .iter()
            .map(|txt| predict_one(model.get_ref(), txt, k, threshold))
            .collect(),
    };
    web::Json(ret)
}

async fn sentence_vector(
    model: web::Data<FastText>,
    texts: web::Json<Vec<String>>,
) -> web::Json<Vec<Vec<f32>>> {
    let text_count = texts.len();
    let ret: Vec<Vec<f32>> = match text_count {
        0 => Vec::new(),
        _ => texts
            .iter()
            .map(|txt| model.get_sentence_vector(txt))
            .collect(),
    };
    web::Json(ret)
}

pub(crate) fn runserver(model: FastText, address: &str, port: u16, workers: usize) {
    let addr = Address::from((address, port));
    log::info!("Listening on {}", addr);
    let model_data = web::Data::new(model);
    let mut server = HttpServer::new(move || {
        App::new()
            .service(
                web::resource("/predict")
                    .app_data(model_data.clone())
                    .app_data(web::Json::<Vec<String>>::configure(|cfg| {
                        cfg.limit(20_971_520) // 20MB
                            .content_type(|_mime| true) // Accept any content type
                    }))
                    .route(web::post().to(predict)),
            )
            .service(
                web::resource("/sentence-vector")
                    .app_data(model_data.clone())
                    .app_data(web::Json::<Vec<String>>::configure(|cfg| {
                        cfg.limit(20_971_520) // 20MB
                            .content_type(|_mime| true) // Accept any content type
                    }))
                    .route(web::post().to(sentence_vector)),
            )
    })
    .workers(workers);

    let sys = System::new();
    server = match addr {
        Address::IpPort(address, port) => server.bind((&address[..], port)).expect("bind failed"),
        Address::Unix(path) => {
            #[cfg(unix)]
            {
                server.bind_uds(path).expect("bind failed")
            }
            #[cfg(not(unix))]
            {
                panic!("Unix domain socket is not supported on this platform")
            }
        }
    };
    sys.block_on(async { server.run().await }).unwrap();
}

#[cfg(test)]
mod test {
    use super::predict;
    use actix_web::http::StatusCode;
    use actix_web::test::{call_service, init_service, TestRequest};
    use actix_web::{web, App};
    use fasttext::FastText;

    #[actix_rt::test]
    async fn test_predict_empty_input() {
        let mut fasttext = FastText::new();
        fasttext
            .load_model("models/cooking.model.bin")
            .expect("Failed to load fastText model");
        let model_data = web::Data::new(fasttext);
        let mut srv = init_service(
            App::new()
                .app_data(model_data)
                .service(web::resource("/predict").route(web::post().to(predict))),
        )
        .await;
        let data: Vec<String> = Vec::new();
        let req = TestRequest::post()
            .uri("/predict")
            .set_json(&data)
            .to_request();
        let resp = call_service(&mut srv, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[actix_rt::test]
    async fn test_predict() {
        let mut fasttext = FastText::new();
        fasttext
            .load_model("models/cooking.model.bin")
            .expect("Failed to load fastText model");
        let model_data = web::Data::new(fasttext);
        let mut srv = init_service(
            App::new()
                .app_data(model_data)
                .service(web::resource("/predict").route(web::post().to(predict))),
        )
        .await;
        let data = vec!["Which baking dish is best to bake a banana bread?"];
        let req = TestRequest::post()
            .uri("/predict")
            .set_json(&data)
            .to_request();
        let resp = call_service(&mut srv, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
