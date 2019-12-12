use std::net::ToSocketAddrs;
use std::sync::Arc;

use fasttext::FastText;
use futures::StreamExt;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

use crate::predict_one;

#[allow(non_camel_case_types)]
mod fasttext_serving {
    tonic::include_proto!("fasttext_serving");
}

use fasttext_serving::{
    fasttextserving_server as server, PredictRequest, PredictResponse, Prediction,
};

#[derive(Debug, Clone)]
struct FastTextServingService {
    model: Arc<FastText>,
}

#[tonic::async_trait]
impl server::FasttextServing for FastTextServingService {
    async fn predict(
        &self,
        request: Request<Streaming<PredictRequest>>,
    ) -> Result<Response<PredictResponse>, Status> {
        let stream = request.into_inner();
        futures::pin_mut!(stream);
        let model = self.model.clone();
        let mut predictions = Vec::new();
        while let Some(req) = stream.next().await {
            let req = req?;
            let text = req.text;
            let k = req.k.unwrap_or(1);
            let threshold = req.threshold.unwrap_or(0.0);
            let (labels, probs) = predict_one(&model, &text, k, threshold);
            predictions.push(Prediction { labels, probs });
        }
        Ok(Response::new(PredictResponse { predictions }))
    }
}

pub(crate) fn runserver(model: FastText, address: &str, port: u16, num_threads: usize) {
    let instance = FastTextServingService {
        model: Arc::new(model),
    };
    let service = server::FasttextServingServer::new(instance);
    let addr = (address, port).to_socket_addrs().unwrap().next().unwrap();
    let server = Server::builder().add_service(service);
    log::info!("Listening on {}:{}", address, port);
    tokio::runtime::Builder::new()
        .enable_all()
        .threaded_scheduler()
        .num_threads(num_threads)
        .build()
        .unwrap()
        .block_on(async {
            server.serve(addr).await.unwrap();
        });
}

// FIXME: add test case
