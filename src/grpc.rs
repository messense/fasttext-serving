use std::net::ToSocketAddrs;
use std::sync::Arc;

use fasttext::FastText;
use futures::StreamExt;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

use crate::predict_one;

#[allow(non_camel_case_types)]
mod proto {
    tonic::include_proto!("fasttext_serving");

    pub(crate) const FILE_DESCRIPTOR_SET: &'static [u8] =
        tonic::include_file_descriptor_set!("fasttext_serving_descriptor");
}

use proto::{
    fasttext_serving_server as server, PredictRequest, PredictResponse, Prediction, SentenceVector,
    SentenceVectorRequest, SentenceVectorResponse,
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
            let k = req.k.unwrap_or(5);
            let threshold = req.threshold.unwrap_or(0.0);
            let (labels, probs) = predict_one(&model, &text, k, threshold);
            predictions.push(Prediction { labels, probs });
        }
        Ok(Response::new(PredictResponse { predictions }))
    }

    async fn sentence_vector(
        &self,
        request: Request<Streaming<SentenceVectorRequest>>,
    ) -> Result<Response<SentenceVectorResponse>, Status> {
        let stream = request.into_inner();
        futures::pin_mut!(stream);
        let mut vectors = Vec::new();
        let model = self.model.clone();
        while let Some(req) = stream.next().await {
            let req = req?;
            let text = req.text;
            let values = model
                .get_sentence_vector(&text)
                .expect("get_sentence_vector failed");
            vectors.push(SentenceVector { values });
        }
        Ok(Response::new(SentenceVectorResponse { vectors }))
    }
}

pub(crate) fn runserver(model: FastText, address: &str, port: u16, num_threads: usize) {
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
        .build()
        .unwrap();
    let instance = FastTextServingService {
        model: Arc::new(model),
    };
    let service = server::FasttextServingServer::new(instance);
    let addr = (address, port).to_socket_addrs().unwrap().next().unwrap();
    let server = Server::builder()
        .add_service(reflection_service)
        .add_service(service);
    log::info!("Listening on {}:{}", address, port);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(num_threads)
        .build()
        .unwrap()
        .block_on(async {
            server.serve(addr).await.unwrap();
        });
}

// FIXME: add test case
