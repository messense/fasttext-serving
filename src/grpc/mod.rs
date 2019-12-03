use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use fasttext::FastText;
use futures::{Future, Stream};
use grpcio::{
    ChannelBuilder, ClientStreamingSink, Environment, RequestStream, RpcContext, Server,
    ServerBuilder,
};

mod predict;
mod predict_grpc;

use self::predict::{PredictRequest, PredictResponse, Prediction};
use self::predict_grpc::FasttextServing;
use crate::predict_one;

#[derive(Clone)]
pub struct FasttextServingService {
    model: Arc<FastText>,
}

impl FasttextServing for FasttextServingService {
    fn predict(
        &mut self,
        ctx: RpcContext,
        stream: RequestStream<PredictRequest>,
        sink: ClientStreamingSink<PredictResponse>,
    ) {
        let model = self.model.clone();
        let f = stream
            .fold(Vec::new(), move |mut preds, req| {
                let text = req.get_text();
                let k = req.get_k();
                let threshold = req.get_threshold();
                preds.push(predict_one(&model, text, k, threshold));
                Ok(preds) as grpcio::Result<_>
            })
            .and_then(move |preds| {
                let predictions: Vec<Prediction> = preds
                    .into_iter()
                    .map(|pred| {
                        let mut prediction = Prediction::new();
                        let (labels, probs) = pred;
                        prediction.set_labels(labels.into());
                        prediction.set_probs(probs);
                        prediction
                    })
                    .collect();
                let mut resp = PredictResponse::new();
                resp.set_predictions(predictions.into());
                sink.success(resp)
            })
            .map_err(|e| match e {
                grpcio::Error::RemoteStopped => {}
                _ => log::error!("Failed to predict: {:?}", e),
            });
        ctx.spawn(f)
    }
}

fn start_server(model: FastText, address: &str, port: u16, num_threads: usize) -> Server {
    let instance = FasttextServingService {
        model: Arc::new(model),
    };
    let service = self::predict_grpc::create_fasttext_serving(instance);
    let env = Arc::new(Environment::new(num_threads));
    let channel_args = ChannelBuilder::new(Arc::clone(&env))
        .max_receive_message_len(20_971_520) // 20MB
        .max_send_message_len(10_485_760) // 10MB
        .build_args();
    let mut server = ServerBuilder::new(env)
        .channel_args(channel_args)
        .register_service(service)
        .bind(address, port)
        .build()
        .unwrap();
    server.start();
    server
}

pub(crate) fn runserver(model: FastText, address: &str, port: u16, num_threads: usize) {
    let mut server = start_server(model, address, port, num_threads);
    for &(ref host, port) in server.bind_addrs() {
        log::info!("Listening on {}:{}", host, port);
    }
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }
    let _ = server.shutdown().wait();
}

#[cfg(test)]
mod test {
    use super::predict::PredictRequest;
    use super::predict_grpc::FasttextServingClient;
    use super::start_server;
    use fasttext::FastText;
    use futures::{future, Future, Sink};
    use grpcio::{ChannelBuilder, Environment, WriteFlags};
    use std::sync::Arc;

    #[test]
    fn test_grpc_predict() {
        let mut fasttext = FastText::new();
        fasttext
            .load_model("models/cooking.model.bin")
            .expect("Failed to load fastText model");
        let server = start_server(fasttext, "127.0.0.1", 0, 1);
        let port = server.bind_addrs()[0].1;
        let env = Arc::new(Environment::new(1));
        let channel = ChannelBuilder::new(env).connect(&format!("127.0.0.1:{}", port));
        let client = FasttextServingClient::new(channel);

        let (mut sink, receiver) = client.predict().unwrap();

        let mut req = PredictRequest::new();
        req.set_text("Which baking dish is best to bake a banana bread?".to_string());
        sink = sink.send((req, WriteFlags::default())).wait().unwrap();
        // flush
        future::poll_fn(|| sink.close()).wait().unwrap();
        let res = receiver.wait().unwrap();
        let preds = res.get_predictions();
        assert_eq!(1, preds.len());
        let pred = &preds[0];
        let labels = pred.get_labels();
        let probs = pred.get_probs();
        assert_eq!(1, labels.len());
        assert_eq!(1, probs.len());
        assert_eq!("baking", &labels[0]);
        assert!(*&probs[0] > 0.7);
    }
}
