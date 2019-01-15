use std::thread;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::{Future, Stream};
use grpcio::{RpcContext, Environment, Server, ServerBuilder, ChannelBuilder, RequestStream, ClientStreamingSink};
use fasttext::FastText;

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
    fn predict(&mut self, ctx: RpcContext, stream: RequestStream<PredictRequest>, sink: ClientStreamingSink<PredictResponse>) {
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
                grpcio::Error::RemoteStopped => {},
                _ => error!("Failed to predict: {:?}", e),
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
        .max_receive_message_len(20_971_520)  // 20MB
        .max_send_message_len(10_485_760)  // 10MB
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
        info!("Listening on {}:{}", host, port);
    }
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }
    let _ = server.shutdown().wait();
}