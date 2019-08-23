// This file is generated. Do not edit
// @generated

// https://github.com/Manishearth/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy::all)]

#![cfg_attr(rustfmt, rustfmt_skip)]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unsafe_code)]
#![allow(unused_imports)]
#![allow(unused_results)]

const METHOD_FASTTEXT_SERVING_PREDICT: ::grpcio::Method<super::predict::PredictRequest, super::predict::PredictResponse> = ::grpcio::Method {
    ty: ::grpcio::MethodType::ClientStreaming,
    name: "/fasttext_serving.FasttextServing/predict",
    req_mar: ::grpcio::Marshaller { ser: ::grpcio::pb_ser, de: ::grpcio::pb_de },
    resp_mar: ::grpcio::Marshaller { ser: ::grpcio::pb_ser, de: ::grpcio::pb_de },
};

#[derive(Clone)]
pub struct FasttextServingClient {
    client: ::grpcio::Client,
}

impl FasttextServingClient {
    pub fn new(channel: ::grpcio::Channel) -> Self {
        FasttextServingClient {
            client: ::grpcio::Client::new(channel),
        }
    }

    pub fn predict_opt(&self, opt: ::grpcio::CallOption) -> ::grpcio::Result<(::grpcio::ClientCStreamSender<super::predict::PredictRequest>, ::grpcio::ClientCStreamReceiver<super::predict::PredictResponse>)> {
        self.client.client_streaming(&METHOD_FASTTEXT_SERVING_PREDICT, opt)
    }

    pub fn predict(&self) -> ::grpcio::Result<(::grpcio::ClientCStreamSender<super::predict::PredictRequest>, ::grpcio::ClientCStreamReceiver<super::predict::PredictResponse>)> {
        self.predict_opt(::grpcio::CallOption::default())
    }
    pub fn spawn<F>(&self, f: F) where F: ::futures::Future<Item = (), Error = ()> + Send + 'static {
        self.client.spawn(f)
    }
}

pub trait FasttextServing {
    fn predict(&mut self, ctx: ::grpcio::RpcContext, stream: ::grpcio::RequestStream<super::predict::PredictRequest>, sink: ::grpcio::ClientStreamingSink<super::predict::PredictResponse>);
}

pub fn create_fasttext_serving<S: FasttextServing + Send + Clone + 'static>(s: S) -> ::grpcio::Service {
    let mut builder = ::grpcio::ServiceBuilder::new();
    let mut instance = s;
    builder = builder.add_client_streaming_handler(&METHOD_FASTTEXT_SERVING_PREDICT, move |ctx, req, resp| {
        instance.predict(ctx, req, resp)
    });
    builder.build()
}
