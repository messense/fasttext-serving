fn main() {
    #[cfg(feature = "grpc")]
    {
        tonic_build::compile_protos("proto/fasttext_serving.proto")
            .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
    }
}
