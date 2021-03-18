fn main() {
    #[cfg(feature = "grpc")]
    {
        use std::env;
        use std::path::PathBuf;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        tonic_build::configure()
            .file_descriptor_set_path(out_dir.join("fasttext_serving_descriptor.bin"))
            .compile(&["proto/fasttext_serving.proto"], &["proto"])
            .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
    }
}
