FROM ubuntu:20.04

COPY target/release/fasttext-serving /usr/bin/fasttext-serving

ENTRYPOINT ["/usr/bin/fasttext-serving"]
