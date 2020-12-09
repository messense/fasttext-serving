# fasttext-serving

[![GitHub Actions](https://github.com/messense/fasttext-serving/workflows/CI/badge.svg)](https://github.com/messense/fasttext-serving/actions?query=workflow%3ACI)
[![Crates.io](https://img.shields.io/crates/v/fasttext-serving.svg)](https://crates.io/crates/fasttext-serving)
[![Docker Pulls](https://img.shields.io/docker/pulls/messense/fasttext-serving)](https://hub.docker.com/r/messense/fasttext-serving)

fastText model serving service

## Installation

You can download prebuilt binary from [GitHub releases](https://github.com/messense/fasttext-serving/releases),
or install it using Cargo:

```bash
cargo install fasttext-serving
```

Using Docker:

```bash
docker pull messense/fasttext-serving
```

## Usage

```bash
$ fasttext-serving --help

USAGE:
    fasttext-serving [OPTIONS] --model <model>

FLAGS:
        --grpc       Serving gRPC API instead of HTTP API
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -a, --address <address>    Listen address [default: 127.0.0.1]
    -m, --model <model>        Model path
    -p, --port <port>          Listen port [default: 8000]
    -w, --workers <workers>    Worker thread count, defaults to CPU count
```

### Serve HTTP REST API

HTTP API endpoint:

```
POST /predict
```

Post data should be JSON array of string, for example `["abc", "def"]`

CURL example:

```bash
$ curl -X POST -H 'Content-Type: application/json' \
     --data "[\"Which baking dish is best to bake a banana bread?\", \"Why not put knives in the dishwasher?\"]" \
     'http://localhost:8000/predict'
[[["baking"],[0.7152988]],[["equipment"],[0.73479545]]]
```

### Serve gRPC API

Run the command with `--grpc` to serve gRPC API instead of HTTP REST API.

Please refer to gRPC Python client documentation [here](./python).

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.
