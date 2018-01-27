# fasttext-serving

[![Build Status](https://travis-ci.org/messense/fasttext-serving.svg?branch=master)](https://travis-ci.org/messense/fasttext-serving)
[![Crates.io](https://img.shields.io/crates/v/fasttext-serving.svg)](https://crates.io/crates/fasttext-serving)

fastText model serving service

## Installation

You can download prebuilt binary from [GitHub releases](https://github.com/messense/fasttext-serving/releases),
or install it using Cargo:

```bash
cargo install fasttext-serving
```

## Usage

```bash
$ fasttext-serving --help

USAGE:
    fasttext-serving --model <model>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -m, --model <model>    Model path
```

Please refer to [Rocket documentation](https://rocket.rs/guide/configuration/#configuration) for configuration.
Some common environment variables:

```bash
# Rocket configuration environment, use prod for production deployment
ROCKET_ENV=prod
# Rocket listen address
ROCKET_ADDRESS=0.0.0.0
# Rocket listen port
ROCKET_PORT=8080
# Rocket worker threads
ROCKET_WORKERS=8
```

HTTP API endpoint:

```
POST /predict
```

Post data should be JSON array of string, for example `["abc", "def"]`

CURL example:

```bash
curl -X POST \
     --data "[\"Which baking dish is best to bake a banana bread?\", \"Why not put knives in the dishwasher?\"]" \
     'http://localhost:8000/predict'
```

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.
