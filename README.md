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

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.
