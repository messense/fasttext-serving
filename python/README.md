# fasttext-serving

fasttext-serving gRPC client

## Installation

```bash
pip install fasttext-serving
```

## Usage


```python
from fasttext_serving import FasttextServing


client = FasttextServing('127.0.0.1:8000')
predictions = list(client.predict(['abc def']))
```
