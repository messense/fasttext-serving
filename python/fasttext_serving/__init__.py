# -*- coding: utf-8 -*-
import grpc

from .predict_pb2 import PredictRequest
from .predict_pb2_grpc import FasttextServingStub


__all__ = ['FasttextServing']


class FasttextServing(object):
    def __init__(self, host, options=None):
        options = options or [
            ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),  # 10MB
        ]
        self.channel = grpc.insecure_channel(host, options=options)
        self.stub = FasttextServingStub(self.channel)

    def predict(self, texts, k=1, threshold=0.0):
        def _generate():
            for text in texts:
                yield PredictRequest(text=text, k=k, threshold=threshold)

        resp = self.stub.predict(_generate())
        for prediction in resp.predictions:
            yield prediction.labels, prediction.probs
