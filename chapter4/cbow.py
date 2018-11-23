# coding: utf-8
import sys
import numpy as np

sys.path.append('..')
from common.layers import Embedding
from chapter4.negative_sampling_layer import NegativeSamplingLoss


class CBOW:

    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        self.in_layers = []
        for i in range(window_size):
            self.in_layers.append(Embedding(W_in))

        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # すべての重みと勾配を配列にまとめる
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)

        return loss

    def backward(self, dout=1):
        dh = self.ns_loss.backward(dout)
        dh *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dh)

        return None
