# coding: utf-8
import sys

sys.path.append('..')

from chapter7.Seq2seq import Seq2seq
from chapter7.Encoder import Encoder
from chapter7.PeekyDecoder import PeekyDecoder
from common.time_layer import TimeSoftmaxWithLoss


class PeekySeq2seq(Seq2seq):

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
