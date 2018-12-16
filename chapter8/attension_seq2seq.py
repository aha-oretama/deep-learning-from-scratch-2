# coding: utf-8
import sys

sys.path.append('..')

from chapter7.Encoder import Encoder
from chapter8.attention_layer import TimeAttention
from chapter7.Seq2seq import Seq2seq
from common.time_layer import *


class AttentionEncoder(Encoder):

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(2 * H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, hs_enc):
        self.lstm.set_state(hs_enc[:, -1])

        out = self.embed.forward(xs)
        hs_dec = self.lstm.forward(out)
        c = self.attention.forward(hs_enc, hs_dec)
        out = np.concatenate((c, hs_dec), axis=2)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, dhs0_dec = dout[:, :, :H], dout[:, :, H:]
        dhs_enc, dhs1_dec = self.attention.backward(dc)
        dhs_dec = dhs0_dec + dhs1_dec
        dout = self.lstm.backward(dhs_dec)
        dh = self.lstm.dh
        dhs_enc[:, -1] += dh
        self.embed.backward(dout)

        return dhs_enc

    def generate(self, hs_enc, start_id, sample_size):
        sampled = []
        sampled_id = start_id
        h = hs_enc[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sampled_id]).reshape((1, 1))

            out = self.embed.forward(x)
            hs_dec = self.lstm.forward(out)
            c = self.attention.forward(hs_enc, hs_dec)
            out = np.concatenate((c, hs_dec), axis=2)
            score = self.affine.forward(out)

            sampled_id = np.argmax(score.flatten())
            sampled.append(int(sampled_id))

        return sampled


class AttentionSeq2seq(Seq2seq):

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
