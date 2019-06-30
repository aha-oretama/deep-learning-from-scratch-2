6章　ゲート付きRNN（後半）
====

<ゴール>  
- LSTMの仕組みを実装し、単純なRNNとの違いを理解する。  
- LSTMを使用した言語モデルを作成する。  

## 6.3　LSTMの実装
- LSTMの1ステップを処理するクラスを**LSTMクラス**として実装する。
- Tステップ分をまとめて処理するクラスを**TimeLSTMクラス**として実装する。

LSTMクラスで行う計算は以下の通り。  
- ４つの重みの計算  
  - f：忘却ゲート  
  - g：新たに記憶セルに追加する情報  
  - i：入力ゲート  
  - o：出力ゲート  

![](images/6-3-01.PNG)  

- 記憶セルの計算  
![](images/6-3-02.PNG)  

- 隠れ状態の計算  
![](images/6-3-03.PNG)  

LSTMの計算グラフ  
![](images/6-3-04.PNG)  
https://colah.github.io/posts/2015-08-Understanding-LSTMs/  

４つのアフィン変換（xW<sub>x</sub>+hW<sub>h</sub>+b）は、ひとつの式でまとめて計算することが出来る。  

LSTMクラスの初期化  
```python
    class LSTM:
        def __init__(self, Wx, Wh, b):
            '''
            Parameters
            ----------
            Wx: 入力`x`用の重みパラーメタ（4つ分の重みをまとめる）
            Wh: 隠れ状態`h`用の重みパラメータ（4つ分の重みをまとめる）
            b: バイアス（4つ分のバイアスをまとめる）
            '''
            self.params = [Wx, Wh, b]
            self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
            self.cache = None
```

順伝播の実装
```python
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        # N:バッチ数、H:記憶セルと隠れ状態の次元数
        N, H = h_prev.shape

        # ４つのパラメータをまとめて計算する
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        #スライスして取り出す
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        # 順伝播での中間結果を保持し、逆伝播の計算で使用する
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
```

逆伝播の実装
```python
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        # 配列を横方向に連結する
        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev
```


### 6.3.1　TimeLSTMの実装
TimeLSTMは、T個分の時系列データをまとめて処理するレイヤ。

RNNで学習を行う際は、Truncated BPTTを行う。
- 逆伝播のつながりを適当な長さで断ち切る。
- 順伝播の流れは維持する。
  - 隠れ状態と記憶セルをメンバ変数に保持させる。

```python
    class TimeLSTM:
        def __init__(self, Wx, Wh, b, stateful=False):
            self.params = [Wx, Wh, b]
            self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
            self.layers = None

            self.h, self.c = None, None
            self.dh = None
            self.stateful = stateful

        def forward(self, xs):
            Wx, Wh, b = self.params
            N, T, D = xs.shape
            H = Wh.shape[0]

            self.layers = []
            hs = np.empty((N, T, H), dtype='f')

            if not self.stateful or self.h is None:
                self.h = np.zeros((N, H), dtype='f')
            if not self.stateful or self.c is None:
                self.c = np.zeros((N, H), dtype='f')

            for t in range(T):
                layer = LSTM(*self.params)
                self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
                hs[:, t, :] = self.h

                self.layers.append(layer)

            return hs

        def backward(self, dhs):
            Wx, Wh, b = self.params
            N, T, H = dhs.shape
            D = Wx.shape[0]

            dxs = np.empty((N, T, D), dtype='f')
            dh, dc = 0, 0

            grads = [0, 0, 0]
            for t in reversed(range(T)):
                layer = self.layers[t]
                dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
                dxs[:, t, :] = dx
                for i, grad in enumerate(layer.grads):
                    grads[i] += grad

            for i, grad in enumerate(grads):
                self.grads[i][...] = grad
            self.dh = dh
            return dxs

        def set_state(self, h, c=None):
            self.h, self.c = h, c

        def reset_state(self):
            self.h, self.c = None, None
```


## 6.4　LSTMを使った言語モデル
- 5章で実装した「RNNを使った言語モデル」とほとんど同じ。
- Time RNNレイヤを Time LSTMレイヤに変える。  

![](images/6-4-01.PNG)  

Rnnlmクラスの実装  
```python
    class Rnnlm(BaseModel):
        def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
            V, D, H = vocab_size, wordvec_size, hidden_size
            rn = np.random.randn

            # 重みの初期化
            embed_W = (rn(V, D) / 100).astype('f')
            lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
            lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
            lstm_b = np.zeros(4 * H).astype('f')
            affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
            affine_b = np.zeros(V).astype('f')

            # レイヤの生成
            self.layers = [
                TimeEmbedding(embed_W),
                TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
                TimeAffine(affine_W, affine_b)
            ]
            self.loss_layer = TimeSoftmaxWithLoss()
            self.lstm_layer = self.layers[1]

            # すべての重みと勾配をリストにまとめる
            self.params, self.grads = [], []
            for layer in self.layers:
                self.params += layer.params
                self.grads += layer.grads

        def predict(self, xs):
            for layer in self.layers:
                xs = layer.forward(xs)
            return xs

        def forward(self, xs, ts):
            score = self.predict(xs)
            loss = self.loss_layer.forward(score, ts)
            return loss

        def backward(self, dout=1):
            dout = self.loss_layer.backward(dout)
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            return dout

        def reset_state(self):
            self.lstm_layer.reset_state()
```

学習のためのコード  
```python
    import sys
    sys.path.append('..')
    from common.optimizer import SGD
    from common.trainer import RnnlmTrainer
    from common.util import eval_perplexity
    from dataset import ptb
    from rnnlm import Rnnlm


    # ハイパーパラメータの設定
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
    time_size = 35  # RNNを展開するサイズ
    lr = 20.0
    max_epoch = 4
    max_grad = 0.25

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    # モデルの生成
    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 勾配クリッピングを適用して学習
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
                eval_interval=20)
    trainer.plot(ylim=(0, 500))

    # テストデータで評価
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print('test perplexity: ', ppl_test)

    # パラメータの保存
    model.save_params()
```


## 6.5　RNNLMのさらなる改善
6.4で説明したRNNLMの改善ポイント３点  
- LSTMレイヤの多層化
- Dropout
- 重み共有


### 6.5.1　LSTMレイヤの多層化
LSTMレイヤを何層も深く重ねることで、モデルの表現力が増し、複雑な依存関係を学習することが期待できる。  

![](images/6-5-1-01.PNG)  

どれだけ層を重ねるべきか？  
- 問題の複雑さや、用意された学習データの量に応じて適宜決める必要がある。
- PTBデータセットの言語モデルの場合は、LSTMの層数は2～4程度が良い結果を得られている。
- Google翻訳で使われているGNMTと呼ばれるモデルはLSTM層を8層重ねている。


### 6.5.2　Dropoutによる過学習の抑制
層を深くすることでモデルの表現力が増すが、過学習を起こしやすくなる。  
- 過学習とは、訓練データだけに対して正しい答えを出し、汎化能力が欠如した状態を指す。  

過学習を抑制する方法は？  
- 訓練データを増やす
- モデルの複雑さを減らす
- 正則化を行う（重みの値が大きくなりすぎることにペナルティを課す）
  - Dropoutも正則化の１種と考えられる。

Dropoutは、訓練時にレイヤ内のニューロンのいくつかをランダムに無視して学習を行う。  
![](images/6-5-2-01.PNG)  
http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf  

左が通常のニューラルネットワーク。右がDropoutを適用したネットワーク。

Dropoutはランダムにニューロンを無視することで、汎化性能を向上させることができる。  

RNNを使ったモデルでは、どこにDropoutレイヤを挿入すべきか？
- LSTMレイヤの時系列方向
  - 時間が進むのに比例してDropoutによるノイズが蓄積して、学習がうまく進まない。
- LSTMレイヤの深さ方向（上下方向）
  - 時間が進んでも情報が失われず、深さ方向にだけ有効に働く。  

RNNの時間軸方向の正則化を目的とした手法
- 変分ドロップアウト（Variational Dropout）  
同じ階層にあるドロップアウトでは、共通のマスクを利用する。（マスクは「データを通す/通さない」の二値のランダムパターン）

![](images/6-5-2-02.PNG)  
https://arxiv.org/pdf/1512.05287.pdf  


### 6.5.3　重み共有
Embeddingレイヤの重みとAffineレイヤの重みを共有する。

![](images/6-5-3-01.PNG)  

なぜ重み共有は有効なのか？
- 重みを共有することで、学習すべきパラメータ数を減らすことができる。
- パラメータ数が減ることで、過学習を抑制することができる。


### 6.5.4　より良いRNNLMの実装
言語モデルの改善テクニックを使ったモデルを確認する。
- LSTMレイヤの多層化
- Dropout
- 重み共有

```python
    class BetterRnnlm(BaseModel):
        '''
         LSTMレイヤを2層利用し、各層にDropoutを使うモデル
         [1]で提案されたモデルをベースとし、weight tying[2][3]を利用

         [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
         [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
         [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
        '''
        def __init__(self, vocab_size=10000, wordvec_size=650,
                     hidden_size=650, dropout_ratio=0.5):
            V, D, H = vocab_size, wordvec_size, hidden_size
            rn = np.random.randn

            embed_W = (rn(V, D) / 100).astype('f')
            lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
            lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
            lstm_b1 = np.zeros(4 * H).astype('f')
            lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
            lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
            lstm_b2 = np.zeros(4 * H).astype('f')
            affine_b = np.zeros(V).astype('f')

            # 3つの改善!
            self.layers = [
                TimeEmbedding(embed_W),
                TimeDropout(dropout_ratio),
                TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
                TimeDropout(dropout_ratio),
                TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
                TimeDropout(dropout_ratio),
                TimeAffine(embed_W.T, affine_b)  # weight tying!!
            ]
            self.loss_layer = TimeSoftmaxWithLoss()
            self.lstm_layers = [self.layers[2], self.layers[4]]
            self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

            self.params, self.grads = [], []
            for layer in self.layers:
                self.params += layer.params
                self.grads += layer.grads

        def predict(self, xs, train_flg=False):
            for layer in self.drop_layers:
                layer.train_flg = train_flg

            for layer in self.layers:
                xs = layer.forward(xs)
            return xs

        def forward(self, xs, ts, train_flg=True):
            score = self.predict(xs, train_flg)
            loss = self.loss_layer.forward(score, ts)
            return loss

        def backward(self, dout=1):
            dout = self.loss_layer.backward(dout)
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            return dout

        def reset_state(self):
            for layer in self.lstm_layers:
                layer.reset_state()
```


### 6.5.5　最先端の研究へ
PTBデータセットに対する各モデルのパープレキシティの結果
![](images/6-5-5-01.PNG)  
https://arxiv.org/abs/1708.02182

上記表のモデルにおいても、本章と同様に多層のLSTM、Dropoutベースの正則化、重み共有が使われている。  
一番下のモデル「AWS-LSTM 3-layer LSTM(tied) + continuous cache pointer」に出てくる
**continuous cache pointer**は、  
8章で学ぶ**Attention**をベースとしたものである。


## まとめ
- LSTMレイヤにゲートの仕組みを実装することで、5章で作成したRNNを使った言語モデルよりも精度が良くなる。
- LSTMを使った言語モデルの改善テクニックとしては以下が有効である。
  - LSTMレイヤの多層化
  - Dropout
  - 重み共有
