# coding: utf-8
import sys

sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from chapter3.simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

# パラメータ
window_size = 1
hidden_size = 3
batch_size = 3
max_epoch = 1000

# 前準備
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# トレイニング
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)

trainer.fit(contexts,target,max_epoch,batch_size)
trainer.plot()