# chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, cuda
from chainer import serializers
from chain import MyChain
import chainer

import numpy as np


# 検証用にランダムデータを生成
x = (
    np.random.rand(1, 10).astype(np.float32),
    np.random.rand(1, 10).astype(np.float32),
    np.random.rand(1, 10).astype(np.float32),
)
t = chainer.Variable(np.random.randint(0, 5, 1)) # 0~4
print(x)
