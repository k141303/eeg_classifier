# -*- coding: utf-8 -*-
"""
ネットワーク設計を記載します。
"""
from chainer import Chain ,Variable
import chainer.links as L
import chainer.functions as F
import numpy as np

class MyChain(Chain):
    def __init__(self,in_units,hd_units,out_units,dropout = 0.5):
        super(MyChain, self).__init__(
            l1 = L.NStepLSTM(n_layers=2, in_size=in_units,
                      out_size=hd_units, dropout=dropout),
            l2 = L.Linear(hd_units,hd_units),
            l3 = L.Linear(hd_units,out_units),
        )

    def v(self,x):
        return Variable(np.asarray(x, dtype=np.float32))

    def predict(self, x):
        _,_,h = self.l1(None,None,x)
        #h = F.vstack(map(lambda y: y[-1], ys))
        h = self.v([_h[-1].data for _h in h])
        h2 = F.relu(self.l2(h))
        out = F.relu(self.l3(h2))
        return  F.sigmoid(out)
