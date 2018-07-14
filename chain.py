# -*- coding: utf-8 -*-
"""
ネットワーク設計を記載します。
"""
from chainer import Chain
import chainer.links as L
import chainer.functions as F

class MyChain(Chain):
    def __init__(self,in_units,hd_units,out_units):

        super(MyChain, self).__init__(
            l1 = L.Linear(in_units,hd_units),
            l2 = L.LSTM(hd_units,hd_units),
            l3 = L.LSTM(hd_units,hd_units),
            l4 = L.Linear(hd_units,out_units),
        )

    def predict(self, x):
        self.reset()
        for idx,i in enumerate(x):
            h1 = F.dropout(F.relu(self.l1(x)))
            h2 = F.dropout(self.l2(h1))
            h3 = F.dropout(self.l3(h2))
            out = F.softmax(self.l4(h3))
        return  out

    def reset(self):
        self.l2.reset_state()
        self.l3.reset_state()

    def output(self,x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return  self.l4(h3)
