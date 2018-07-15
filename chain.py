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
            l1 = L.NStepLSTM(n_layers=2, in_size=in_units,
                      out_size=hd_units, dropout=0.5),
            l2 = L.Linear(hd_units,hd_units),
            l3 = L.Linear(hd_units,out_units),
        )

    def predict(self, x):
        _,_,ys = self.l1(None,None,x)
        h = F.vstack(map(lambda y: y[-1], ys))
        h2 = F.dropout(F.relu(self.l2(h)))
        out = F.softmax(self.l3(h2))
        return  out

    def reset(self):
        pass
        #self.l1.reset_state()

    def output(self,x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        return  self.l3(h2)
