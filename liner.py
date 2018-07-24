#!/usr/bin/env python
"""
線形モデルで脳波のパワースペクトルを分類します。
"""

import argparse
import numpy as np
import os

import chainer
from chainer import configuration, cuda, Variable
from chainer.dataset import convert
from chainer.iterators import MultiprocessIterator
import chainer.links as L
import chainer.functions as F
from chainer import serializers

from get_eeg import data

import matplotlib.pyplot as plt

# ネットワーク詳細
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # モデルの設計
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        #順伝搬
        h1 = F.dropout(F.relu(self.l1(x)),  ratio = 0)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio = 0)
        return self.l3(h2)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot using model '
                             'and state files in the specified directory')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    #ディレクトリ作成
    if not os.path.isdir(args.out+'/'):
        os.makedirs(args.out+'/')

    # ニューラルネットを設定
    model = L.Classifier(MLP(args.unit, 2))
    if args.gpu >= 0:
        # GPU向けに配列を変換
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    # オプティマイザーを設定
    optimizer = chainer.optimizers.AdaGrad()
    optimizer.setup(model)

    # 気にしないでください。
    if args.resume:
        # Resume from a snapshot
        serializers.load_npz('{}/mlp.model'.format(args.resume), model)
        serializers.load_npz('{}/mlp.state'.format(args.resume), optimizer)

    # EEGをロード
    ishida = data(eeg = 'pilot_project/ishida/math_2018.07.10_16.24.29.csv')
    ishida_fft = ishida.get_fft(window = 128,slide = 128,band = None)

    djuned = data(eeg = 'pilot_project/djuned/math_2018.07.10_17.01.52.csv')
    djuned_fft = djuned.get_fft(window = 128,slide = 128,band = None)

    train_x,train_t,test_x,test_t = [_i + _j for _i,_j in zip(ishida_fft,djuned_fft)]
    #train_x,train_t,test_x,test_t = ishida_fft

    # NN用にデータセットを変換
    train_x = [xp.array(_x,dtype=np.float32) for _x in train_x]
    test_x = [xp.array(_x,dtype=np.float32) for _x in test_x]
    train = list(zip(train_x,train_t))
    test = list(zip(test_x,test_t))


    print('# length: {}'.format(train_x[0].shape[0]))

    train_count = len(train_x)
    test_count = len(test_x)

    print('# train-size: {}'.format(train_count))
    print('# test-size: {}'.format(test_count))

    train_log_loss,test_log_loss = [],[]
    train_log_act,test_log_act = [],[]

    with MultiprocessIterator(train, args.batchsize) as train_iter, \
        MultiprocessIterator(test, args.batchsize,
                             repeat=False, shuffle=False) as test_iter:

        sum_accuracy = 0
        sum_loss = 0
        sum_batch = 0

        #　エポック数に達するまで学習
        while train_iter.epoch < args.epoch:
            batch = train_iter.next()
            x, t = [xp.array(_batch) for _batch in zip(*batch)]
            optimizer.update(model, x, t)
            #経過の格納
            sum_batch += len(t)
            sum_loss += float(model.loss.data) * len(t)
            sum_accuracy += float(model.accuracy.data) * len(t)

            if train_iter.is_new_epoch:
                # 経過の出力と保存
                print('epoch: {}'.format(train_iter.epoch))
                print('train mean loss: {}, accuracy: {}'.format(
                    sum_loss / sum_batch, sum_accuracy / sum_batch))
                train_log_loss.append(sum_loss / sum_batch)
                train_log_act.append(sum_accuracy / sum_batch)

                # テストモード
                sum_accuracy = 0
                sum_loss = 0
                sum_batch = 0
                with configuration.using_config('train', False):
                    # テストモードでの動作を指定
                    with chainer.using_config('enable_backprop', False):
                        for batch in test_iter:
                            x, t = [xp.array(_batch) for _batch in zip(*batch)]
                            loss = model(x, t)
                            #経過の格納
                            sum_batch += len(t)
                            sum_loss += float(loss.data) * len(t)
                            sum_accuracy += float(model.accuracy.data) * len(t)

                test_iter.reset()
                print('test mean  loss: {}, accuracy: {}'.format(
                    sum_loss / sum_batch, sum_accuracy / sum_batch))
                test_log_loss.append(sum_loss / sum_batch)
                test_log_act.append(sum_accuracy / sum_batch)
                sum_accuracy = 0
                sum_loss = 0
                sum_batch = 0

        # モデルとオプティマイザー設定の保存
        print('save the model')
        serializers.save_npz('{}/mlp.model'.format(args.out), model)
        print('save the optimizer')
        serializers.save_npz('{}/mlp.state'.format(args.out), optimizer)


        #logの出力
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        #誤差
        axL.plot(train_log_loss, label = "train")
        axL.plot(test_log_loss, label = "test")
        axL.set_title('loss')
        axL.set_xlabel("epoch")
        axL.set_ylabel("loss")
        #正答率
        axR.plot(train_log_act, label = "train")
        axR.plot(test_log_act, label = "test")
        axR.set_title('accuracy')
        axR.set_xlabel("epoch")
        axR.set_ylabel("accuracy")
        #出力
        fig.savefig("log/log.png")

if __name__ == '__main__':
    main()
