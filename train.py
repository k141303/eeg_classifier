# -*- coding: utf-8 -*-

# 数値計算関連
import math
import random
import numpy as np

# chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, cuda
from chainer import serializers
from chain import MyChain

# default
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# データ取得
from get_eeg import data

# 損失関数の計算
def forward(x, y, model):
    t = model.predict(x)
    loss = F.softmax_cross_entropy(t, y)
    return loss

def main():
    # 実行時変数
    parser = argparse.ArgumentParser(description='NNEEG')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=70,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    #ディレクトリ作成
    if not os.path.isdir('model/'):
        os.makedirs('model/')
    if not os.path.isdir('log/'):
        os.makedirs('log/')

    # NNモデルを宣言
    model = MyChain(14,args.unit,3)

    #GPU設定
    if args.gpu != -1:
        gpu_device = args.gpu
        cuda.get_device(gpu_device).use()
        model.to_gpu(gpu_device)
        xp = cuda.cupy
    else:
        xp = np

    # 乱数のシードを固定
    random.seed(1)

    # 学習用データの取得
    eeg = data(eeg = 'pilot_project/ishida/math_2018.07.10_16.24.29.csv',
                eeg_bp = 'pilot_project/ishida/math_2018.07.10_16.24.29.bp.csv',
                eeg_pm = 'pilot_project/ishida/math_2018.07.10_16.24.29.pm.csv')


    # chainerのoptimizer
    #   最適化のアルゴリズムには Adam を使用
    optimizer = optimizers.Adam()
    #optimizer = optimizers.SGD()
    # modelのパラメータをoptimizerに渡す
    optimizer.setup(model)

    #eegデータを取得
    x,y,_,_ = eeg.get()

    #NN用に変換
    x = Variable(xp.array(x, dtype=np.float32))
    y = Variable(xp.array(y, dtype=np.int32))

    # パラメータの学習を繰り返す
    log_loss = []
    print('学習中...')
    for e in range(args.epoch):

        #データのシャッフル
        p = np.random.permutation(len(x))
        x = x[p]
        y = y[p]

        #バッジごとに処理
        sum_loss = []
        for x_batch,y_batch in zip(x,y):
            loss = forward(x_batch, y_batch, model)
            sum_loss.append(loss.data/len(x))
            optimizer.update(forward, x_batch, y_batch, model)
            #loss.unchain_backward()

        #LSTMをリセット
        model.reset()

        print(e+1,sum(sum_loss)/len(sum_loss))

        #誤差出力
        log_loss.append(sum(sum_loss)/len(sum_loss))
        plt.plot(log_loss)
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig("log/loss.png")

        #モデル出力
        if args.gpu != -1:
            model.to_cpu()
        serializers.save_npz("model/model.npz", model)
        if args.gpu != -1:
            model.to_gpu(gpu_device)

    #モデル出力
    if args.gpu != -1:
        model.to_cpu()
    serializers.save_npz("model/model.npz", model)

if __name__ == '__main__':
    main()