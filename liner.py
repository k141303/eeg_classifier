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

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            self.l4 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.dropout(F.relu(self.l1(x)),  ratio = 0.5)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio = 0.5)
        h3 = F.dropout(F.relu(self.l3(h2)), ratio = 0.5)
        return self.l4(h3)

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
    print('')

    #ディレクトリ作成
    if not os.path.isdir(args.out+'/'):
        os.makedirs(args.out+'/')

    # Set up a neural network to train
    model = L.Classifier(MLP(args.unit, 2))
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
        xp = cuda.cupy
    else:
        xp = np

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if args.resume:
        # Resume from a snapshot
        serializers.load_npz('{}/mlp.model'.format(args.resume), model)
        serializers.load_npz('{}/mlp.state'.format(args.resume), optimizer)

    # Load the EEG dataset
    eeg = data(eeg = 'pilot_project/ishida/math_2018.07.10_16.24.29.csv',
                eeg_bp = 'pilot_project/ishida/math_2018.07.10_16.24.29.bp.csv',
                eeg_pm = 'pilot_project/ishida/math_2018.07.10_16.24.29.pm.csv')
    train_x,train_t,test_x,test_t = eeg.get_fft(window = (128 * 5),slide = 128,band = [5,80])
    train_x = [xp.array(_x,dtype=np.float32) for _x in train_x]
    test_x = [xp.array(_x,dtype=np.float32) for _x in test_x]
    train = list(zip(train_x,train_t))
    test = list(zip(test_x,test_t))

    train_count = len(train_x)
    test_count = len(test_x)

    with MultiprocessIterator(train, args.batchsize) as train_iter, \
        MultiprocessIterator(test, args.batchsize,
                             repeat=False, shuffle=False) as test_iter:

        sum_accuracy = 0
        sum_loss = 0

        while train_iter.epoch < args.epoch:
            batch = train_iter.next()
            x, t = [xp.array(_batch) for _batch in zip(*batch)]
            optimizer.update(model, x, t)
            sum_loss += float(model.loss.data) * len(t)
            sum_accuracy += float(model.accuracy.data) * len(t)

            if train_iter.is_new_epoch:
                print('epoch: {}'.format(train_iter.epoch))
                print('train mean loss: {}, accuracy: {}'.format(
                    sum_loss / train_count, sum_accuracy / train_count))
                # evaluation
                sum_accuracy = 0
                sum_loss = 0
                # Enable evaluation mode.
                with configuration.using_config('train', False):
                    # This is optional but can reduce computational overhead.
                    with chainer.using_config('enable_backprop', False):
                        for batch in test_iter:
                            x, t = [xp.array(_batch) for _batch in zip(*batch)]
                            loss = model(x, t)
                            sum_loss += float(loss.data) * len(t)
                            sum_accuracy += float(model.accuracy.data) * len(t)

                test_iter.reset()
                print('test mean  loss: {}, accuracy: {}'.format(
                    sum_loss / test_count, sum_accuracy / test_count))
                sum_accuracy = 0
                sum_loss = 0

        # Save the model and the optimizer
        print('save the model')
        serializers.save_npz('{}/mlp.model'.format(args.out), model)
        print('save the optimizer')
        serializers.save_npz('{}/mlp.state'.format(args.out), optimizer)


if __name__ == '__main__':
    main()
