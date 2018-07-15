import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from get_eeg import data

def dataset(total_size, test_size):
    x, y = [], []
    for i in range(total_size):
        if np.random.rand() <= 0.5:
            # 長さ 10 ~ 20のsin波
            _x = np.sin(np.arange(0, np.random.randint(10, 20)) + np.random.rand())
            # ノイズを付加
            _x += np.random.rand(len(_x)) * 0.05
            x.append(v(_x[:, np.newaxis]))
            y.append(np.array([1]))

        else:
            # 長さ 10 ~ 20の[0,1]の乱数列
            _x = np.random.rand(np.random.randint(10, 20))
            x.append(v(_x[:, np.newaxis]))
            y.append(np.array([0]))

    x_train = x[:-test_size]
    y_train = vi(y[:-test_size])
    x_test = x[-test_size:]
    y_test = vi(y[-test_size:])
    return x_train, x_test, y_train, y_test



def v(x):
    return Variable(np.asarray(x, dtype=np.float32))


def vi(x):
    return Variable(np.asarray(x, dtype=np.int32))


class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__()
        input_dim = 14
        hidden_dim = 100
        output_dim = 1

        with self.init_scope():
            self.lstm = L.NStepLSTM(n_layers=1, in_size=input_dim,
                      out_size=hidden_dim, dropout=0.3)
            self.l1 = L.Linear(hidden_dim, hidden_dim)
            self.l2 = L.Linear(hidden_dim, output_dim)

    def __call__(self, xs):
        """
        Parameters
        xs : list(Variable)

        """
        _, __, h = self.lstm(None, None, xs)
        h = v([_h[-1].data for _h in h])
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return F.sigmoid(y)

def forward(x, y, model):
    t = model(x)
    loss = F.sigmoid_cross_entropy(t, y)
    return loss

def train(max_epoch, train_size, valid_size):
    model = RNN()

    # train に1000サンプル、 testに1000サンプル使用
    #x_train, x_test, y_train, y_test = dataset(train_size + valid_size, train_size)

    # 学習用データの取得
    eeg = data(eeg = 'pilot_project/ishida/math_2018.07.10_16.24.29.csv',
                eeg_bp = 'pilot_project/ishida/math_2018.07.10_16.24.29.bp.csv',
                eeg_pm = 'pilot_project/ishida/math_2018.07.10_16.24.29.pm.csv')

    #eegデータを取得
    x_train,y_train,x_test,y_test = eeg.get(window = 256,slide = 10)
    y_train,y_test = vi(y_train),vi(y_test)
    x_train = [Variable(np.array(i,dtype=np.float32)) for i in x_train]
    x_test = [Variable(np.array(i,dtype=np.float32)) for i in x_test]

    #optimizer = optimizers.RMSprop(lr=0.03)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    early_stopping = 20
    min_valid_loss = 1e8
    min_epoch = 0

    train_loss, valid_loss = [], []

    for epoch in range(1, max_epoch):
        _y = model(x_test)
        y = _y.data
        y = np.array([1 - y, y], dtype='f').T[0]
        accuracy = F.accuracy(y, y_test.data.flatten()).data

        _train_loss = F.sigmoid_cross_entropy(model(x_train), y_train).data
        _valid_loss = F.sigmoid_cross_entropy(_y, y_test).data
        train_loss.append(_train_loss)
        valid_loss.append(_valid_loss)

        # valid_lossが20回連続で更新されなかった時点で学習を終了
        if min_valid_loss >= _valid_loss:
            min_valid_loss = _valid_loss
            min_epoch = epoch

        elif epoch - min_epoch >= early_stopping:
            break

        optimizer.update(forward, x_train, y_train, model)
        print('epoch: {} acc: {} loss: {} valid_loss: {}'.format(epoch, accuracy, _train_loss, _valid_loss))

    loss_plot(train_loss, valid_loss)
    serializers.save_npz('model.npz', model)


def loss_plot(train_loss, valid_loss):
    import matplotlib.pyplot as plt
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, valid_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss.png')


if __name__ == '__main__':
    train(max_epoch=1000, train_size=1000, valid_size=1000)
