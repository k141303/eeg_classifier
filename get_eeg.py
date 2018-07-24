# -*- coding: utf-8 -*-
"""
eegデータをcsvから取得します。
"""
import pandas as pd
import csv
import numpy as np
from collections import defaultdict
from chainer import Chain, Variable
import matplotlib.pyplot as plt

#テスト用
import random
from copy import deepcopy as copy
import collections

def prolong(array,window = 128,slide = 1):
    """
    配列から窓をスライドさせてバッチに変換します。
    波形の最後に終了を示すフラグも添付します。
    window : 窓の幅
    slide : スライド数
    """
    dataset = []
    #array = [[j/400 - 10 for j in i] for i in array]
    array = [[j for j in i] for i in array]
    end = [0.0 for i in range(len(array[0]))] #終了フラグ
    for i in range(0,len(array)-window+1,slide):
        pick = copy(array[i:i+window])
        #pick.append(copy(end))  #終了フラグの結合
        dataset.append(pick)
    return dataset

def to_fft(array,rate = 128,band = [8,50]):
    """
    配列をフーリエ変換しパワースペクトルに変換します。
    array : 入力配列
    rate : サンプリングレート(Hz)
    band : パワースペクトルの切り出し範囲 Noneとすると切り出さずに返します。
    """
    freq = np.linspace(0, rate, len(array)) # 周波数軸
    t = np.arange(0, len(array)*(1/rate), (1/rate)) # 時間軸

    # 高速フーリエ変換
    F = [np.fft.fft(_arrayT) for _arrayT in np.array(array).T]

    # 振幅スペクトルを計算
    Amp = np.array([np.abs(_F) for _F in F]).T

    if band == None:    #切り出し範囲が指定されていない場合
        return Amp.flatten()

    #帯域切り出し
    freq = freq[int(len(array)/rate)*8:int(len(array)/rate)*50]
    Amp = Amp[int(len(array)/rate)*8:int(len(array)/rate)*50]

    # グラフ表示
    """
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8
    plt.subplot(121)
    for idx,i in enumerate(np.array(array).T):
        plt.plot(t, i,label = idx)
    plt.legend(['Data 1','Data 2'])
    plt.xlabel("Time", fontsize=8)
    plt.ylabel("Signal", fontsize=8)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=8)
    leg.get_frame().set_alpha(1)
    plt.subplot(122)
    for idx,i in enumerate(Amp.T):
        plt.plot(freq, i,label = idx)
    plt.legend(['Data 1','Data 2'])
    plt.xlabel('Frequency', fontsize=8)
    plt.ylabel('Amplitude', fontsize=8)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=8)
    leg.get_frame().set_alpha(1)
    plt.show()
    """
    return Amp.flatten()


class data:
    def __init__(self,eeg):
        #pandas形式(未使用)
        self.eeg = pd.read_csv(eeg)

        #独自形式
        self.number_eeg,self.header_eeg = self.read_csv(eeg)

    def read_csv(self,filepass):
        """
        csvをcsvモジュールを用いて読み込みます。
        """
        numbers = []
        with open(filepass, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)   # ヘッダーを読み飛ばしたい時
            for row in reader:
                number = [float(i) for i in row]
                numbers.append(number) # 1行づつ取得できる
        return numbers,header

    def get(self,window = 128,slide = 25,remove = (1/6)):
        """
        NN向けにデータセットを変換して渡します。
        前処理としてeegデータをキーストロークで分割します。
        batch_size : バッチサイズ
        remove : 前後からこの範囲をノイズとしてカットします
        """
        #脳波部の抽出
        signal = [list(map(float, i[2:16])) for i in self.number_eeg]
        #signal = [list(map(float, [i[3]])) for i in self.number_eeg]

        #キーストロークの抽出
        key = [int(i[19]) for i in self.number_eeg]

        #キーストロークのインデックスを作成
        tag = [(i,idx) for idx,i in enumerate(key) if i != 0]

        #脳波部をキーストロークから分類
        spl = defaultdict(lambda: [])
        for idx,(i,j) in enumerate(tag):
            if i == -1 or idx + 1 == len(tag):  #終了
                break

            range = int((tag[idx+1][1] - j) * remove)   #前後の除去幅(ノイズ対策)
            pick = signal[j+range:tag[idx+1][1]-range] #切り取り

            spl[str(i)+str(tag[idx+1][0])].append(pick) #辞書にまとめる

        #脳波を学習用にバッチに分割
        train_x,train_y = [],[]
        test_x,test_y = [],[]
        for key,item in spl.items():
            if key == '122' or key == '24-1':
                continue
            for idx,line in enumerate(item):
                x = prolong(line,window,slide)
                y = [0 if key[:2] == '22' else 1 for i in x]
                if idx % 2 == 0:
                    train_x.extend(x)
                    train_y.extend(y)
                else:
                    test_x.extend(x)
                    test_y.extend(y)
        return train_x,train_y,test_x,test_y

    def get_fft(self,window = (128 * 1),slide = (128*1),band = None):
        """
        データをパワースペクトルで取得します。
        """
        #eegデータの取得
        x,y,test_x,test_y = self.get(window = window,slide = slide)

        #フーリエ変換しパワースペクトルで取得
        x = [to_fft(_x ,band = band) for _x in x]
        test_x = [to_fft(_test_x ,band = band) for _test_x in test_x]

        return x,y,test_x,test_y



if __name__ == '__main__':
    data = data(eeg = 'pilot_project/kusano/1st/tetris_2018.07.23_15.40.45.csv')
    x,y,test_x,test_y = data.get_fft()
    print(np.array(x).shape,np.array(y).shape,np.array(test_x).shape,np.array(test_y).shape)
