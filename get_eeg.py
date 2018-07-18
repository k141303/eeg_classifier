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

point = ['AF3_THETA','AF3_ALPHA','AF3_LOW_BETA','AF3_HIGH_BETA',
        'AF3_GAMMA','F7_THETA','F7_ALPHA','F7_LOW_BETA','F7_HIGH_BETA',
        'F7_GAMMA','F3_THETA','F3_ALPHA','F3_LOW_BETA','F3_HIGH_BETA',
        'F3_GAMMA','FC5_THETA','FC5_ALPHA','FC5_LOW_BETA','FC5_HIGH_BETA',
        'FC5_GAMMA','T7_THETA','T7_ALPHA','T7_LOW_BETA','T7_HIGH_BETA',
        'T7_GAMMA','P7_THETA','P7_ALPHA','P7_LOW_BETA','P7_HIGH_BETA',
        'P7_GAMMA','O1_THETA','O1_ALPHA','O1_LOW_BETA','O1_HIGH_BETA',
        'O1_GAMMA','O2_THETA','O2_ALPHA','O2_LOW_BETA','O2_HIGH_BETA',
        'O2_GAMMA','P8_THETA','P8_ALPHA','P8_LOW_BETA','P8_HIGH_BETA',
        'P8_GAMMA','T8_THETA','T8_ALPHA','T8_LOW_BETA','T8_HIGH_BETA',
        'T8_GAMMA','FC6_THETA','FC6_ALPHA','FC6_LOW_BETA','FC6_HIGH_BETA',
        'FC6_GAMMA','F4_THETA','F4_ALPHA','F4_LOW_BETA','F4_HIGH_BETA',
        'F4_GAMMA','F8_THETA','F8_ALPHA','F8_LOW_BETA','F8_HIGH_BETA',
        'F8_GAMMA','AF4_THETA','AF4_ALPHA','AF4_LOW_BETA','AF4_HIGH_BETA','AF4_GAMMA']

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
    def __init__(self,eeg,eeg_bp,eeg_pm):
        #pandas形式(未使用)
        self.eeg = pd.read_csv(eeg)
        self.eeg_bp = pd.read_csv(eeg_bp)
        self.eeg_pm = pd.read_csv(eeg_pm)

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

    def get_bp(self,batch_size = 50):
        """
        NN向けにデータセットを変換して渡します。
        (未完成)
        """
        size = int(self.eeg_bp.shape[0]/batch_size)
        out = np.array([[0 for i in range(batch_size-1)] for i in range(size)])
        ans = np.array([[random.randrange(2)+1] for i in range(size)])
        out = np.concatenate([out, ans], axis=1)
        train = np.array([self.eeg_bp.ix[i * batch_size:(i+1)*batch_size-1,point].as_matrix() for i in range(size)])
        return train,out

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
            if i == -3 or idx + 1 == len(tag):  #終了
                break

            range = int((tag[idx+1][1] - j) * remove)   #前後の除去幅(ノイズ対策)
            pick = signal[j+range:tag[idx+1][1]-range] #切り取り

            spl[str(i)+str(tag[idx+1][0])].append(pick) #辞書にまとめる

        #脳波を学習用にバッチに分割
        train_x,train_y = [],[]
        test_x,test_y = [],[]
        for key,item in spl.items():
            if not key == '12' and not key == '21':
                continue
            for idx,line in enumerate(item):
                x = prolong(line,window,slide)
                y = [[0] if key == '12' else [1] for i in x]
                if idx <= 2:
                    train_x.extend(x)
                    train_y.extend(y)
                else:
                    test_x.extend(x)
                    test_y.extend(y)
        return train_x,train_y,test_x,test_y

    def get_fft(self,window = (128 * 5),slide = 128,band = [5,80]):
        """
        データをパワースペクトルで取得します。
        """
        #eegデータの取得
        x,y,test_x,test_y = self.get(window = window,slide = slide)
        print(np.array(x).shape,np.array(test_x).shape)

        #フーリエ変換しパワースペクトルで取得
        x = [to_fft(_x ,band = band) for _x in x]
        test_x = [to_fft(_test_x ,band = band) for _test_x in test_x]

        return x,y,test_x,test_y



if __name__ == '__main__':
    data = data(eeg = 'pilot_project/ishida/math_2018.07.10_16.24.29.csv',
                eeg_bp = 'pilot_project/ishida/math_2018.07.10_16.24.29.bp.csv',
                eeg_pm = 'pilot_project/ishida/math_2018.07.10_16.24.29.pm.csv')
    x,y,test_x,test_y = data.get_fft()
    print(np.array(x).shape,np.array(y).shape,np.array(test_x).shape,np.array(test_y).shape)
