# -*- coding: utf-8 -*-
"""
eegデータをcsvから取得します。
"""
import pandas as pd
import csv
import numpy as np
from collections import defaultdict

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

def prolong(array,window,slide):
    """
    配列から窓をスライドさせてバッチに変換します。
    """
    dataset = []
    for i in range(0,len(array)-window+1,slide):
        dataset.append(array[i:i+window])
    return dataset


class data:
    def __init__(self,eeg,eeg_bp,eeg_pm):
        self.eeg = pd.read_csv(eeg)
        self.eeg_bp = pd.read_csv(eeg_bp)
        self.eeg_pm = pd.read_csv(eeg_pm)
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
        signal = [list(map(int, i[2:16])) for i in self.number_eeg]

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
                y = [[0 if key == '12' else 1 for i in x[0]] for j in x]
                if idx <= 2:
                    train_x.extend(x)
                    train_y.extend(y)
                else:
                    test_x.extend(x)
                    test_y.extend(y)
        return train_x,train_y,test_x,test_y


if __name__ == '__main__':
    data = data(eeg = 'pilot_project/ishida/math_2018.07.10_16.24.29.csv',
                eeg_bp = 'pilot_project/ishida/math_2018.07.10_16.24.29.bp.csv',
                eeg_pm = 'pilot_project/ishida/math_2018.07.10_16.24.29.pm.csv')
    #x,y = data.get_bp(batch_size = 50)
    #print(x.shape,y.shape)
    x,y,_,_ = data.get()
    print(y[10])
