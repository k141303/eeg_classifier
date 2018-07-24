EEGを集中時とリラックス時で分別するプログラム

設定できるパラメータのリストです

◎データ取得のパラメータ(3パラメータ)
kunii_1st = data(eeg = 'pilot_project/kunii/1st/tetris_2018.07.23_15.07.37.csv')
kunii_1st_fft = kunii_1st.get_fft(window = (128 * 1),slide = (128 * 1),band = None)
window = (128 * w) => wでウィンドウ幅(秒)を設定
slide = (128 * s) => sでスライド幅(秒)を指定
slide = [a,b] => 切り出し範囲a(Hz)〜b(Hz)を指定(0〜128)

◎ネットワーク設計のパラメータ(2パラメータ)
h1 = F.dropout(F.relu(self.l1(x)),  ratio = 0)
relu => 活性化関数(他を指定してもいいです。)
ratio => ドロップアウト率(0〜1)

◎実行時パラメータ(3パラメータ)
python liner.py
-b n => バッジ数をnに設定
-e n => エポック数をnに設定
-u n => ユニット数をnに設定
-r -g -o => 使わなくて大丈夫
