import cnn_model as cm
import os.path as op

i = 0
for filename in cm.FileNames:
    # ディレクトリ名入力
    while True:
        dirname = input(">>「" + cm.ClassNames[i] + "」の画像のあるディレクトリ ： ")
        if op.isdir(dirname):
            break
        print(">> そのディレクトリは存在しません！")

    # 関数実行
    cm.PreProcess(dirname, filename, var_amount=3)
    i += 1
