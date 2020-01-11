import cnn_model as cm
import os.path as op

while True:
    while True:
        imgname = input("\n>> 入力したい画像ファイル(「END」で終了) ： ")
        if op.isfile(imgname) or imgname == "END":
            break
        print(">> そのファイルは存在しません！")
    if imgname == "END":
        break

    # 関数実行
    cm.TestProcess(imgname)
