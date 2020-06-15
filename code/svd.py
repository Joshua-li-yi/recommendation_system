# -*- coding:utf-8 -*-
# @Time： 2020-06-14 15:18
# @Author: Joshua_yi
# @FileName: submit.py
# @Software: PyCharm
# @Project: recommendation_system
import pandas as pd
import numpy as np
from surprise.dump import load
import time
import os

# 文件位置
FILE_PATH = r''


# 加载test文件
def load_test_data(filepath, output_csv=False):
    print('begin load test data')
    # 打开文件
    with open(filepath, 'r') as f:
        test = []
        while True:
            line = f.readline()
            if not line or line == '\n':
                break

            id, item_num = line.split('|')
            # 类型转化
            id = int(id)
            item_num = int(item_num)

            # 遍历之后的内容
            for i in range(item_num):
                line = f.readline()
                item_id = line
                # 数据类型转化
                item_id = int(item_id)
                # 放入test中
                test.append([id, item_id, 0])
    # 转为df类型
    test = pd.DataFrame(data=test, columns=['user', 'ID', 'score'])
    # test.set_index('user', inplace=True)

    if output_csv is True:
        test.to_csv(FILE_PATH + 'test.csv')
    print('load test data finish')
    return test


def load_model(filepath):
    temp_pred, algo = load(filepath)
    del temp_pred
    print('load model finish')
    return algo


# 相当于swithch case 语句
def rescale1_5(score):
    switcher = {
        1: 10,
        2: 30,
        3: 50,
        4: 70,
        5: 90,
    }
    # 默认值为50
    return switcher.get(score, 50)


if __name__ == '__main__':
    begin = time.perf_counter()
    test = load_test_data(FILE_PATH + 'test.txt', output_csv=False)
    algo = load_model(FILE_PATH+'svd.model')
    pred = []
    for row in test.itertuples():
        # 注意这里一定要 把 user ， item_id 转为str格式的
        pred.append(algo.predict(str(row[1]), str(row[2]), r_ui=row[3]).est)
    del algo
    print('predict data finish')
    # 四舍五入
    pred_round = np.round(pred)
    # 从1-5转到原来的数据
    pred_score = []
    for p in pred_round:
        # 先转化为int
        pred_score.append(rescale1_5(int(p)))
    test['pred'] = pred_score
    test.drop('score', axis=1, inplace=True)
    print(test.head(10))

    # 写入text
    with open("new_submit.txt", "w") as f:
        temp_user = 1
        for row in test.itertuples():
            if temp_user != row[1]:
                f.write(str(row[1]) + '|6\n')
                temp_user = row[1]
            f.write(str(row[2]) + "  " + str(row[3]) + "\n")
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - begin))
    os.system('pause')