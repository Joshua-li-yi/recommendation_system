import pickle
import numpy as np
from numpy.random import choice
from math import floor
import pandas as pd
#%%
# 全局变量
# 设置随机数种子
SEED = 1
# 设置取的数据的比例
raw_fraction = 1.
# 文件位置
FILE_PATH = 'data/'

# 将dict类型的train数据转为df类型，并于item_plus 合并
def train_data_to_df(train=[], item_plus=[], output_csv=False, input_csv=False):
    if input_csv is True:
        item_plus = pd.read_csv(FILE_PATH + 'item_plus.csv')
        train = pickle.load(open(FILE_PATH + 'train.pickle', 'rb'))
    # 将dict类型的train数据转为df类型
    # print(len(train))
    # print(train)
    # print(train[19834])

    item_list = []
    test = []
    for id, item in train.items():
        test_item_list = choice(list(item.keys()), size=floor(0.2*len(item.keys())), replace=False)
        for test_item in test_item_list:
            test.append([id,test_item, item[test_item]])
        for item_id, score in item.items():
            item_list.append([id, item_id, score])
    # print(test)

    # return item_list, test


# 加载train_data 数据类型dict嵌套
# {use_ed:{item_id:score}}
def load_train_data(filepath, output_pickle=False, frac=1., input_pickle=False):
    print('begin load train data')
    # 如果选择导入pickle格式的train数据集
    if input_pickle is True:
        train = pickle.load(open(FILE_PATH + 'train.pickle', 'rb'))
    else:  # 选择导入 txt格式的训练集
        with open(filepath, 'r') as f:
            train = {}
            while True:
                line = f.readline()
                if not line or line == '\n':
                    break

                id, item_num = line.split('|')
                id = int(id)
                item_num = int(item_num)
                item = {}
                # 遍历之后的内容
                for i in range(item_num):
                    line = f.readline()
                    item_id, score = line.split("  ")[:2]
                    # 数据类型转化
                    score = int(score)
                    item_id = int(item_id)
                    # 放入字典中
                    item[item_id] = score
                # 字典嵌套
                print('---------------id------------')
                print(id)
                print('-------------item-------------')
                print(item)
                train[id] = item

    # print(train)

    # 使用dump()将数据序列化到文件中
    # if output_pickle is True:
    #     with open(FILE_PATH + 'train.pickle', 'wb') as handle:
    #         pickle.dump(train, handle)
    # print('load train data finish')

    return train

# train = load_train_data(FILE_PATH+'train.txt',input_pickle=False)
# total, test = train_data_to_df(input_csv=True)
# print('-------------------------------')
# print(total)
# 将dict类型的train数据转为df类型，并于item_plus 合并
# test_size = 0.2 选取的测试集的比例
def train_test_divide(train, output_csv=False, input_data=False, test_size=0.2):
    print('begin divide train and test')
    if input_data is True:
        # 导入item_plus
        item_plus = pd.read_csv(FILE_PATH + 'item_plus.csv')
        # item_plus = pd.read_csv(FILE_PATH + 'item_plus.csv', header=0, names=['Unnamed','user','ID_Power2','attribute2','attribute1','user_Power2','ID','user_Power2_multiply_ID','user_Power2_multiply_user','ID_multiply_ID_Power2'])
        # 导入所有的数据集
        train = pickle.load(open(FILE_PATH + 'train.pickle', 'rb'))
    # print(item_plus.head())

    # 去掉第一列
    # item_plus.drop(['Unnamed'], axis=1, inplace=True)
    # print(item_plus.head())

    total_list = []
    # 存放test的数据
    testset = []
    print('begin divide test set')
    for id, item in train.items():
        # 从一个用户的所用评分中随机选择test_size比例的数据，作为测试集，不重复
        test_item_list = choice(list(item.keys()), size=floor(test_size * len(item.keys())), replace=False)
        # test
        for test_item in test_item_list:
            testset.append([id, test_item, item[test_item]])
        # 所有的评分
        for item_id, score in item.items():
            total_list.append([id, item_id, score])
    # 将testset有dict转为df类型
    test_df = pd.DataFrame(data=testset, columns=['user', 'ID', 'score'])
    # 删去testset
    del testset

    print('begin divide traint set')
    # 选区total中有的但是测试集中没有的数据作为训练集
    # trainset = [i for i in total_list if i not in testset]
    # 将dict类型数据转为df类型
    total_df = pd.DataFrame(data=total_list, columns=['user', 'ID', 'score'])
    # 删去total list
    del total_list
    print(total_df.tail())

    # total_df.dropna(axis=0, how='any', inplace=True) #drop all rows that have any NaN values
    print(total_df.tail())
    print(total_df)
    # 先扩展再去重，得到trian
    # test_df.reset_index()
    train_df = total_df.append(test_df)
    print(test_df)
    train_df['user'] = train_df['user'].astype(int)
    # 所有的train数据减去重复的就是所得剩下的train（此train包含着验证集，也就是说验证集还没有划分）
    train_df.drop_duplicates(subset=['user', 'ID', 'score'], keep=False, inplace=True)
    print('--------begin merge test and item plus ------')
    test_df_plus = pd.merge(test_df, item_plus, on=['user', 'ID'], how='left')

    print('-----------test_df_plus.describe()---------')
    print(test_df_plus.describe())

    if output_csv is True:
        print('---------save as csv----------')
        test_df.set_index('user', inplace=True)
        test_df.to_csv(FILE_PATH + 'testset.csv')
        test_df_plus.set_index('user', inplace=True)
        test_df_plus.to_csv(FILE_PATH + 'testset_plus.csv')

    del test_df_plus

    if output_csv is True:
        print('---------save as csv----------')
        total_df.set_index('user', inplace=True)
        total_df.to_csv(FILE_PATH + 'train.csv')
    del total_df

    print('--------begin merge train and item plus ------')
    train_df_plus = pd.merge(train_df, item_plus, on=['user', 'ID'], how='left')
    if output_csv is True:
        print('---------save as csv----------')

        train_df.set_index('user', inplace=True)
        train_df.to_csv(FILE_PATH + 'trainset.csv')
        train_df_plus.set_index('user', inplace=True)
        train_df_plus.to_csv(FILE_PATH + 'trainset_plus.csv')
    del train_df

    print('------------train_df_plus.describe()------------')
    print(train_df_plus.describe())
    del train_df_plus
    # 保存item plus
    # if output_csv is True:
        # print('---------save as csv----------')

        # item_plus.set_index('user', inplace=True)
        # item_plus.to_csv(FILE_PATH+'item_plus.csv')
    print('divide train and test end ')
    return 0
# train = []
# train_test_divide(train, input_data=True, output_csv=True)


data_for_lightfm(input_csv=True, output_csv=True)
