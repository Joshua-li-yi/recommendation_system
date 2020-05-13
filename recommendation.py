import numpy as np
import pandas as pd
import time
from math import sqrt
# 垃圾回收，内存管理
import gc
# 打包文件
import pickle
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, BaselineOnly, accuracy, SVD
from surprise.model_selection import cross_validate, KFold, PredefinedKFold, train_test_split
# 随机选择的包
from numpy.random import choice
# 向下取整
from math import floor
import os
import warnings

warnings.filterwarnings('ignore')

# 设置pycharm显示宽度和高度
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 全局变量
# 设置随机数种子
SEED = 1
# 设置取的数据的比例
raw_fraction = 1.
# 文件位置
FILE_PATH = 'data/'


# 加载item属性数据
# item dataframe columns=['ID', 'attribute1', 'attribute2']
def load_item(filepath, output_csv=False, frac=1., input_csv=False):
    print("begin load data")

    if input_csv is True:
        item = pd.read_csv(FILE_PATH + 'item.csv')
    else:
        txt = np.loadtxt(filepath, dtype=str, delimiter='|')
        item = pd.DataFrame(data=txt, columns=['ID', 'attribute1', 'attribute2'])
        # item.set_index('ID', inplace=True)
        # print(item.describe())
        if frac != 1.:
            print('random select', frac * 100, '% data')
            item = item.sample(frac=frac, random_state=SEED)

        # 将None替换为0
        # item 属性中没有值为0的数据，所以这里可以用0来填充
        item.replace('None', 0, inplace=True)

        if output_csv is True:
            item.to_csv(FILE_PATH + 'item.csv')
        # print(item.describe())
        # 类型转换
        item['ID'] = item['ID'].astype(int)
        item['attribute1'] = item['attribute1'].astype(int)
        item['attribute2'] = item['attribute2'].astype(int)

    print('load item data finish')
    return item


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
                train[id] = item
    # print(train)

    # 使用dump()将数据序列化到文件中
    if output_pickle is True:
        with open(FILE_PATH + 'train.pickle', 'wb') as handle:
            pickle.dump(train, handle)
    print('load train data finish')

    return train


# 数据清洗
def item_data_clearning(item, output_csv=False, already_cleaning=False):
    print('begin item data cleaning')
    print('------------------item data head----------------------')
    print(item.head())
    print('------------------item data describe----------------------')
    print(item.describe())
    print('------------------item data info ----------------------')
    print(item.info())
    if already_cleaning is True:
        return item
    else:
        # none值处理
        # attribute1的平均值
        attribute1_avg = item['attribute1'].mean()
        # 将0值即之间的none值替换为均值
        item['attribute1'].replace(0, attribute1_avg, inplace=True)
        # attribute2的平均值
        attribute2_avg = item['attribute2'].mean()
        # 将0值即之间的none值替换为均值
        item['attribute2'].replace(0, attribute2_avg, inplace=True)
        # print(item.head())

        # 物品属性缺失处理
        # ID max = 624960+1 从零开始计数的
        # ID rows = 507172
        # 缺失 624960+1-507172个，将这些值使用平均值进行填充，
        ID_max = item['ID'].max()
        print('--------------------ID_max------------------------')
        print(ID_max)
        # 实际上应该有的所有ID
        ID_full_list = set(range(ID_max))
        print('--------------------ID_full_list------------------------')
        # print(ID_full_list)
        # 数据集中给出的ID
        ID_list = set(item['ID'].tolist())
        print('--------------------ID_list------------------------')
        # print(ID_list)
        # 两者做差集求出缺失的ID
        ID_null = ID_full_list - ID_list
        print('--------------------ID_null------------------------')
        print(len(ID_null))
        # 将缺失的ID用均值进行填充
        df_list = []
        # 显示进度条
        with tqdm(total=len(ID_null), desc='ID null fill process') as bar:
            for id_null in ID_null:
                temp_dict = {'ID': id_null, 'attribute1': attribute1_avg, 'attribute2': attribute2_avg}
                df_list.append(temp_dict)
                bar.update(1)
        # 将list形式转化为df形式
        temp_df = pd.DataFrame(data=df_list, columns=['ID', 'attribute1', 'attribute2'])
        # 将新生成的df添加到item中
        item = item.append(temp_df, ignore_index=True)
        # 安装ID从小到大排序
        item.sort_values(by=['ID'], ascending=True, inplace=True)
        # 将ID设置为index
        item.set_index('ID', inplace=True)

        # 重复值处理
        # 无重复值
        print("--------------------- item duplication---------------------")
        print(item[item.duplicated()])

        # 输出为csv
        if output_csv is True:
            item.to_csv(FILE_PATH + 'item.csv')

    print('item data cleaning finish')
    return item


# 特征工程
# 特征构建，特征提取，特征选择
# 在jupyter上
# def item_data_construction(item_attributes, output_csv=False):
#     print('begin item data construction')
#
#     item_attributes.set_index('ID', inplace=True)
#     # 主要采取归一化
#     # 归一化(Normalization)
#     item_attributes['atbt1_normalized'] = (item_attributes['attribute1'] - item_attributes['attribute1'].min()) / (
#             item_attributes['attribute1'].max() - item_attributes['attribute1'].min())
#     # 标准化（Standardization）z - score方法规范化(x - mean(x)) / std(x)
#     # item_attributes['atbt1_standard'] = (item_attributes['attribute1'] - item_attributes['attribute1'].mean) / item_attributes['attribute1'].std()
#
#     # 归一化(Normalization)
#     item_attributes['atbt2_normalized'] = (item_attributes['attribute2'] - item_attributes['attribute2'].min()) / (
#             item_attributes['attribute2'].max() - item_attributes['attribute2'].min())
#     # 标准化（Standardization）z - score方法规范化(x - mean(x)) / std(x)
#     # item_attributes['atbt2_standard'] = (item_attributes['attribute1'] - item_attributes['attribute1'].mean) / item_attributes['attribute1'].std()
#
#     # 正则化
#     # a1_normalized = preprocessing.normalize(np.array(item_attributes['attribute1']).reshape(-1,1))
#     # a1_normalized = pd.DataFrame(a1_normalized)
#     # item_attributes['attribute1_normalized'] = a1_normalized
#
#     # 其他
#     item_attributes['atbt1+atbt2'] = item_attributes['atbt1_normalized'] + item_attributes['atbt2_normalized']
#     item_attributes['atbt1/atbt2'] = item_attributes['atbt1_normalized'] / item_attributes['atbt2_normalized']
#     item_attributes['atbt1_log'] = np.log(item_attributes['attribute1'])
#     item_attributes['atbt2_log'] = np.log(item_attributes['attribute2'])
#
#     print(item_attributes.head())
#     print(item_attributes.describe())
#
#     if output_csv is True:
#         item_attributes.to_csv(FILE_PATH + 'item_plus.csv')
#
#     print('item data construction finish')
#     # item_attributes.reset_index()
#     return item_attributes


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
    print('--------begin merge test and item plus ------')
    test_df_plus = pd.merge(test_df, item_plus, on=['user', 'ID'], how='left')
    if output_csv is True:
        print('---------save as csv----------')
        test_df.set_index('user', inplace=True)
        test_df.to_csv(FILE_PATH + 'testset.csv')
        test_df_plus.set_index('user', inplace=True)
        test_df_plus.to_csv(FILE_PATH + 'testset_plus.csv')

    print('-----------test_df_plus.describe()---------')
    print(test_df_plus.describe())
    del test_df_plus

    print('begin divide traint set')
    # 选区total中有的但是测试集中没有的数据作为训练集
    # trainset = [i for i in total_list if i not in testset]
    # 将dict类型数据转为df类型
    total_df = pd.DataFrame(data=total_list, columns=['user', 'ID', 'score'])
    # 删去total list
    del total_list
    # 先扩展再去重，得到trian
    train_df = total_df.append(test_df)
    # 所有的train数据减去重复的就是所得剩下的train（此train包含着验证集，也就是说验证集还没有划分）
    train_df = train_df.drop_duplicates(subset=['user', 'ID', 'score'], keep=False)

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


# 探索性数据分析（Exploratory Data Analysis ,EDA）
def EDA(result_df, input_csv=False):
    plt.figure(figsize=(16, 9))  # figsize可以设置保存图片的比例
    if input_csv is True:
        result_df = pd.read_csv(FILE_PATH + 'trainset_plus.csv')
    col_list = result_df.columns
    print(col_list)

    # 绘制子图
    plt.subplot(231)
    plt.xlabel('atbt1_log')
    plt.ylabel('score')
    plt.scatter(y=result_df['score'], x=result_df['atbt1_log'])
    plt.subplot(232)
    plt.xlabel('atbt2_log')
    plt.ylabel('score')
    plt.scatter(y=result_df['score'], x=result_df['atbt2_log'])
    plt.subplot(233)
    plt.xlabel('atbt1_normalized')
    plt.ylabel('score')
    plt.scatter(y=result_df['score'], x=result_df['atbt1_normalized'])
    plt.subplot(234)
    plt.xlabel('atbt2_normalized')
    plt.ylabel('score')
    plt.scatter(y=result_df['score'], x=result_df['atbt2_normalized'])
    plt.subplot(235)
    plt.xlabel('atbt1+atbt2')
    plt.ylabel('score')
    plt.scatter(y=result_df['score'], x=result_df['atbt1+atbt2'])
    plt.subplot(236)
    plt.xlabel('atbt1/atbt2')
    plt.ylabel('score')
    plt.scatter(y=result_df['score'], x=result_df['atbt1/atbt2'])

    plt.title('EDA')
    # 保存图片
    plt.savefig('img/EDA.png')
    plt.show()

    return 0


# # 数据划分，将train划分为数据集，验证集，测试集（20%）
# def divide_data(train_data, input_csv=False, output_csv=False):
#     print('begin divide data')
#     # 告诉文本阅读器，文本的格式是怎么样的
#     reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
#     # 从csv中加载文件
#     if input_csv is True:
#         # 指定文件所在路径
#         # file_path = os.path.expanduser(FILE_PATH + 'train.csv')
#         # 加载数据
#         # train_cf = Dataset.load_from_file(file_path, reader=reader)
#
#         # 导入train.csv 的训练集 as df
#         # 和上面的格式不同
#         trainset_df = pd.read_csv(FILE_PATH + 'train.csv')
#         # train_cf2 = train_cf.build_full_trainset()
#     else:  # 从以有得df中加载
#         train_cf = Dataset.load_from_df(train_data, reader=reader)
#         trainset_df = train_data
#
#
#     # # 划分训练集和测试集
#     # trainset, testset = train_test_split(train_cf, test_size=.20, random_state=SEED)
#     # # testset 转为df格式
#     # testset = pd.DataFrame(data=testset, columns=['user', 'ID', 'score'])
#     # # 用来求得trainset_df
#     # trainset_df = trainset_df.append(testset)
#     # # 所有的train数据减去其中的test数据就是所得剩下的trainset（此trainset包含着验证集，也就是说验证集还没有划分）
#     # trainset_df = trainset_df.drop_duplicates(subset=['user', 'ID', 'score'], keep=False)
#
#     # 打印相关信息
#     print('-----------trainset_df.head----------')
#     print(trainset_df.head())
#
#     # 此时划分后的user 和 ID 即item_id 为字符串类型的，所以不会进行统计，只出现score一列
#     print('---------trainset_df.describe----------')
#     print(trainset_df.describe())
#
#     print('-----------testset.head---------')
#     # print(testset.head())
#     # 此时划分后的user 和 ID 即item_id 为字符串类型的，所以不会进行统计，只出现score一列
#     print('---------testset.describe---------')
#     # print(testset.describe())
#
#     # 将训练集和测试集保存下来
#     if output_csv is True:
#
#         # set_index 用于去掉最左边的索引列
#         testset.set_index('user', inplace=True)
#         # 保存testset.csv
#         testset.to_csv(FILE_PATH + 'testset.csv')
#
#         trainset_df.set_index('user', inplace=True)
#         # 保存为trainset.csv
#         trainset_df.to_csv(FILE_PATH+'trainset.csv')
#
#         # # 保存trianset为pickle
#         # with open(FILE_PATH + 'trainset.pickle', 'wb') as handle:
#         #     pickle.dump(trainset, handle)
#
#     print('divide data finish')
#
#     # 返回trainset_df df 格式 ，testset df格式
#     # trainset Trainset 格式 用于surprise包进行user cf 的处理
#     return trainset_df, testset


# 0模型
def zero_model(testset, input_csv=False):
    print('zero model begin')
    if input_csv is True:
        testset = pd.read_csv(FILE_PATH+'testset.csv')

    # 注意直接使用 test_predict1 = testset为浅拷贝
    # 使用testset.copy(deep=True) 时为深拷贝
    # 使用testset.copy(deep=False) 时为浅拷贝，相当于 test_predict1 = testset
    test_predict1 = testset.copy(deep=True)
    test_predict2 = testset.copy(deep=True)

    test_predict1['score'] = testset['score'].mean()
    test_predict2['score'] = 50
    print('-----------test_predict first detail -----------')
    print(test_predict1.head())
    print(test_predict1.describe())

    print('-----------test_predict second detail -----------')
    print(test_predict2.head())
    print(test_predict2.describe())

    # 计算两种的rmse
    # zero_model_rmse1 38.223879691036416
    # 所得模型的rmse 必须比该值低才有效果
    zero_model_rmse1 = sqrt(mean_squared_error(testset['score'],test_predict1['score']))
    # zero_model_rmse2 38.22696157454122
    zero_model_rmse2 = sqrt(mean_squared_error(testset['score'],test_predict2['score']))
    print('--------------zero_model_rmse 1----------------')
    print(zero_model_rmse1)

    print('---------------zero_model_rmse 2----------------')
    print(zero_model_rmse2)
    print('zero model finish')
    return 0


# 基模型
def user_cf(train, input_csv=False, output_csv=False):
    print('begin user_cf')
    user_cf_begin = time.perf_counter()
    # 告诉文本阅读器，文本的格式是怎么样的
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    # 从csv中加载数据
    if input_csv is True:
        # 指定文件所在路径
        file_path = os.path.expanduser(FILE_PATH + 'trainset.csv')
        # 加载数据
        train_cf = Dataset.load_from_file(file_path, reader=reader)
        # train_cf2 = train_cf.build_full_trainset()
    else:
        # 从已有得df中加载数据
        train_cf = Dataset.load_from_df(train, reader=reader)


    # perf = cross_validate(algo, train_cf, verbose=True, measures=['rmse', 'mae'],cv=3)

    # define a cross-validation iterator
    kf = KFold(n_splits=3)  # 定义交叉验证迭代器
    # define svd
    algo = SVD()
    print('------begin train user cf model------------')
    for trainset, testset in kf.split(train_cf):
        # 训练并测试算法
        print('fit begin')
        fit_time_begin = time.perf_counter()

        algo.fit(trainset)

        fit_time_end = time.perf_counter()
        print('fit end')

        print('Running time: %s Seconds' % (fit_time_end - fit_time_begin))

        print('test begin')
        test_time_begin = time.perf_counter()

        predictions = algo.test(testset)

        test_time_end = time.perf_counter()
        print('test end')
        print('Running time: %s Seconds' % (test_time_end - test_time_begin))

        # 计算并打印RMSE
        accuracy.rmse(predictions, verbose=True)

    user_cf_end = time.perf_counter()
    print('Running time: %s Seconds' % (user_cf_begin - user_cf_end))

    return 0


# 强模型和弱模型混合

# 调参过程

# 预测的准确性评估（测试据数据和真实数据之间的差值，RMSE等评估方法）
def estimate(test, really_data):
    return 0


# 评估出来后，保存数据，用以分析，进一步调参

# 预测test数据，写入结果文件
# 加载test文件
def load_test_data(filepath, output_csv=False):
    print('begin load test data ')
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
    test.set_index('user', inplace=True)

    if output_csv is True:
        test.to_csv(FILE_PATH+'test.csv')
    print('load test data finish')
    return test


# 开始预测
def predict(model1, model2, test_data):
    return 0


# 是否已经有数据生成，若已经有数据生成即为False，当第一次执行，或要进行数据修改时改为True
GENERATE_DATA = False


# 主函数
def main():
    if GENERATE_DATA is True:
        # 使用itemAttribute.txt作为测试
        item = load_item(FILE_PATH + 'itemAttribute.txt', frac=raw_fraction, output_csv=True, input_csv=False)
        item = item_data_clearning(item, already_cleaning=False, output_csv=True)
        # item = item_data_construction(item)
        # 使用train.txt作为测试
        train = load_train_data(filepath=FILE_PATH + 'train.txt', frac=raw_fraction, output_pickle=True,
                                input_pickle=False)
        # print(train[0])

        # print(train[0])
        # result_df = []
        # EDA(result_df, input_csv=False)
        train_test_divide(train, input_data=True, output_csv=True, test_size=0.2)


    else:
        # 平常使用item.csv 数据集 速度更快
        # 注意此时的index会重新生成，并非上次保存的index
        # item = load_item(FILE_PATH + 'itemAttribute.txt', frac=raw_fraction, output_csv=False, input_csv=True)
        #
        # item = item_data_clearning(item, already_cleaning=True, output_csv=False)
        # print(item.head())
        # print(item.describe())
        # item =item_data_construction(item, output_csv=False)

        # 平常使用train.pickle 数据集 速度更快

        # train = load_train_data(filepath=FILE_PATH + 'train.txt', frac=raw_fraction, output_pickle=False, input_pickle=True)
        # result_df, train_df = train_data_to_df(train, item, input_csv=False,output_csv=True)
        # print(train[0])
        # result_df = []
        # EDA(result_df, input_csv=False)
        train = []
        train_test_divide(train, input_data=True, output_csv=True,test_size=0.2)
        # train = pickle.load(open(FILE_PATH + 'testset.pickle', 'rb'))
        # print(train)

        # 计算0模型的RMSE，作为一个基准
        test = []
        # zero_model(test, input_csv=True)
        # trainset_df = []
        # user_cf(trainset_df, input_csv=True, output_csv=False)


        # test = load_test_data(FILE_PATH+'test.txt', output_csv=True)
    return 0


if __name__ == '__main__':
    main()
