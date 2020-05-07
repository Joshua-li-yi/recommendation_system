import numpy as np
import pandas as pd
# 垃圾回收，内存管理
import gc
# 打包文件
import pickle

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
    if already_cleaning is True:
        return item
    else:
        print('begin item data cleaning')
        print('------------------item data head----------------------')
        print(item.head())
        print('------------------item data describe----------------------')
        print(item.describe())
        print('------------------item data info ----------------------')
        print(item.info())
        print('item data cleaning finish')
        # none值处理

        # attribute1_avg = item['attribute1'].mean()
        # item['attribute1'].replace(0, attribute1_avg, inplace=True)
        # attribute2_avg = item['attribute2'].mean()
        # item['attribute2'].replace(0, attribute2_avg, inplace=True)

        # 物品属性缺失处理
        # ID max = 624960
        # ID rows = 507172
        # 缺失 624960-507172个，将这些值使用平均值进行填充，

        # 重复值处理

        # 输出为csv
        if output_csv is True:
            item.to_csv(FILE_PATH + 'item.csv')

    return item


# 特征工程
# 特征构建，特征提取，特征选择
def item_data_construction(item_attributes):
    item_attributes.set_index('ID', inplace=True)
    # 正则化

    # 其他
    item_attributes['attribute3'] = item_attributes['attribute1'] + item_attributes['attribute2']

    return item_attributes


# 数据划分，将train划分为数据集，验证集，测试集（20%）
def divide_data(train_data):
    test = train_data.sample(0.2)
    # 不同的调参过程选择不同的划分方式
    train = train_data
    verificate = train_data
    return train, verificate, test


# 0模型
def Zero_model(test):
    return 0


# 基模型

# 强模型和弱模型混合

# 调参过程

# 预测的准确性评估（测试据数据和真实数据之间的差值，RMSE等评估方法）
def estimate(test, really_data):
    return 0


def Zero_model_estimate(model, test):
    return


# 评估出来后，保存数据，用以分析，进一步调参

# 预测test数据，写入结果文件
# 加载test文件

def load_test_data(filepath, frac=1.):
    return 0


# 开始预测
def predict(model1, model2, test_data):
    return 0


# 主函数
def main():
    # 使用itemAttribute.txt作为测试
    # item = load_item(FILE_PATH + 'itemAttribute.txt', frac=raw_fraction, output_csv=True, input_csv=False)
    # 平常使用i tem.csv 数据集 速度更快
    item = load_item(FILE_PATH + 'itemAttribute.txt', frac=raw_fraction, output_csv=False, input_csv=True)
    item = item_data_clearning(item,already_cleaning=False)
    # 使用train.txt作为测试
    # train = load_train_data(filepath=FILE_PATH+'train.txt', frac=raw_fraction, output_pickle=True,input_pickle=False)
    # print(train[0])

    # 平常使用train.pickle 数据集 速度更快
    # train = load_train_data(filepath=FILE_PATH + 'train.txt', frac=raw_fraction, output_pickle=False, input_pickle=True)
    # print(train[0])
    return 0


if __name__ == '__main__':
    main()
