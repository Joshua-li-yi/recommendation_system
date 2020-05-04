import numpy as np
import pandas as pd
# 垃圾回收，内存管理
import gc

# 全局变量
# 设置随机数种子
SEED = 1
# 设置取的数据的比例
raw_fraction = 0.01
# 文件位置
FILE_PATH = 'data/'


# 加载item属性数据
def load_item(filepath, output_csv=False,frac=1. ):
    print("begin load data")
    txt = np.loadtxt(filepath, dtype=str, delimiter='|')
    item = pd.DataFrame(data=txt, columns=['ID', 'attribute1', 'attribute2'])
    print(item.describe())
    if frac != 1.:
        print('random select', frac * 100, '% data')
        item = item.sample(frac=frac, random_state=SEED)

    # 将None替换为0
    item.replace('None', 0, inplace=True)

    if output_csv is True:
        item.to_csv(FILE_PATH+'item.csv')
    print(item.describe())
    # 类型转换
    item['ID'] = item['ID'].astype(int)
    item['attribute1'] = item['attribute1'].astype(int)
    item['attribute2'] = item['attribute2'].astype(int)
    return item,

# 加载train_data
def load_train_data(filepath, output_csv=False,frac=1. ):
   return 0

# 数据清洗

# 特征工程
# 特征构建，特征提取，特征选择

# 数据划分，将train划分为数据集，验证集，测试集（20%）
def divide_data(train_data):
    test = train_data.sample(0.2)
    # 不同的调参过程选择不同的划分方式
    train = train_data
    verificate = train_data
    return train,verificate,test
# 0模型
def Zero_model(test):
    return 0
# 基模型

# 强模型和弱模型混合

# 调参过程

# 预测的准确性评估（测试据数据和真实数据之间的差值，RMSE等评估方法）
def estimate(test,really_data):
    return 0


def Zero_model_estimate(model,test):
    return


# 评估出来后，保存数据，用以分析，进一步调参

# 预测test数据，写入结果文件
# 加载test文件

def load_test_data(filepath,frac=1. ):
    return 0

# 开始预测
def predict(model1,model2,test_data):
    return 0

# 主函数
def main():
    return 0

if __name__ == '__main__':
    item = load_item(FILE_PATH+'itemAttribute.txt', frac=raw_fraction, output_csv=True)

