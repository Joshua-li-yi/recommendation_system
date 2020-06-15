<!-- TOC -->

- [一、**问题描述**](#一问题描述)
- [二、**环境介绍**](#二环境介绍)
- [三、**数据集基础分析**](#三数据集基础分析)
- [四、**实验原理**](#四实验原理)
- [五、**完成情况**](#五完成情况)
- [六、**实现思路及流程**](#六实现思路及流程)
  - [1、 数据预处理](#1-数据预处理)
  - [2、transet和testset划分](#2transet和testset划分)
  - [3、 零模型](#3-零模型)
  - [4、SVD(0-100)](#4svd0-100)
  - [5、user cf(1-5)](#5user-cf1-5)
  - [6、svd(1-5)](#6svd1-5)
  - [7、EDA](#7eda)
    - [（1）0-100数据的基本分布情况](#10-100数据的基本分布情况)
    - [（2）1-5的数据分布情况](#21-5的数据分布情况)
    - [（3）矩阵稀疏程度](#3矩阵稀疏程度)
  - [8、 生成预测的分数](#8-生成预测的分数)
- [七、**算法实现关键代码**](#七算法实现关键代码)
  - [1、 较科学的模型评估](#1-较科学的模型评估)
  - [2、 模型调参过程](#2-模型调参过程)
  - [3、将评分0-100转化为1-5](#3将评分0-100转化为1-5)
  - [4、SVD模型的选择](#4svd模型的选择)
- [八、**实验结果分析**](#八实验结果分析)
- [九、**遇到困难及解决方案**](#九遇到困难及解决方案)

<!-- /TOC -->

<h1><center>推荐系统实验报告</center></h1>
<center>1811471陈燕__1811379李毅__1811482姜欣妮</center>

### 一、**问题描述**

利用所学算法预测出测试集中每位用户对于给定商品的打分值，并对于预测结果进行误差分析，可以使用任何课内或课外所学算法
<br>
<br>

### 二、**环境介绍**	

1、 语言：`Python`

2、 项目管理工具：`git`

4、Python运行环境：`google colab`，`pycharm`,`surprise`, `pycaret`

<br>
<br>

### 三、**数据集基础分析**

1、数据集说明及格式：

    train.txt   用于训练模型
    <user id>|<numbers of rating items>
    <item id>   <score>
    
    test.txt    用于验证模型
    <user id>|<numbers of rating items>
    <item id>
    
    itemAttribute.txt   用于训练模型(可选)
    <item id>|<attribute_1>|<attribute_2>('None' means this item is not belong to any of attribute_1/2)


2、数据集统计信息

（1）训练集

    用户数量：19835
    商品数量：455309
    评分数量：5001507
    min|max item id：0|6249600
    min|max user id：0|19834
    min|max rating score： 0|100


（2）测试集

    用户数量：19835
    min|max item id:34|624958
    min|max user id:0|19834
    待评价数量:119010


（3）商品集

    商品数量：507172
    min|max item id: 0|624960
    min|max attribute1: 0|624934
    min|max attribute2: 0|624951
    
    mean attribute1: 288394.873765
    mean attribute2: 272492.800332
    
    std attribute1: 193840.412811
    std attribute2: 197227.095516
    
    attribute1 none: 42240
    attribute2 none:  63485
    商品属性两个都为None的数量：117789

<br>
<br>

### 四、**实验原理**

1、SVD算法

存在一个评分矩阵A，每行代表一个user，每列代表一个item，其中的元素表示user对item的打分，空表示user未对item打分，也就是我们需要预测的值，可以知道，矩阵A是一个极其稀疏的矩阵。 矩阵A可分解为矩阵乘积：

$$
R _ { U \times I } = P _ { U \times K } Q _ { K \times I }
$$

 U表示用户数，I表示商品数。然后就是利用R中的已知评分训练P和Q使得P和Q相乘的结果最好地拟合已知的评分，那么未知的评分也就可以用P的某一行乘上Q的某一列得到了： 
$$
\hat { r } _ { u i } = p _ { u } ^ { T } q _ { i }
$$
 这是预测用户u对商品i的评分，它等于P矩阵的第u行乘上Q矩阵的第i列。这时需要通过已知评分训练得到P和Q的具体数值，假设已知的评分为： $r _ { u i }$

则真实值与预测值的误差为：  

$$e _ { u i } = r _ { u i } - \hat { r } _ { u i }$$  

继而可以计算出总的误差平方和：  
$$
SSE = \sum _ { u , i } e _ { u i } ^ { 2 } = \sum _ { u , i } ( r _ { u i } - \sum _ { k = 1 } ^ { K } p _ { u k } q _ { k i } ) ^ { 2 }
$$
只要通过训练把SSE降到最小，那么P、Q就能最好地拟合R了。 利用梯度下降法可以求得SSE在Puk变量（也就是P矩阵的第u行第k列的值）处的梯度： 
$$
\frac { \partial } { \partial p _ { u k } } S S E = \frac { \partial } { \partial p _ { u k } } ( e _ { u i } ^ { 2 } )
$$
 求导后有： 
$$
\frac { \partial } { \partial p _ { u k } } S S E = \frac { \partial } { \partial p _ { u k } } ( e _ { u i } ^ { 2 } ) = 2 e _ { u i } \frac { \partial } { \partial p _ { u k } } e _ { u i } = - 2 e _ { u i } q _ { k i }
$$

$$
S S E = \frac { 1 } { 2 } \sum _ { u , i } e _ { u i } ^ { 2 } = \frac { 1 } { 2 } \sum _ { u , i } ( r _ { u i } - \sum _ { k = 1 } ^ { K } p _ { u k } q _ { k i } ) ^ { 2 }
$$

$$
\frac { \partial } { \partial p _ { u k } } S S E = - e _ { u i } q _ { k i }
$$

现在得到了目标函数在Puk处的梯度了，那么按照梯度下降法，将Puk往负梯度方向变化： 令更新的步长（也就是学习速率）为  $\eta$ 则Puk的更新式为 
$$
p _ { u k } : = p _ { u k } - \eta ( - e _ { u i } q _ { k i } ) : = p _ { u k } + \eta e _ { u i } q _ { k i }
$$
同样的方式可得到Qik的更新式为 
$$
q _ { k i } : = q _ { k i } - \eta ( - e _ { u i } p _ { u k } ) : = q _ { k i } + \eta e _ { u i } p _ { u k }
$$
使用随机梯度下降法，每计算完一个$e _ { u i }$后立即对pu和qi进行更新。

2、`UserCF`算法p

基于用户的协同过滤算法。

核心思想：

找到和用户A喜好相似的其他用户集合，根据用户集合的打分对预测物品进行打分。

用户相似度：

计算(shrunk)皮尔逊相关系数之间的所有对用户(或项目)使用基线居中，而不是平均值。

shrunk值可以用来帮助降低过拟合



公式如下

$$
\text { pearson_baseline_sim }(u, v)=\hat{\rho}_{u v}=\frac{\sum_{i \in I_{u v}}\left(r_{u i}-b_{u i}\right) \cdot\left(r_{v i}-b_{v i}\right)}{\sqrt{\sum_{i \in I_{u v}}\left(r_{u i}-b_{u i}\right)^{2}} \cdot \sqrt{\sum_{i \in I_{u v}}\left(r_{v i}-b_{v i}\right)^{2}}}
$$
or
$$
\text { pearson_baseline_sim }(i, j)=\hat{\rho}_{i j}=\frac{\sum_{u \in U_{i j}}\left(r_{u i}-b_{u i}\right) \cdot\left(r_{u j}-b_{u j}\right)}{\sqrt{\sum_{u \in U_{i j}}\left(r_{u i}-b_{u i}\right)^{2}} \cdot \sqrt{\sum_{u \in U_{i j}}\left(r_{u j}-b_{u j}\right)^{2}}}
$$

$$
\text{pearson_baseline_shrunk_sim }(u, v)=\frac{\left|I_{u v}\right|-1}{\left|I_{u v}\right|-1+\text { shrinkage }} \cdot \hat{\rho}_{u v}
$$

$$
\text {pearson_baseline_shrunk_sim} (i, j)=\frac{\left|U_{i j}\right|-1}{\left|U_{i j}\right|-1+\text { shrinkage }} \cdot \hat{\rho}_{i j}
$$



3、`StackNet`

`StackNet`是一种计算性的、可扩展的、分析性的`meta-modeling`框架，它类似于前后向的神经网络，并使用`Wolpert`的多层`stacked gengeralization`来提高机器学习预测问题的准确性。

`StackNet`大概可以分为如下几个基本性能：

+ Computational拥有很强的计算能力

+ Scalable多个模型能够被并行运行，多线程将会使得结果更快

+ Analytical很大程度上基于数据分析(或数据科学)的原理，特别是当涉及到数据预处理、交叉验证、通过各种度量来度量性能的时候。

+ Meta-modelling引入meta learners的概念。换句话说，它将一些算法的预测输出作为其他算法的输入特征。

+ Wolpert’s stacked generalization因为meta-learners是用在hold-out 数据集中的预测结果组合技术来创建。

+ Feedforward neural network and multiple levels Stacking并不限于stacking部分这个四个阶段，而是它能够在预测中重复多次创造更多的数据集，比如B2,C2直到Bn，Cn。

<br>
<br>

### 五、**完成情况**

1、使用、对比多种模型，0模型、基于用户的协同过滤模型、SVD、ML建模回归，最终选取的模型为SVD，最终在testset（自己划分）上RMSE分数为13.87345509156839

2、使用了ItemAttribute.txt 数据集并对模型进行了一定程度上的优化
<br>
<br>

### 六、**实现思路及流程**

#### 1、 数据预处理

将`train.txt`转为`dict`类型并保存为`pickle`文件


格式`{use_id:{ item_id: score }}`

代码：

```python
import pickle
# 加载train_data 数据类型dict嵌套
# {use_ed:{item_id:score}}
def load_train_data(filepath, output_pickle=False, input_pickle=False):
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
                print(item)
                train[id] = item

    # print(train)

    # 使用dump()将数据序列化到文件中
    if output_pickle is True:
        with open(FILE_PATH + 'train.pickle', 'wb') as handle:
            pickle.dump(train, handle)
    print('load train data finish')

    return train
```


#### 2、transet和testset划分

划分方式在`train.pickle`文件中
从每个用户的评分中随机选取20%的数据，最后整合为一整个testset
而其他的作为`trainset`

之后的分析和训练过程全部只运用`trainset`


`testset`只作为最后的模型评估与评价，因此不会造成数据泄露等问题

最后在预测`test.txt`文件里的数据的时候，会将`trainset`和`testset`作为全部的训练集

从理论上说，在真正`test`数据集上的分数要比在`testset`上的评估分数要高一些。

代码

```python
# 随机选择的包
from numpy.random import choice
# 向下取整
from math import floor
import numpy as np
import pandas as pd
import pickle

# 将dict类型的train数据转为df类型,存到trainset.csv 和testset.csv
# test_size = 0.2 选取的测试集的比例
def train_test_divide(train=[], output_csv=False, input_data=False, test_size=0.2):
    print('begin divide train and test')
    if input_data is True:
        # 导入所有的数据集
        train = pickle.load(open(FILE_PATH + 'train.pickle', 'rb'))

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

    # 先扩展再去重，得到trian
    train_df = total_df.append(test_df)
    train_df['user'] = train_df['user'].astype(int)
    # 所有的train数据减去重复的就是所得剩下的train（此train包含着验证集，也就是说验证集还没有划分）
    train_df.drop_duplicates(subset=['user', 'ID', 'score'], keep=False, inplace=True)

    if output_csv is True:
        print('---------save as csv----------')
        total_df.set_index('user', inplace=True)
        total_df.to_csv(FILE_PATH + 'train.csv')
        del total_df
        train_df.set_index('user', inplace=True)
        train_df.to_csv(FILE_PATH + 'trainset.csv')
        del train_df
        test_df.set_index('user', inplace=True)
        test_df.to_csv(FILE_PATH + 'testset.csv')
        del test_df
    print('divide train and test end ')
```

`trainset,` `testset`基本数据分布

```
trainset
    用户数量：19835
    商品数量：455309
	评分数量：4008915
	min|max item id：1|6249600
	min|max user id：0|19834
	min|max rating score： 0|100
testset
    用户数量：19835
    商品数量：455309
	评分数量：992592
	min|max item id：0|6249600
	min|max user id：0|19834
	min|max rating score： 0|100
```

#### 3、 零模型

我们选取了`trainset`中`score`的均值，50，中位数，众数作为基准，计算了testset中的RMSE值分别为

```
mean rmse:  38.22805419301362
50 rmse:  38.23329255713531
median rmse:  38.23329255713531
freq rmse:  62.37919526513844
```

由此看出我们之后选取的模型，最后的RMSE值需要低于`38.22805419301362`

代码

```python
from sklearn.metrics import mean_squared_error
import pandas as pd
from math import sqrt
# 0模型
def zero_model(testset=[], input_csv=False):
    print('zero model begin')
    if input_csv is True:
        testset = pd.read_csv(FILE_PATH + 'testset.csv')
        trainset = pd.read_csv(FILE_PATH + 'trainset.csv')
    # 注意直接使用 test_predict1 = testset为浅拷贝
    # 使用testset.copy(deep=True) 时为深拷贝
    # 使用testset.copy(deep=False) 时为浅拷贝，相当于 test_predict1 = testset
    real_score = testset['score']
    test_predict = testset.copy(deep=True)
    del testset
    # 均值
    test_predict['pred_mean'] = trainset['score'].mean()
    test_predict['pred_50'] = 50
    # 中值
    test_predict['pred_median'] = trainset['score'].median()
    # 众值
    test_predict['pred_freq'] = trainset['score'].mode()[0]
    # 计算两种的rmse
    # zero_model_rmse1 38.223879691036416
    # 所得模型的rmse 必须比该值低才有效果
    zero_model_rmse1 = sqrt(mean_squared_error(real_score,test_predict['pred_mean']))
    # zero_model_rmse2 38.22696157454122
    zero_model_rmse2 = sqrt(mean_squared_error(real_score,test_predict['pred_50']))
    # 38.23329255713531
    zero_model_rmse3 = sqrt(mean_squared_error(real_score,test_predict['pred_median']))
    # 62.3791952651384
    zero_model_rmse4 = sqrt(mean_squared_error(real_score,test_predict['pred_freq']))
    print('mean rmse: ',zero_model_rmse1, '50 rmse: ', zero_model_rmse2, 'median rmse: ', zero_model_rmse3, 'freq rmse: ', zero_model_rmse4)

    print('zero model finish')
```

#### 4、SVD(0-100)

```python
# 基模型
def svd(train=[], input_csv=False, output_csv=False):
    print('begin svd')
    user_cf_begin = time.perf_counter()
    # 告诉文本阅读器，文本的格式是怎么样的
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1,rating_scale=(0,100))
    # 从csv中加载数据
    if input_csv is True:
        # 指定文件所在路径
        file_path = os.path.expanduser('trainset.csv')
        # 加载数据
        train_cf = Dataset.load_from_file(file_path, reader=reader)
        # 如果训练所有数据取消掉下面的注释
        #---------------------------------------------------
        trainset = train_cf.build_full_trainset()
        #---------------------------------------------------
    else:
        # 从已有得df中加载数据
        train_cf = Dataset.load_from_df(train, reader=reader)

    algo = SVD(n_epochs=5, lr_all=0.002, reg_all=0.2, n_factors=650)
    print('fit begin')
    fit_time_begin = time.perf_counter()

    algo.fit(trainset)

    fit_time_end = time.perf_counter()
    print('fit end')
    print('Running time: %s Seconds' % (fit_time_end - fit_time_begin))

    user_cf_end = time.perf_counter()
    print('Running time: %s Seconds' % (user_cf_end - user_cf_begin))
    return algo

algo = svd(trainset_df, input_csv=True, output_csv=False)

test = pd.read_csv('testset.csv')
# 加载真实的score数据
test_score = test['score'].tolist()
# 遍历测试集进行预测
pred =[]
for row in test.itertuples():
  # 注意这里一定要 把 user ， item_id 转为str格式的
  pred.append(algo.predict(str(row[1]), str(row[2]), r_ui=row[3]).est)
del algo
# 四舍五入
pred_round = np.round(pred)
from sklearn.metrics import mean_squared_error
# 计算rmse
rmse = np.sqrt(mean_squared_error(test_score,pred_round))
print('rmse on test scale 0-100:', rmse)
```

经手动+网格搜索调参后目前为止最好的参数为

```
n_epochs=5, lr_all=0.002, reg_all=0.2, n_factors=650
```

运行时间以及RMSE为

```
Running time: 512.2356659339998 Seconds
rmse on test scale 0-100: 26.8254386470559
```

#### 5、user cf(1-5)

尝试了将`0-100`的scale转换到`1-5` , `0-5` ，`1-10`，`0-10`多种类别上

发现在1-5上的表现最好

将rating scale转换到1-5上

```python
def scale5(score):
  if 0 <= score < 20:
    return 1
  elif 20 <= score < 40:
    return 2
  elif 40 <= score < 60:
    return 3
  elif 60 <= score < 80:
    return 4
  elif 80 <= score <=100:
    return 5 
```

<br>
<br>

```python
# 基模型
def user_cf(train=[], input_csv=False):
    print('begin user cf')
    begin = time.perf_counter()
    # 告诉文本阅读器，文本的格式是怎么样的
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1,rating_scale=(1,5))
    # 从csv中加载数据
    if input_csv is True:
        # 指定文件所在路径
        file_path = os.path.expanduser('trainset_score[1,5].csv')
        # 加载数据
        train_cf = Dataset.load_from_file(file_path, reader=reader)
        # 如果训练所有数据取消掉下面的注释
        #---------------------------------------------------
        trainset = train_cf.build_full_trainset()
        #---------------------------------------------------
    else:
        # 从已有得df中加载数据
        train_cf = Dataset.load_from_df(train, reader=reader)

    sim_options = {'name': 'pearson_baseline',
                  'shrinkage': 100  #  shrinkage,防止过拟合
               }
    algo = KNNWithMeans(k=40,sim_options=sim_options) # 1.52

    print('fit begin')
    fit_time_begin = time.perf_counter()

    algo.fit(trainset)

    fit_time_end = time.perf_counter()
    print('fit end')
    print('Running time: %s Seconds' % (fit_time_end - fit_time_begin))

    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - begin))
    return algo

algo = user_cf(input_csv=True)
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
  return switcher.get(score,50)
  
total = pd.read_csv('train.csv')
test = pd.read_csv('testset_score[1,5].csv')
# 合并两个df
temp_test = pd.merge(total, test, on=['user','ID'], how='inner')
# print(temp_test.head(10))
del total,test
# 加载真实的score数据
test_score = temp_test['score'].tolist()
# 遍历测试集进行预测
pred =[]
for row in temp_test.itertuples():
  # 注意这里一定要 把 user ， item_id 转为str格式的
  pred.append(algo.predict(str(row[1]), str(row[2]), r_ui=row[4]).est)
# 计算在1-5评分上的rmse
rmse_test = np.sqrt(mean_squared_error(temp_test['score[1,5]'].tolist(),pred))
print('rmse on test scale[1,5]:', rmse_test)
# 四舍五入
pred_round = np.round(pred)
# 从1-5转到原来的数据
pred_score = []
for p in pred_round:
  # 先转化为int
  pred_score.append(rescale1_5(int(p)))
from sklearn.metrics import mean_squared_error
# 计算rmse
rmse = np.sqrt(mean_squared_error(test_score,pred_score))
print('rmse on test scale 0-100:', rmse)
```



结果

```
Running time: 428.300035019 Seconds
rmse on test scale[1,5]: 0.7787991038365966
rmse on test scale 0-100: 19.330745525426316
```


#### 6、svd(1-5)

尝试了将`0-100`的scale转换到`1-5` , `0-5` ，`1-10`，`0-10`多种类别上

发现在1-5上的表现最好

```python
# 基模型
def svd(train=[], input_csv=False):
    print('begin svd')
    begin = time.perf_counter()
    # 告诉文本阅读器，文本的格式是怎么样的
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1,rating_scale=(1,5))
    # 从csv中加载数据
    if input_csv is True:
        # 指定文件所在路径
        file_path = os.path.expanduser('trainset_score[1,5].csv')
        # 加载数据
        train_cf = Dataset.load_from_file(file_path, reader=reader)
        # 如果训练所有数据取消掉下面的注释
        #---------------------------------------------------
        trainset = train_cf.build_full_trainset()
        #---------------------------------------------------
    else:
        # 从已有得df中加载数据
        train_cf = Dataset.load_from_df(train, reader=reader)

    algo = SVD(n_epochs=350, lr_all=0.003, reg_all=0.01, n_factors=250)
    print('fit begin')
    fit_time_begin = time.perf_counter()

    algo.fit(trainset)

    fit_time_end = time.perf_counter()
    print('fit end')
    print('Running time: %s Seconds' % (fit_time_end - fit_time_begin))

    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - begin))
    return algo

algo = svd(input_csv=True)
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
  return switcher.get(score,50)
  
total = pd.read_csv('train.csv')
test = pd.read_csv('testset_score[1,5].csv')
# 合并两个df
temp_test = pd.merge(total, test, on=['user','ID'], how='inner')
# print(temp_test.head(10))
del total,test
# 加载真实的score数据
test_score = temp_test['score'].tolist()
# 遍历测试集进行预测
pred =[]
for row in temp_test.itertuples():
  # 注意这里一定要 把 user ， item_id 转为str格式的
  pred.append(algo2.predict(str(row[1]), str(row[2]), r_ui=row[4]).est)
# 计算在1-5评分上的rmse
rmse_test = np.sqrt(mean_squared_error(temp_test['score[1,5]'].tolist(),pred))
print('rmse on test scale[1,5]:', rmse_test)
# 四舍五入
pred_round = np.round(pred)
# 从1-5转到原来的数据
pred_score = []
for p in pred_round:
  # 先转化为int
  pred_score.append(rescale1_5(int(p)))
from sklearn.metrics import mean_squared_error
# 计算rmse
rmse = np.sqrt(mean_squared_error(test_score,pred_score))
print('rmse on test scale 0-100:', rmse)
```

模型结果为


```
Running time: 8831 Seconds
rmse on test scale[1,5]:0.541747156173787
rmse on test scale0-100:13.87345509156839
```

#### 7、EDA

##### （1）0-100数据的基本分布情况


![](https://www.showdoc.cc/server/api/common/visitfile/sign/bbc7b100011879f23b07aacd4dd43236?showdoc=.jpg)

##### （2）1-5的数据分布情况

```
score[1,5]         
5           1448238
1           1255993
4            485706
3            473685
2            345293
```

##### （3）矩阵稀疏程度


99.9676581902835%


```python
train = pd.read_csv('trainset.csv')
item_max = train['ID'].max()
user_max = train['user'].max()
length = len(train['user'])
# 计算矩阵的稀疏程度
sparsity = 1-length/(item_max*user_max)
```


#### 8、 生成预测的分数

```python
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
```

```python
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
  return switcher.get(score,50)
  
test = pd.read_csv('test.csv')
# 遍历测试集进行预测
pred =[]
for row in test.itertuples():
  # 注意这里一定要 把 user ， item_id 转为str格式的
  pred.append(algo2.predict(str(row[1]), str(row[2]), r_ui=row[3]).est)
# 四舍五入
pred_round = np.round(pred)
# 从1-5转到原来的数据
pred_score = []
for p in pred_round:
  # 先转化为int
  pred_score.append(rescale1_5(int(p)))

test['pred'] = pred_score
test.drop('score', axis=1, inplace=True)
test.set_index('user', inplace=True)
# 保存为csv
test.to_csv('submit.csv')

test = pd.read_csv('submit.csv')
# 写入text
with open("svd.txt","w") as f:
  temp_user = 1
  for row in test.itertuples():
    if temp_user != row[1]:
      f.write(str(row[1])+'|6\n')
      temp_user = row[1]
    f.write(str(row[2])+ "  "+ str(row[3])+"\n")
```

<br>
<br>

### 七、**算法实现关键代码**

#### 1、 较科学的模型评估

`trainset` 和`testset`的划分

划分方式在`train.pickle`文件中
从每个用户的评分中随机选取`20%`的数据，最后整合为一整个`testset`
而其他的作为`trainset`, 之后的分析和训练过程全部只运用`trainset`，
`testset`（992592条评分）只作为最后的模型评估与评价，因此不会造成数据泄露等问题

最后在预测`test.txt`文件里的数据的时候，会将`trainset`和`testset`作为全部的训练集，从理论上说，在真正`test`数据集上的分数要比在`testset`上的评估分数要高一些。



代码
划分trainset和testset

```python
# 随机选择的包
from numpy.random import choice
# 向下取整
from math import floor
import numpy as np
import pandas as pd
import pickle

# 将dict类型的train数据转为df类型,存到trainset.csv 和testset.csv
# test_size = 0.2 选取的测试集的比例
def train_test_divide(train=[], output_csv=False, input_data=False, test_size=0.2):
    print('begin divide train and test')
    if input_data is True:
        # 导入所有的数据集
        train = pickle.load(open(FILE_PATH + 'train.pickle', 'rb'))

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

    # 先扩展再去重，得到trian
    train_df = total_df.append(test_df)
    train_df['user'] = train_df['user'].astype(int)
    # 所有的train数据减去重复的就是所得剩下的train（此train包含着验证集，也就是说验证集还没有划分）
    train_df.drop_duplicates(subset=['user', 'ID', 'score'], keep=False, inplace=True)

    if output_csv is True:
        print('---------save as csv----------')
        total_df.set_index('user', inplace=True)
        total_df.to_csv(FILE_PATH + 'train.csv')
        del total_df
        train_df.set_index('user', inplace=True)
        train_df.to_csv(FILE_PATH + 'trainset.csv')
        del train_df
        test_df.set_index('user', inplace=True)
        test_df.to_csv(FILE_PATH + 'testset.csv')
        del test_df
    print('divide train and test end ')
```

#### 2、 模型调参过程

手动调参与网格调参相结合，加快调参速度

```python
param_grid = {'n_factors': [400, 500, 650], 'n_epochs': [30], 'lr_all': [0.006],
                  'reg_all': [0.01]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(train)
# best RMSE score
print(gs.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
```



#### 3、将评分0-100转化为1-5

由推荐系统先验知识可以想到将`0-100`的scale转换到`1-5` , `0-5` ，`1-10`，`0-10`多种类别上

经过验证后在1-5，五级评分上模型表现较好

将评分0-100转化为1-5五级评分，提升了模型的精度，最后将1-5再转化为0-100，不影响最后的结果

也是该模型RMSE降到20以下的一个关键点

```python
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
  return switcher.get(score,50)
```

#### 4、SVD模型的选择

使用google colaboratory 免费的12.72GB RAM，以及107.77GB Disk 给予了本次实验的算力支持，使本次实验可以迭代多次

最终选择 lr_all=0.003, reg_all=0.01, n_factors=250 迭代350次

是本次实验可以RMSE降到15以下的一个关键点
<br>
<br>

<br>
<br>

<br>
<br>

### 八、**实验结果分析**

1、

|    model     |    RMSE(0-100)     |      Time(s)      |
| :----------: | :----------------: | :---------------: |
|   0 model    | 38.22805419301362  |        --         |
|  SVD(0-100)  |  26.8254386470559  | 512.2356659339998 |
| user cf(1-5) | 19.330745525426316 |   428.300035019   |
|      ML      | 17.251743221583663 |        --         |
|   SVD(1-5)   | 13.87345509156839  |  8831.020578701   |

空间消耗（SVD）

空间复杂度: 由于采用稀疏矩阵来存储 user 与 item 之间的打分关系，故空间复杂度即 O(n)
实际上在整个程序运行时消耗RAM：4G

2、训练集数据本身结构（如0分过多）、矩阵过于稀疏（稀疏率99.9676581902835%）对降低RMSE带来一定的困难

3、所给数据集中只含有item的属性，隐式特征较少，若所给数据集中含有其他的隐式特征，比如用户的点击，用户的浏览时间等的行为，可以选择结合多种算法NCF，显隐式推荐等可使模型的精度进一步提高

### 九、**遇到困难及解决方案**

1、模型无法保存下来，不会使用dump model

+ 解决方法：阅读包源码发现load 之前dump 的model之后是一个(predictions, algo)的元组，找到了问题所在

2、模型的调参时，结果不收敛以及，模型精度提升效果与模型训练时间不成正比

+ 解决方法：多次调整模型及参数，选择更优的配置，提高模型的精度

3、设备硬件不满足训练所需

+ 解决方法：改用google colab平台+本地运行配合使用

4、在使用item属性进行的item相似度计算时，即item CF时，内存消耗过大>25g，没有解决，换用其他方法

5、使用深度学习构建user*item的矩阵进行迭代运算，内存消耗过大，无法实现。

6、 在进行文件打包的时候提示`pkg_resources.DistributionNotFound: The 'scikit-surprise' distribution was not found and is required by the application`

+ 解决办法从网上搜索相应解决办法，最终得以解决