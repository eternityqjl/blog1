---
title: 机器学习——k近邻(k-NN)算法
date: 2021-05-12 14:57:17
tags: kNN
categories: 机器学习
mathjax: true
---

## 基本概念及原理

k近邻(k-nearest neighbors)算法是一种基本分类和回归方法。

该算法是给定一个**训练数据集**，对新的**输入测试集**，在训练集中找到与该测试实例**最邻近**的k个实例，这k个实例的多数属于某个类，就把该输入实例分类到这个类中。

![220px-KnnClassification.svg](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/220px-KnnClassification.svg.png)

有两类不同样本分别用红色三角形和蓝色正方形表示，途中绿色圆形为待分类的数据。这时我们根据k-近邻的思想进行分类。

* 当k=3时，判定样本属于红色三角形这一类；
* 当k=5时，判定样本属于蓝色正方形这一类。

<!--more-->

## k的选取及特征归一化

### 选取k值及其影响

选取较小的k值，整体模型会变得更加复杂，容易发生**过拟合**。

> 过拟合就是在训练集上准确率非常高，而在测试集上准确率低。

k太小会导致过拟合，容易将一些噪声学习到模型中。

选取较大的k值，整体模型变得简单，因为当k等于训练样本个数时，无论输入什么测试实例，都将简单地预测它属于**在训练实例中最多的类**，相当于没有训练模型。

所以，模型即不能过大也不能过小，一般选取一个较小的数值，通过采取**交叉验证法**来选取最优的k值，即通过实验调参选取。

### 距离的度量

我们通常使用常见的**欧氏距离**来衡量高维空间中两个点的距离，即：
$$
L_2(x_i,x_j)=(\sum_{l=1}^{n}| x_i^{(l)}-x_j^{(l)} |)^{\frac{1}{2}}
$$
其中，$x_i=(x_i^{(1)}, x_i^{(2)},...,x_i^{(n)})$，同理$x_j$。

### 特征归一化的必要性

如果不进行归一化，让每个特征都同等重要，就会偏向于第一维度的特征，导致多个特征并不是等价重要的，会导致距离计算错误，最终导致预测结果错误。

进行KNN分类使用的样本特征是$\{ (x_{i1}, x_{i2},...,x_{in}) \}_{i=1}^m$，取每个轴上的最大值减去最小值得：
$$
M_j=\max_{i=1,...,m}x_{ij}-\min_{i=1,...,m}x_{ij}
$$
并在计算距离时将每一个坐标轴除以相应的$M_j$进行归一化，即：
$$
d((y_1,...,y_n),(z_1,...,z_n))=\sqrt{\sum_{j=1}^{n}(\frac{y_j}{M_j}-\frac{z_j}{M_j})^2}
$$

## k-NN实现Iris鸢尾花数据集聚类

### K-Means聚类算法的实现步骤：

* 为待聚类的点随机寻找几个聚类中心(类别个数)；
* 计算每个点到聚类中心的距离，将各个点归类到离该点最近的聚类中去；
* 计算每个聚类中所有点的坐标平均值，并将这个平均值作为新的聚类中心，反复执行上一步和该步，直到聚类中心不再进行大范围移动或聚类迭代次数达到要求位置。

### 代码实现

```python
import numpy as np
import pandas as pd

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)    # 下载iris数据集
#data = pd.read_csv('./data/iris.data.csv', header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']    # 特征及类别名称

X = data.iloc[0:150, 0:4].values
y = data.iloc[0:150, 4].values
y[y == 'Iris-setosa'] = 0                                 # Iris-setosa 输出label用0表示
y[y == 'Iris-versicolor'] = 1                             # Iris-versicolor 输出label用1表示
y[y == 'Iris-virginica'] = 2                              # Iris-virginica 输出label用2表示
X_setosa, y_setosa = X[0:50], y[0:50]                     # Iris-setosa 4个特征
X_versicolor, y_versicolor = X[50:100], y[50:100]         # Iris-versicolor 4个特征
X_virginica, y_virginica = X[100:150], y[100:150]         # Iris-virginica 4个特征


# training set
X_setosa_train = X_setosa[:30, :]
y_setosa_train = y_setosa[:30]
X_versicolor_train = X_versicolor[:30, :]
y_versicolor_train = y_versicolor[:30]
X_virginica_train = X_virginica[:30, :]
y_virginica_train = y_virginica[:30]
X_train = np.vstack([X_setosa_train, X_versicolor_train, X_virginica_train])
y_train = np.hstack([y_setosa_train, y_versicolor_train, y_virginica_train])

# validation set
X_setosa_val = X_setosa[30:40, :]
y_setosa_val = y_setosa[30:40]
X_versicolor_val = X_versicolor[30:40, :]
y_versicolor_val = y_versicolor[30:40]
X_virginica_val = X_virginica[30:40, :]
y_virginica_val = y_virginica[30:40]
X_val = np.vstack([X_setosa_val, X_versicolor_val, X_virginica_val])
y_val = np.hstack([y_setosa_val, y_versicolor_val, y_virginica_val])

# test set
X_setosa_test = X_setosa[40:50, :]
y_setosa_test = y_setosa[40:50]
X_versicolor_test = X_versicolor[40:50, :]
y_versicolor_test = y_versicolor[40:50]
X_virginica_test = X_virginica[40:50, :]
y_virginica_test = y_virginica[40:50]
X_test = np.vstack([X_setosa_test, X_versicolor_test, X_virginica_test])
y_test = np.hstack([y_setosa_test, y_versicolor_test, y_virginica_test])


class KNearestNeighbor(object):
   def __init__(self):
       pass

   # 训练函数
   def train(self, X, y):
       self.X_train = X
       self.y_train = y

   # 预测函数
   def predict(self, X, k=1):
       # 计算L2距离
       num_test = X.shape[0]
       num_train = self.X_train.shape[0]
       dists = np.zeros((num_test, num_train))    # 初始化距离函数
       # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
       d1 = -2 * np.dot(X, self.X_train.T)    # shape (num_test, num_train)
       d2 = np.sum(np.square(X), axis=1, keepdims=True)    # shape (num_test, 1)
       d3 = np.sum(np.square(self.X_train), axis=1)    # shape (1, num_train)
       dist = np.sqrt(d1 + d2 + d3)
       # 根据K值，选择最可能属于的类别
       y_pred = np.zeros(num_test)
       for i in range(num_test):
           dist_k_min = np.argsort(dist[i])[:k]    # 最近邻k个实例位置
           y_kclose = self.y_train[dist_k_min]     # 最近邻k个实例对应的标签
           y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))    # 找出k个标签中从属类别最多的作为预测类别

       return y_pred

if __name__ == "__main__":
    KNN = KNearestNeighbor()
    KNN.train(X_train, y_train)
    y_pred = KNN.predict(X_test, k=6)
    accuracy = np.mean(y_pred == y_test)
    print('测试集预测准确率：%f' % accuracy)
```

结果如下：

```
测试集预测准确率：1.000000
```
