---
title: cs231n——1.Nearest Neighbor Classifier
date: 2021-06-09 17:41:44
tags: ['机器学习','k-NN']
categories: 机器学习
mathjax: true
---

图像分类规范化步骤：

* 输入：输入带有不同标签的图像作为训练集
* 学习：使用训练集学习每种类别的抽象特征
* 评估：通过测试集评估分类器的质量。

## Nearest Neighbor Classifier

CIFAR-10 dataset:

* 10 classes (airplane, automobile, bird, etc)
* 60,000 tiny images that are 32 pixels high and wide

k-NN通过将一张测试集图片与所有的训练集图片进行比较，预测结果。

<!--more-->

### L1 distance:

$$
d_{1}\left(I_{1}, I_{2}\right)=\sum_{p}\left|I_{1}^{p}-I_{2}^{p}\right|
$$

将两张图片分别表示为$I_1, I_2$两个向量，使用L1距离来比较两张图片。

通常使用准确率来评估分类器

```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```



Here is an implementation of a simple Nearest Neighbor classifier with the L1 distance that satisfies this template:

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]	#训练样本数量
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

使用L1距离测试的准确率只有38.6%.

### L2 distance

$$
d_{2}\left(I_{1}, I_{2}\right)=\sqrt{\sum_{p}\left(I_{1}^{p}-I_{2}^{p}\right)^{2}}
$$

only replace a single line of code above:

```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```

## k - Nearest Neighbor Classifier

找到距离最近的k个图片，k个图片中数量最多的一个标签即为测试图片所分的类别。

## Validation sets for Hyperparameter tuning

有很多我们可以选择的距离函数，例如L1 norm, L2 norm等，这些选择称为

Hyperparameter 。通常不能使用test set来调整Hyperparameter。Evaluate on the test set only a single time, at the very end.

将训练集划分为一个小一些的训练集和一个验证集(validation set)。例如使用CIFAR-10数据集时将训练集分为49,000个训练集和1,000个验证集。

验证集本质上是一个假的测试集，是用来调整Hyperparameter的。

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

### Cross-validation

当训练集数量较少时，使用更为复杂的交叉验证来调整hyperparameter。我们通过迭代不同的验证集并平均计算这些验证集的结果来确定k。例如在5次迭代的交叉验证中，将训练集分为5份，使用4份训练，1份验证，然后改变验证集，迭代5次，最终使用5次的平均结果确定k。

## Nearest Neighbor classifier的优缺点

验证时的时间成本较高，每个验证样本都要与每一个测试集样本进行比较，效率较低。

Nearest Neighbor classifier在低维度训练中较为常用，但在图像分类中很少用，因为图像是高维对象，高维对象之间的距离计算式违反直觉的。

## Summery

1. 如果数据的维数很高，可以使用降维技术，例如PCA(主成分分析), NCA(邻域成分分析) or Random Projections(随机投影).

