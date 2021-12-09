### Q3：

##### 1.支持向量机

​		超平面是分割输入变量空间的线。在SVM中，选择超平面以最佳地将输入变量空间中的点与它们的类（0或1）分开。在二维中，可以将其视为一条线，并假设所有输入点都可以被这条线完全分开。SVM学习算法找到导致超平面最好地分离类的系数。超平面与最近数据点之间的距离称为边距。可以将两个类分开的最佳或最佳超平面是具有最大边距的线。只有这些点与定义超平面和分类器的构造有关。这些点称为支持向量。它们支持或定义超平面。实际上，优化算法用于找到使裕度最大化的系数的值。

##### 2. 程序结构：

0.算法首先导入一些用到的包，以及做一些设置

1.定义一个进行数据处理的子程序，包括数据的缩放，数据的初始化，数据的洗牌

2.对数据进行预处理，处理一些无法识别的数据，比如本次作业中有“？”的数据。使数据可以被顺利处理

3.创建SVM模型，这里会让由于正则化参数是一个超参数，这里先让模型运行50次，程序将会自动识别到使模型性能最好的超参数，并且将超参数代入到模型中，对数据进行训练。

![image-20211209203220963](https://gitee.com/Kenyon01/markdown-pic/raw/master/img/20211209203220.png)

4.计算训练集和测试集的准确率

![image-20211209203234238](https://gitee.com/Kenyon01/markdown-pic/raw/master/img/20211209203234.png)

5.对无标签的测试集进行预测

![image-20211209203241759](https://gitee.com/Kenyon01/markdown-pic/raw/master/img/20211209203241.png)

[0 1 0 0 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 1 1 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 1 1 0 1 0 0 1 1 1 0 0 1 1 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 1 0 0 1 1 1 1]

##### 3. 代码：

```python
# 0.导入包，设置中文字体和负号正确显示
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import classification_report
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.定义数据处理函数
def preprocess(X_train,y_train):
    # 数据提取
    X = X_train
    y = y_train
    # 特征缩放
    X -= np.mean(X,axis=0)
    X /= np.std(X,axis=0,ddof=1)
    # 数据初始化
    m = len(X)
    X = np.c_[X]
    y = np.c_[y]
    # 洗牌
    np.random.seed(3)
    o = np.random.permutation(m)
    X_train = X[o]
    y_train = y[o]
    return X_train,y_train


#2.进行数据的读取和预处理
X_train = pd.read_table('traindata.txt',header =None,delim_whitespace=True)
y_train = pd.read_table('trainlabel.txt',header =None,delim_whitespace=True)
y_train[y_train > 1] = 1
dirty_I = []
for i in range(X_train.shape[0]):
    if (X_train.iloc[i, :] == '?').any():
        dirty_I.append(i)
X_train = X_train.drop(dirty_I).astype("float")
y_train = y_train.drop(dirty_I)

X_train,y_train = preprocess(X_train,y_train)

# 3.创建svm模型， 算法将会自动选取最优的一个参数

result = []
index = range(1, 50, 1)
for C in index:
    model = SVC(C=C, kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    result.append(score)
plt.plot(list(index), result)
plt.show()
print(np.argmax(result))
model = SVC(C=index[np.argmax(result)],kernel='rbf', probability= True)
print('最好的超参数C是 = ', index[np.argmax(result)])
model.fit(X_train,y_train)


# 4.分别求出训练集和测试集的准确率
print('训练集的准确率是：',model.score(X_train,y_train))
model.score(X_train, y_train)
print('Training accuracy = {0}%'.format(np.round(model.score(X_train, y_train) * 100, 2)))
print(classification_report(y_train,model.predict(X_train)))

# 5.对无标签数据进行预测
X_test = pd.read_table('testdata.txt', header=None, delim_whitespace=True)
for i in range(X_test.shape[1]):
    tmp = (X_test.iloc[:, i] == '?')
    if tmp.any():
        X_test.iloc[X_test[tmp].index, i] = pd.Series(data=X_test.iloc[:, i]).mode()[0]
X_test = X_test.astype("float")
X_test = (X_test - X_test.mean()) / X_test.std()
y_predict = model.predict(X_test)
print('pre_test_labels:')
print(y_predict)
```

##### 4 

1. 代码和文件已经上传至github：https://github.com/kenyon01/Class_project.git

