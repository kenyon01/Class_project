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

