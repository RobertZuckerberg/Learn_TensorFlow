# 提取花瓣长度、花瓣宽度两个属性进行训练

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"  # 加载训练数据
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
df_iris_train = pd.read_csv(train_path, header=0)

TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"  # 加载测试数据
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)
df_iris_test = pd.read_csv(test_path,header=0)

iris_train = np.array(df_iris_train)
x_train = iris_train[:, 2:4]  # 提取训练集花瓣长度、花瓣宽度属性
y_train = iris_train[:, 4]
num_train = len(x_train)

iris_test = np.array(df_iris_test)
x_test = iris_test[:, 2:4]  # 提取测试集花瓣长度、花瓣宽度属性
y_test = iris_test[:, 4]
num_test = len(x_test)

x0_train = np.ones(num_train).reshape(-1, 1)
X_train = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)  # 属性值转化为32位浮点数
Y_train = tf.one_hot(tf.constant(y_train, dtype=tf.int32), 3)  # 把标签值转化成独热编码的形式（正交向量）

x0_test = np.ones(num_test).reshape(-1, 1)
X_test = tf.cast(tf.concat([x0_test, x_test], axis=1), tf.float32)
Y_test = tf.one_hot(tf.constant(y_test, dtype=tf.int32), 3)

# 设置超参数
learn_rate = 0.2
iter = 500
display_step = 100
# 设置模型参数初始值
np.random.seed(612)
# 权值矩阵W   偏置项B   W 3行3列2维张量，正态分布的随机值
# B是长度为3的一维张量，初始化为0 （一般情况）
W = tf.Variable(np.random.randn(3, 3), dtype=tf.float32)
B = tf.Variable(np.zeros([3]), dtype=tf.float32)

# 训练集和测试集的准确率和交叉熵损失
acc_train = []
cce_train = []
acc_test = []
cce_test = []

# 迭代学习模型参数
for i in range(0, iter + 1):
    with tf.GradientTape() as tape:  # 梯度方法
        PRED_train = tf.nn.softmax(tf.matmul(X_train, W)+B)  # 训练集数据在神经网络上的输出
        Loss_train = -tf.reduce_sum(Y_train * tf.math.log(PRED_train)) / num_train  # 训练集的交叉熵损失

    # 测试集的输出和交叉熵损失
    PRED_test = tf.nn.softmax(tf.matmul(X_test, W)+B)
    Loss_test = -tf.reduce_sum(Y_test * tf.math.log(PRED_test)) / num_test

    # 训练集和测试集的准确度
    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(), axis=1), y_train), tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(), axis=1), y_test), tf.float32))

    # 损失追加到相应的列表中  方便绘制曲线
    acc_train.append(accuracy_train)
    cce_train.append(Loss_train)
    acc_test.append(accuracy_test)
    cce_test.append(Loss_test)

    # 训练集数据取得W B的偏导数
    grads = tape.gradient(Loss_train, [W, B])
    # 对W的偏导数
    W.assign_sub(learn_rate * grads[0])
    # 对B的偏导数
    B.assign_sub(learn_rate * grads[1])

    if i % display_step == 0:
        # 分别更新模型W和B  显示准确率和损失
        print('i:%i,TrainAcc:%f,TrainLoss:%f,TestAcc:%f,TestLoss:%f' % (i, accuracy_train, Loss_train,accuracy_test,Loss_test))
tf.reduce_sum(PRED_train, axis=1)
tf.argmax(PRED_train.numpy(), axis=1)

M = 500
x1_min, x2_min = x_train.min(axis=0)
x1_max, x2_max = x_train.max(axis=0)
t1 = np.linspace(x1_min, x1_max, M)
t2 = np.linspace(x2_min, x2_max, M)
m1, m2 = np.meshgrid(t1, t2)


m0 = np.ones(M * M)
X_ = tf.cast(np.stack((m0, m1.reshape(-1), m2.reshape(-1)), axis=1), tf.float32)
Y_ = tf.nn.softmax(tf.matmul(X_, W)+B)
Y_ = tf.argmax(Y_.numpy(), axis=1)
n = tf.reshape(Y_, m1.shape)

# 设置中文字体
plt.rcParams['font.sans-serif']="SimHei"
plt.rcParams['axes.unicode_minus']="False"

# 绘制训练集散点图
plt.figure()
cm_bg = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt.scatter(x_train[:, 0], x_train[:, 1],  c=y_train, cmap='brg')
plt.xlabel('花瓣-length')
plt.ylabel('花瓣-width')
plt.title('Anderson鸢尾花训练数据集\n(Blue->Setosa | Red->Versicolor | Green- >Virginica )',fontsize=16)
plt.show()

# 绘制训练集损失变化曲线
plt.figure()
plt.plot(cce_train,color="blue",label="损失")
plt.xlabel('迭代次数',fontsize=12)
plt.ylabel('损失',fontsize=12)
plt.title('训练集损失函数曲线',fontsize=16)
plt.legend()
plt.show()

# 绘制训练集准确率变化曲线
plt.figure()
plt.plot(acc_train,color="red",label="准确率")
plt.xlabel('迭代次数',fontsize=12)
plt.ylabel('准确率',fontsize=12)
plt.title('训练集准确率曲线',fontsize=16)
plt.legend()
plt.show()

# 绘制测试集散点图
plt.figure()
cm_bg = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt.scatter(x_test[:, 0], x_test[:, 1],  c=y_test, cmap='brg')
plt.xlabel('花瓣-length')
plt.ylabel('花瓣-width')
plt.title('Anderson鸢尾花测试数据集\n(Blue->Setosa | Red->Versicolor | Green- >Virginica )',fontsize=16)
plt.show()

# 绘制测试集损失变化曲线
plt.figure()
plt.plot(cce_test,color="blue",label="损失")
plt.xlabel('迭代次数',fontsize=12)
plt.ylabel('损失',fontsize=12)
plt.title('测试集损失函数曲线',fontsize=16)
plt.legend()
plt.show()

# 绘制测试集准确率变化曲线
plt.figure()
plt.plot(acc_test,color="red",label="准确率")
plt.xlabel('迭代次数',fontsize=12)
plt.ylabel('准确率',fontsize=12)
plt.title('测试集准确率曲线',fontsize=16)
plt.legend()
plt.show()