import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 使用全部506条数据，实现波士顿房价数据集可视化
boston_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(_,_) = boston_housing.load_data(test_split=0)  # 提取出全部数据作为训练集

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
          "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LSTAT", "MEDV"]

plt.figure(figsize=(12,12))

for i in range(13):
    plt.subplot(4,4,(i+1))
    plt.scatter(train_x[:,i], train_y)
    plt.xlabel(titles[i])
    plt.ylabel( "Price($1000's)")
    plt.title(str(i+1)+ "." +titles[i]+" - Price" )

plt.tight_layout()
plt.suptitle("各个属性与房价的关系",x=0.5, y=1.02, fontsize=20)
plt.show()

# 要求用户选择属性,根据用户的选择，输出对应属性的散点图
print("1 -- CRIM")
print("2 -- ZN")
print("3 -- INDUS")
print("4 -- CHAS")
print("5 -- NOX")
print("6 -- RM")
print("7 -- AGE")
print("8 -- DIS")
print("9 -- RAD")
print("10 -- TAX")
print("11 -- PTRATIO")
print("12 -- B-1000")
print("13 -- LSTAT")
num=int(input("请输入一个整数："))
plt.figure(figsize=(3,3))
plt.scatter(train_x[:,num-1], train_y)
plt.xlabel(titles[num-1])
plt.ylabel( "Price($1000's)")
plt.title(str(num)+ "." +titles[num-1]+" - Price" )
