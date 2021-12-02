## Anderson鸢尾花数据集

#### 简介

鸢尾花数据集最初由 Edgar Anderson 测量得到，而后在著名的统计学家和生物学家 R.A Fisher 于1936年发表的文章「The use of multiple measurements in taxonomic problems」中被使用，用其作为线性判别分析（Linear Discriminant Analysis）的一个例子，证明分类的统计方法，从此而被众人所知，尤其是在机器学习这个领域。 

数据中的两类鸢尾花记录结果是在加拿大加斯帕半岛上，于同一天的同一个时间段，使用相同的测量仪器，在相同的牧场上由同一个人测量出来的。这是一份有着70年历史的数据，虽然老，但是却很经典。

鸢尾花数据集共收集了三类鸢尾花，分别是山鸢尾(Setosa) 、 变色鸢尾(Versicolor) 和维吉尼亚鸢尾(Virginica) 。

每一类鸢尾花收集了50条样本记录，共计150条。

##### Features

数据集包括4个属性，分别为：

花萼长度（Sepal Length）

花萼宽度（Sepal Width）

花瓣长度（Petal Length）

花瓣宽度（Petal Width）

##### Lables

数据集包括1个标签，int 类型, 对三种分类做编号。

0 表示 Setosa 

1 表示 Versicolor 

2 表示 Virginica

|     列名     |                             说明                             | 类型  |
| :----------: | :----------------------------------------------------------: | :---: |
| Sepal Length |                           花萼长度                           | float |
| Sepal Width  |                           花萼宽度                           | float |
| Petal Length |                           花瓣长度                           | float |
| Petal Width  |                           花瓣宽度                           | float |
|    Class     | 类别变量。0 表示山鸢尾，1 表示变色鸢尾，2 表示维吉尼亚鸢尾。 |  int  |

#### 数据可视化

