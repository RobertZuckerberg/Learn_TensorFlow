import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

for i in range(16):
    num = np.random.randint(1,10000)
    plt.subplot(4, 4, i+1)
    plt.axis("off")
    plt.imshow(test_x[num], cmap='gray')
    plt.title(test_y[num], fontsize=14)

plt.tight_layout()
plt.suptitle("MNIST测试集样本", x=0.5, y=1.02, color='red', fontsize=20)
plt.show()
