'''
README:

本代码主要用于快速上手神经网络
主要基于numpy实现一个手写的神经网络

请遵循以下方式进行学习：
1. 阅读main.py以及注释，以及Utils里的两个文件
2. 阅读model.py和各个类/文档的docstring
3. 自己完成model.py的各个类和方法
'''

import keras #base环境里没有这个库，所以建议从命令行运行`python main.py`
import numpy as np
from tqdm import tqdm

from model import NeuralNetwork
from Utils.hyperparams import *
from Utils.tool import *

# 设置random seed，意思就是伪随机，每次程序运行的random出来结果都一样
np.random.seed(Params.rand_seed)

# 读取数据
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()

'''
type: numpy.ndarray
train_X.shape: (60000, 28, 28)
train_y.shape: (60000,)
test_X.shape: (10000, 28, 28)
test_y.shape: (10000,)
'''

# 创建网络并且确认图片大小和输入吻合
nn = NeuralNetwork(Params.nn_structure)
assert Params.nn_structure[0]==train_X.shape[1]*train_X.shape[2]

# 读取数据集长度
train_len = train_X.shape[0]
test_len = test_X.shape[0]

# 把标签onehot化
train_y_ori, test_y_ori = train_y, test_y
train_y, test_y = \
    make_one_hot(train_y, Params.nlabels), make_one_hot(test_y, Params.nlabels)

# 对输入降维
train_X = serialize(train_X)
test_X = serialize(test_X)
train_X = normalize(train_X)
test_X = normalize(test_X)

'''
train_y.shape:          (60000, 10)
train_y_ori.shape:      (60000,)
test_y.shape:           (10000, 10)
test_y_ori.shape:       (10000,)
'''

# train
for _ in range(Params.epoch):
    # tqdm让代码有种读取进度条的显示效果
    for i in tqdm(range(train_len)):
        X = train_X[i]
        y = train_y[i]
        nn.forward(X)   # 正向传播 计算结果
        nn.backward(y)  # 反向传播 计算梯度
        nn.update()     # 根据梯度 更新参数

correct = 0
for i in tqdm(range(test_len)):
    X = test_X[i]
    y = test_y_ori[i]
    pred = nn.forward(X)
    y_pred = np.argmax(pred)
    if y_pred==y:
        # print(y, y_pred, pred)
        correct+=1
    count = i+1
print(f"{correct}/{count}, accuracy:{round(correct/count, 2)}")
