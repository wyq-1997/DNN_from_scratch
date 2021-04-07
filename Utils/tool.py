import numpy as np
def make_one_hot(labels, nlabels):
    '''标签转变为onehot向量'''
    I = np.eye(nlabels)
    return I[labels]

def serialize(X):
    '''2d图片变为1d输入'''
    # -1表述让系统计算这个维度的长度
    return X.reshape(X.shape[0], -1)

def normalize(X):
    '''规范化'''
    upper = np.max(X)
    lower = np.min(X)
    half_range = (upper-lower)/2
    return (X-lower-half_range)/half_range