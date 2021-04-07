import numpy as np
from Utils.hyperparams import Params

class NeuralNetwork:
    '''
    神经网络
    attributes:
        input: 输入
        output: 输出
        structure: 结构
        layers: 各层网络
    '''
    @staticmethod
    def create_layers(structure):
        num_layers = len(structure)-1
        layers = [None]*num_layers
        for i in range(num_layers):
            layers[i] = FullyConnectedLayer(structure[i], structure[i+1]
                                                , is_output=(i==num_layers-1))
        return layers

    def __init__(self, nn_structure):
        self.input = None
        self.output = None
        self.structure = nn_structure
        self.layers = NeuralNetwork.create_layers(self.structure)
    
    def forward(self, X):
        self.input = X
        for layer in self.layers:
            X = layer.forward(X)
        self.output = X
        return X
    
    def backward(self, yt):
        '''
        我们不在这里把yt变为gradient，我们要告诉各个激活函数他们是不是输出层让他们来计算
        input: yt, y_target是onehot后的标签
        '''
        gradient = yt
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def update(self):
        '''更新layers中的各个网络'''
        for layer in self.layers:
            layer.update()

class FullyConnectedLayer:
    '''
    全连接层，是NN.layers中的每个单位
    '''
    def __init__(self, input_size, output_size, is_output=False):
        '''
        inputs:
            input_size: 输入节点数量
            output_size: 输出节点数量
            is_output: 当前这层是不是输出层
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.w = Weights(input_size, output_size)
        self.b = Bias(output_size)
        # 根据超参数来定义激活函数
        if Params.use_softmax_xent and is_output:
            self.activator = Softmax(output_size, is_output=is_output)
        else:
            self.activator = Sigmoid(output_size, is_output=is_output)
    
    def forward(self, X):
        X = self.w.forward(X)
        X = self.b.forward(X)
        return self.activator.forward(X)
    
    def backward(self, grad):
        grad = self.activator.backward(grad)
        self.b.backward(grad)
        return self.w.backward(grad)
    
    def update(self):
        '''Weights, Bias需要update'''
        self.w.update()
        self.b.update()

class Differentiable:
    '''
    作为所有可求导对象的父类
    attrs:
        input_size
        output_size
        input
        output
        grad: 在backward的时候求的梯度
        vars: 计算时候的变量参数，是个矩阵（Weights）或向量（Bias）
    '''
    def __init__(self, *args):
        if len(args)==2:
            self.input_size, self.output_size = args
        else:
            self.input_size = args[0]
            self.output_size = args[0]
        self.input = None
        self.output = None
        self.grad = None
        self.vars = None
    
    def update(self):
        '''参数 = 参数-学习率*梯度'''
        self.vars -= Params.lr*self.grad

class Bias(Differentiable):
    def __init__(self, output_size):
        # 根据父类，初始化一些attrs
        super(Bias, self).__init__(output_size)
        # 初始化参数，长为output_size的向量
        self.vars = np.random.normal(0.,pow(output_size,-0.5),output_size)
    
    def forward(self, X):
        '''
        return 给到activator的输入
        '''
        self.input = X
        self.output = X+self.vars
        return self.output
    
    def backward(self, grad):
        '''
        input:
            grad: 从activator返回的梯度
        return b对上一层结果的梯度
        D(L->b) = D(L->{Xw+b})*D({Xw+b->b}) = D(L->{Xw+b})*1（链式法则）
        此处的grad恰为D(L->{Xw+b})
        由于b不含上一层结果，所以返回的梯度为0
        '''
        self.grad = grad
        return 0

class Weights(Differentiable):
    '''
    参考Bias的设计，完成Weights的设计
    Weights的设计和Bias的差别很大，需要处处小心
    '''
    def __init__(self, input_size, output_size):
        super(Weights, self).__init__(input_size, output_size)
        self.vars = np.random.normal(0.,pow(output_size,-0.5),(input_size,output_size))
    
    def forward(self, X):
        '''
        输入X为一个向量，参数vars是个input_size(X)*output_size(Xw)的矩阵
        '''
        self.input = X
        self.output = np.dot(X, self.vars)
        return self.output
    
    def backward(self, grad):
        '''
        参照forward，想想怎么定义grad和return
        注意！grad应该是D(L->w)而return的应该是D(L->X)
        非常建议自己手画一下forward过程中的矩阵运算
        '''
        self.grad = np.dot(self.input.reshape(self.input_size, -1),
                            grad.reshape(-1, self.output_size))
        return np.dot(grad, self.vars.T)

class Sigmoid(Differentiable):
    '''
    激活函数sigmoid
    '''
    def __init__(self, output_size, is_output=False):
        super(Sigmoid, self).__init__(output_size)
        self.is_output = is_output
    
    def forward(self, X):
        self.input = X
        self.output = 1/(1+np.exp(-X))
        return self.output
    
    def backward(self, grad):
        if self.is_output:
            grad = self.output - grad
        return grad*self.output*(1.-self.output)

class Softmax(Differentiable):
    '''
    Forward时Softmax, Backward时Softmax+CrossEntropy
    '''
    def __init__(self, output_size, is_output=False):
        super(Softmax, self).__init__(output_size)
        assert is_output
    
    def forward(self, X):
        self.input = X
        exp = np.exp(X)
        self.output = exp/np.sum(exp)
        return self.output
    
    def backward(self, yt):
        return self.output - yt
    
