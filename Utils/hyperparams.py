class Params:
    '''
    此类用作超参数
    '''
    rand_seed = 1                       #随机数seed
    nlabels = 10                        #类别数量
    nn_structure = [28*28, 50, 10]      #网络结构
    epoch = 5                           #训练epoch数量
    lr = 0.1                            #学习率
    use_softmax_xent = False            #是否在最后一层添加Softmax