# 功能：
# 保存模型需要设定的超参数
# 其成员包含所有可能使用的参数(包括子类)
# 不同模型只需要读取使用自己需要的参数
# 可以获取/更改参数(为方便就直接让用户取成员)
# 可以支持保存和读取
# 好处：
# 把参数和模型、数据分离，解耦合
# 可以把一组参数复用给多个模型
import numpy as np


class Setting():
    # UPDATE: 有任何可能用到的参数都可以往这里加
    # 输入一些预先定义的参数，主要传递命名参数
    # 未指定的参数先给出默认值，用户可直接获取并修改参数
    # @param layers: Layer对象的列表，表示网络
    # @param batch: mini-batch的大小
    # @param alpha: 学习率
    # @param epoch: 总训练周期数
    # @param initialize: w的初始化方法，为字符串
    def __init__(self, layers=[], batch=100, alpha=0.1, epoch=100, initialize='normal'):
        self.depth = len(layers)
        self.layers = layers
        self.batch = batch
        self.alpha = alpha
        self.epoch = epoch
        self.initialize = initialize
        # TODO:
        # 此处依据初始化方式进行初始化
        # 目前有 normal uniform xavier
        self.weight = {}
        if initialize == 'normal':
            for l in range(1, self.depth):
                self.weight[l] = 0.1 * np.random.randn(layers[l], layers[l-1])
        elif initialize == 'uniform':
            for l in range(1, self.depth):
                self.weight[l] = 0.1 * np.random.uniform(0, 1, (layers[l], layers[l-1]))
        elif initialize == 'xavier':
            for l in range(1, self.depth):
                bound = np.sqrt(6./(layers[l] + layers[l-1]))
                self.weight[l] = np.random.uniform(-bound, bound, (layers[l], layers[l-1]))
        else:  # 输入错误
            pass

    # 把自身设置保存到文件(文件结构方便读取和保存就行)
    # @param path: 保存文件的路径名
    def saveSetting(self, path):
        np.savez(path, layers=self.layers, batch=self.batch, alpha=self.alpha, epoch=self.epoch, initialize=self.initialize)
        print("param saved to {}.npz".format(path))

    # 先清空原设置，再从已有文件读取设置，直接更改自身成员
    # @param path: 读取文件的路径名
    def loadSetting(self, path):
        t = np.load(path)
        self.layers = t['layers']
        self.depth = len(self.layers)
        self.batch = t['batch']
        self.alpha = t['alpha']
        self.epoch = t['epoch']
        self.initialize = t['initialize']
        initialize = t['initialize']

        if initialize == 'normal':
            for l in range(1, self.depth):
                self.weight[l] = 0.1 * np.random.randn(self.layers[l], self.layers[l-1])
        elif initialize == 'uniform':
            for l in range(1, self.depth):
                self.weight[l] = 0.1 * np.random.uniform(0, 1, (self.layers[l], self.layers[l-1]))
        elif initialize == 'xavier':
            for l in range(1, self.depth):
                bound = np.sqrt(6./(self.layers[l] + self.layers[l-1]))
                self.weight[l] = np.random.uniform(-bound, bound, (self.layers[l], self.layers[l-1]))
        else:  # 输入错误
            pass

    # DEBUG:
    # 输出所有成员
    def ParamShow(self):
        print('layers =', self.layers)
        print('depth =', self.depth)
        print('batch =', self.batch)
        print('alpha =', self.alpha)
        print('epoch =', self.epoch)
        print('initialize =', self.initialize)


if __name__ == '__main__':
    # 给定初始化参数
    layers = [100, 100, 10]
    s1 = Setting(layers, 50, 0.1, 200, 'uniform')
    s1.ParamShow()
    s1.saveSetting('parmlist')
    print("-"*30)
    # 默认参数
    s2 = Setting()
    s2.ParamShow()
    print("-"*30)
    # 读取文件
    s3 = Setting()
    s3.loadSetting('parmlist.npz')
    s3.ParamShow()
