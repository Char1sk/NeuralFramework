# 功能：
# 保存模型需要设定的超参数
# 其成员包含所有可能使用的参数(包括子类)
# 不同模型只需要读取使用自己需要的参数
# 可以获取/更改参数(为方便就直接让用户取成员)
# 可以支持保存和读取
# 好处：
# 把参数和模型、数据分离，解耦合
# 可以把一组参数复用给多个模型


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
        # TODO:
        # 此处依据初始化方式进行初始化
        # 目前有 normal uniform xavier
        self.weight = []

    # TODO:
    # 把自身设置保存到文件(文件结构方便读取和保存就行)
    # @param path: 保存文件的路径名
    def saveSetting(self, path):
        pass

    # TODO:
    # 先清空原设置，再从已有文件读取设置，直接更改自身成员
    # @param path: 读取文件的路径名
    def loadSetting(self, path):
        pass
