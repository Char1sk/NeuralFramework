# 功能：
# 保存模型需要设定的超参数
# 其成员包含所有可能使用的参数(包括子类)
# 不同模型只需要读取使用自己需要的参数
# 可以获取/更改参数(为方便就直接让用户取成员)
# 可以支持保存和读取
# 好处：
# 把参数和模型、数据分离，解耦合
# 可以把一组参数复用给多个模型
from Layer import Layer
import json


class Setting():
    # UPDATE: 有任何可能用到的参数都可以往这里加
    # 输入一些预先定义的参数，主要传递命名参数
    # 未指定的参数先给出默认值，用户可直接获取并修改参数
    # @param layers: Layer对象的列表，表示网络
    # @param batch: mini-batch的大小
    # @param alpha: 学习率
    # @param epoch: 总训练周期数
    # @param initialize: w的初始化方法，为字符串
    def __init__(self, layers=[], batch=100, alpha=0.1, epoch=100, initialize='normal', costFunction='meanSquareError'):
        self.depth = len(layers)
        self.layers = layers
        self.batch = batch
        self.alpha = alpha
        self.epoch = epoch
        self.initialize = initialize
        self.costFunction = costFunction

    # 把自身设置保存到文件(文件结构方便读取和保存就行)
    # @param path: 保存文件的路径名
    def saveSetting(self, path):
        # np.savez(path, layers=self.layers, batch=self.batch, alpha=self.alpha, epoch=self.epoch, initialize=self.initialize)
        dictSetting = self.__dict__.copy()
        dictSetting['layers'] = []
        for l in self.layers:
            ld = {
                'count': l.count,
                'activation': l.activation
            }
            dictSetting['layers'].append(ld)
        with open(path, 'w') as f:
            json.dump(dictSetting, f)
            print("parameters saved to {}".format(path))

    # 先清空原设置，再从已有文件读取设置，直接更改自身成员
    # @param path: 读取文件的路径名
    def loadSetting(self, path):
        with open(path, 'r') as f:
            dictSetting = json.load(f)
            print("parameters loaded from {}".format(path))

        self.layers = []
        for dl in dictSetting['layers']:
            self.layers.append(Layer(dl['count'], dl['activation']))
        self.depth = len(self.layers)
        self.batch = dictSetting['batch']
        self.alpha = dictSetting['alpha']
        self.epoch = dictSetting['epoch']
        self.initialize = dictSetting['initialize']
        self.costFunction = dictSetting['costFunction']

    # DEBUG:
    # 输出所有成员
    def show(self):
        print('layers =', self.layers)
        print('depth =', self.depth)
        print('batch =', self.batch)
        print('alpha =', self.alpha)
        print('epoch =', self.epoch)
        print('initialize =', self.initialize)
        print('costFunction =', self.costFunction)


if __name__ == '__main__':
    layer1 = Layer(4, 'sigmoid')
    layer2 = Layer(5, 'sigmoid')
    layer3 = Layer(6, 'linear')
    layers = [layer1, layer2, layer3]
    setting = Setting(layers=layers, batch=5, alpha=1, epoch=50, initialize='xavier', costFunction='crossEntropy')
    setting.saveSetting('./testSetting.json')
    newSetting = Setting()
    newSetting.loadSetting('./testSetting.json')
    newSetting.show()
