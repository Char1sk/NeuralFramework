# 功能：
# 保存训练测试使用的数据集
# 提供基本的数据集预处理功能
# 可以获取/更改数据集(成员)
# 可以支持保存和读取(以免以后再次修改)
# 好处：
# 把数据和模型、参数分离，解耦合
# 可以把一组数据复用给多个模型


class Dataset():
    # DISCUSS: 传入是把Data和Label放一起还是分开？
    # 用户指定完整数据集，或者训练和测试集
    # 第i个数据应当对应第i个标签(用户自行遵守该规则)
    # @param allSet: 完整数据集，下标0为输入，下标1为标签
    # @param trainSet: 训练集，下标0为输入，下标1为标签
    # @param validateSet: 验证集，下标0为输入，下标1为标签
    # @param testSet: 测试集，下标0为输入，下标1为标签
    def __init__(self, allSet=[None, None], trainSet=[None, None],
                 validateSet=[None, None], testSet=[None, None]):
        (self.allData, self.allLabel) = allSet
        (self.trainData, self.trainLabel) = trainSet
        (self.validateData, self.validateLabel) = validateSet
        (self.testData, self.testLabel) = testSet

    # TODO:
    # 用户自己调用，把完整数据集随机分为训练集和测试集
    # trainRatio 给训练集
    # validateRatio 给验证集
    # 1 - trainRatio - validateRatio 给 test
    # 若这两者和小于等于1，则抛出异常
    # @param trainRatio: all划分给train的比例
    # @param validateRatio: all划分给validate的比例
    def divideData(self, trainRatio=0.6, validateRatio=0.2):
        pass

    # TODO: (包含 all train validate test 的 getter)
    # 把数据集的输入和标签放在列表输出
    # @return allSet: 下标0为输入，下标1为标签
    def getAllSet(self):
        pass

    # TODO: (包含 all train validate test 的 setter)
    # 把数据集的输入和标签放在列表进行赋值
    # @param allSet: 下标0为输入，下标1为标签
    def setAllSet(self, allSet):
        pass

    # TODO:
    # 把自身数据集保存到文件(文件结构方便读取和保存就行)
    # @param path: 保存文件的路径名
    def saveDataset(self, path):
        pass

    # TODO:
    # 从已有文件读取数据，直接更改自身成员
    # @param path: 读取文件的路径名
    def loadDataset(self, path):
        pass

    # DEBUG:
    # 输出所有成员
    def showall(self):
        print(self.allData)
        print(self.allLabel)
        print(self.trainData)
        print(self.trainLabel)
        print(self.validateData)
        print(self.validateLabel)
        print(self.testData)
        print(self.testLabel)
