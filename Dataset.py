# 功能：
# 保存训练测试使用的数据集
# 提供基本的数据集预处理功能
# 可以获取/更改数据集(成员)
# 可以支持保存和读取(以免以后再次修改)
# 好处：
# 把数据和模型、参数分离，解耦合
# 可以把一组数据复用给多个模型
import numpy as np
import pickle


class Dataset():
    # 用户指定完整数据集，或者训练和测试集
    # 第i个数据应当对应第i个标签(用户自行遵守该规则)
    # @param allSet: 完整数据集，下标0为输入，下标1为标签
    # @param trainSet: 训练集，下标0为输入，下标1为标签
    # @param validateSet: 验证集，下标0为输入，下标1为标签
    # @param testSet: 测试集，下标0为输入，下标1为标签
    # 该dataset类的数据（无论标签还是数据本身）只能接收二维numpy数组（即使是一维也要写成2d array形式）
    def __init__(self, allSet=[None, None], trainSet=[None, None],
                 validateSet=[None, None], testSet=[None, None]):
        # 异常检测
        if allSet[0] is not None and allSet[1] is not None:
            try:
                if allSet[0].shape[1] != allSet[1].shape[1]:
                    raise ValueError("数据与标签不匹配！")
            except ValueError as e:
                print("引发异常：", repr(e))
                raise
        (self.allData, self.allLabel) = allSet
        (self.trainData, self.trainLabel) = trainSet
        (self.validateData, self.validateLabel) = validateSet
        (self.testData, self.testLabel) = testSet

    # 用户自己调用，把完整数据集随机分为训练集和测试集
    # trainRatio 给训练集
    # validateRatio 给验证集
    # 1 - trainRatio - validateRatio 给 test
    # 若这两者和大于等于1，则抛出异常
    # @param trainRatio: all划分给train的比例
    # @param validateRatio: all划分给validate的比例
    def divideData(self, trainRatio=0.6, validateRatio=0.2):
        # Error Check
        try:
            testRatio = 1-trainRatio-validateRatio
            if testRatio < 0:
                raise ValueError("错误的划分比例！")
        except ValueError as e:
            print("引发异常：", repr(e))
            raise
        # determine the set's size
        data_size = self.allLabel.shape[1]
        train_size = int(data_size*trainRatio)
        val_size = int(data_size*validateRatio)
        idxs = np.random.permutation(data_size)     # shuffle serial numbers
        train_idx = idxs[:train_size]
        val_idx = idxs[train_size:train_size+val_size]
        test_idx = idxs[train_size+val_size:]

        # split the allset(both data and label)
        self.trainData = self.allData[:, train_idx]
        self.trainLabel = self.allLabel[:, train_idx]
        self.validateData = self.allData[:, val_idx]
        self.validateLabel = self.allLabel[:, val_idx]
        self.testData = self.allData[:, test_idx]
        self.testLabel = self.allLabel[:, test_idx]

    # 把数据集的输入和标签放在列表输出
    # @return allSet: 下标0为输入，下标1为标签
    def getAllSet(self):
        # use 2d list to return the dataset in class Dataset
        return [[self.allData, self.allLabel],
                [self.trainData, self.trainLabel],
                [self.validateData, self.validateLabel],
                [self.testData, self.testLabel]]

    # 把数据集的输入和标签放在列表进行赋值
    # @param allSet: 下标0为输入，下标1为标签
    def setAllSet(self, allSet):
        # allSet should be a 2d list.
        # the first dimension represent the different set of data,
        # and the second dimension distinguish the data and the label
        self.allData, self.allLabel = allSet[0]
        self.trainData, self.trainLabel = allSet[1]
        self.validateData, self.validateLabel = allSet[2]
        self.testData, self.testLabel = allSet[3]

    # 把自身数据集保存到文件(文件结构方便读取和保存就行)
    # @param path: 保存文件的路径名
    def saveDataset(self, path):
        # use pickle to save dataset, split data only
        with open(path, 'wb') as f:
            pickle.dump([self.trainData, self.trainLabel, self.validateData,
                         self.validateLabel, self.testData, self.testLabel], f)
        print("test data saved to {}".format(path))

    # 从已有文件读取数据，直接更改自身成员
    # @param path: 读取文件的路径名
    def loadDataset(self, path):
        # use pickle to load dataset, split data only
        f = open(path, 'rb')
        (self.trainData, self.trainLabel, self.validateData,
         self.validateLabel, self.testData, self.testLabel)\
            = pickle.load(f)

    # DEBUG:
    # 输出所有成员
    def showall(self):
        print("all set data & label")
        print(self.allData)
        print(self.allLabel)
        print("train set data & label")
        print(self.trainData)
        print(self.trainLabel)
        print("val set data & label")
        print(self.validateData)
        print(self.validateLabel)
        print("test set data & label")
        print(self.testData)
        print(self.testLabel)


if __name__ == '__main__':
    # test dataset class
    alldata = np.zeros((10, 10))
    for i in range(10):
        alldata[i, i] = 1
    alllabel = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    print("*** MODULE CHECK ***")
    print("-"*30)
    print("----- data divide check ------")
    data_process = Dataset(allSet=[alldata, alllabel])
    data_process.divideData()
    data_process.showall()
    print("------- data get check -------")
    [[all_d, all_l], [train_d, train_l], [val_d, val_l], [test_d, test_l]] = data_process.getAllSet()
    print("all set data & label")
    print(all_d, all_l)
    print("train set data & label")
    print(train_d, train_l)
    print("val set data & label")
    print(val_d, val_l)
    print("test set data & label")
    print(test_d, test_l)
    print("------ data save check -------")
    data_process.saveDataset('data.pkl')
    print("------ data load check -------")
    data_process2 = Dataset()
    data_process2.loadDataset('data.pkl')
    data_process2.showall()
    print("------- data set check -------")
    data_process3 = Dataset()
    data_process3.setAllSet([[alldata, alllabel], [None, None], [None, None], [None, None]])
    data_process3.showall()
