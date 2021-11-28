import numpy as np
from Dataset import Dataset
from Setting import Setting
#import Utility as ut
import UtilityJit as ut 

class Model():
    # UPDATE: 有新的参数等直接新增即可
    # DISCUSS 是直接保存dataset/setting对象，还是拷贝其成员？
    #         感觉可以有选择地拷贝成员，只取所需要的即可
    #         如只取dataset的train和test，只取setting的layer
    # @param dataset: Dataset对象，内含训练和测试集
    # @param setting: Setting对象，内含对基类的参数设置
    def __init__(self, dataset, setting):
        # 设置数据集
        self.trainData = dataset.trainData
        self.trainLabel = dataset.trainLabel
        self.validateData = dataset.validateData
        self.validateLabel = dataset.validateLabel
        self.testData = dataset.testData
        self.testLabel = dataset.testLabel
        # 设置模型参数
        self.depth = setting.depth
        self.layers = setting.layers
        for l in self.layers:
            (l.activation, l.dactivation) = ut.strToFunc(l.activation)
        self.batch = setting.batch
        self.alpha = setting.alpha
        self.epoch = setting.epoch
        self.initialize = setting.initialize
        self.weight = {}
        if self.initialize == 'normal':
            for l in range(1, self.depth):
                self.weight[l] = 0.1 * np.random.randn(self.layers[l].count, self.layers[l-1].count)
        elif self.initialize == 'uniform':
            for l in range(1, self.depth):
                self.weight[l] = 0.1 * np.random.uniform(0, 1, (self.layers[l].count, self.layers[l-1].count))
        elif self.initialize == 'xavier':
            for l in range(1, self.depth):
                bound = np.sqrt(6./(self.layers[l] + self.layers[l-1]))
                self.weight[l] = np.random.uniform(-bound, bound, (self.layers[l].count, self.layers[l-1].count))
        elif self.initialize == 'zero':
            for l in range(1, self.depth):
                self.weight[l] = 0.1 * np.zeros((self.layers[l].count, self.layers[l-1].count))
        else:  # 输入错误
            pass

        self.bias = 0  # 暂用!!!!!!!!!!
        (self.costFunction, self.dCostFunction) = ut.strToFunc(setting.costFunction)
        # 混淆矩阵，计算指标时的中间结果
        self.confusionMatrix = None
        # 训练过程中，训练集和验证集的预测结果
        # 在训练时填充列表，记录每一次的结果
        # 在需要获取过程中指标时，再利用这些计算
        self.trainOutputs = []
        self.validateOutputs = []
        # 训练结束后，进行最后检验得到的输出
        # 在训练完后进行赋值，记录结果
        # 在需要获取相应指标时，再利用这些计算
        self.trainResult = None
        self.validateResult = None
        self.testResult = None

    # def strToFunc(self, funcname):
    #     if funcname == 'sigmoid':
    #         return (ut.sigmoid, ut.dSigmoid)
    #     if funcname == 'meanSquareError':
    #         return (ut.meanSquareError, ut.dMeanSquareError)
    #     if funcname == 'crossEntropy':
    #         return (ut.softmaxCrossEntropy, ut.dSoftmaxCrossEntropy)

    # UPDATE: 以后写训练的时候再来改，函数只是表明有这个步骤
    # 进行迭代训练过程，更新权重，记录中间结果
    def train(self):
        for iter in range(self.epoch):
            # UPDATE: 前向计算，计算 z a
            # UPDATE: 反向传播，计算 delta
            # 记录该次训练集和验证集的预测
            self.trainOutputs.append(self.getOutput(self.trainData, self.trainLabel))
            self.validateOutputs.append(self.getOutput(self.validateData, self.validateLabel))
            # UPDATE: 更新权值

    # 进行训练完成后的测试，保存测试结果
    def model(self):
        # 记录训练集、验证集、测试集的预测结果
        self.trainResult = self.getOutput(self.trainData, self.trainLabel)
        self.validateResult = self.getOutput(self.validateData, self.validateLabel)
        self.testResult = self.getOutput(self.testData, self.testLabel)

    # 用于计算给出预测结果和实际标签的准确率
    # @param output: 预测输出
    # @param label: 实际标签
    # @return accuracy: 计算所得的准确率
    # 若参数给的是一次的输出(如trainResult, trainLabel)
    #       则返回一个准确率，即只对该次进行计算
    # 若参数给的是多次输出的列表(如trainOutputs, trainLabel)
    #       则返回准确率列表，即每次的准确率的列表
    def calculateAccuracy(self, output, label):
        if (type(output) != list):  # 判断是否为多个的outputs或单个outputs
            if self.confusionMatrix is None:
                # 构造混淆矩阵
                x_i = output.shape[0]
                y_i = output.shape[1]
                mid = np.zeros((x_i, x_i))
                idx_output = np.argmax(output, axis=0)
                idx_label = np.argmax(label, axis=0)
                for i in range(y_i):
                    mid[idx_output[i]][idx_label[i]] += 1
            # 求Accuracy
            son = 0
            for i in range(mid.shape[0]):
                son += mid[i][i]
            acc = son / output.shape[1]
            return acc
        else:  # 列表情况下求Accuracy，并返回相应列表
            accList = []
            for i in range(len(output)):
                accList.append(self.calculateAccuracy(output[i], label))
            return accList

    # 用于计算给出预测结果和实际标签的准确率
    # @param output: 预测输出
    # @param label: 实际标签
    # @return recall: 计算所得的召回率
    # 若参数给的是一次的输出(如trainResult, trainLabel)
    #       则返回一个准确率，即只对该次进行计算
    # 若参数给的是多次输出的列表(如trainOutputs, trainLabel)
    #       则返回准确率列表，即每次的准确率的列表
    def calculateRecall(self, output, label):
        if (type(output) != list):
            if self.confusionMatrix is None:
                # 构造混淆矩阵
                x_i = output.shape[0]
                y_i = output.shape[1]
                mid = np.zeros((x_i, x_i))
                idx_output = np.argmax(output, axis=0)
                idx_label = np.argmax(label, axis=0)
                for i in range(y_i):
                    mid[idx_output[i]][idx_label[i]] += 1
            # 求Recall
            Recall = np.zeros(x_i)
            for i in range(x_i):
                Recall[i] = mid[i][i] / np.sum(mid, axis=1)[i]
            return Recall
        else:  # 列表情况下求Recall，并返回相应列表
            RecallList = []
            for i in range(len(output)):
                RecallList.append(self.calculateRecall(output[i], label))
            return RecallList

    # 用于计算给出预测结果和实际标签的准确率
    # @param output: 预测输出
    # @param label: 实际标签
    # @return precision: 计算所得的精确率
    # 若参数给的是一次的输出(如trainResult, trainLabel)
    #       则返回一个准确率，即只对该次进行计算
    # 若参数给的是多次输出的列表(如trainOutputs, trainLabel)
    #       则返回准确率列表，即每次的准确率的列表
    def calculatePrecision(self, output, label):
        if (type(output) != list):  # 判断是否为多个的outputs或单个outputs
            if self.confusionMatrix is None:
                # 构造混淆矩阵
                x_i = output.shape[0]
                y_i = output.shape[1]
                mid = np.zeros((x_i, x_i))
                idx_output = np.argmax(output, axis=0)
                idx_label = np.argmax(label, axis=0)
                for i in range(y_i):
                    mid[idx_output[i]][idx_label[i]] += 1
            # 求Precision
            Precision = np.zeros(x_i)
            for i in range(x_i):
                Precision[i] = mid[i][i] / np.sum(mid, axis=0)[i]
            return Precision
        else:  # 列表情况下求 Precision，并返回相应列表
            PrecisionList = []
            for i in range(len(output)):
                PrecisionList.append(self.calculatePrecision(output[i], label))
            return PrecisionList

    # 用于计算给出预测结果和实际标签的准确率
    # @param output: 预测输出
    # @param label: 实际标签
    # @return f1score: 计算所得的f1
    # 若参数给的是一次的输出(如trainResult, trainLabel)
    #       则返回一个准确率，即只对该次进行计算
    # 若参数给的是多次输出的列表(如trainOutputs, trainLabel)
    #       则返回准确率列表，即每次的准确率的列表
    def calculateF1Score(self, output, label):
        if (type(output) != list):
            x_i = output.shape[0]
            Recall = self.calculateRecall(output, label)
            Precision = self.calculatePrecision(output, label)
            F1Score = np.zeros(x_i)
            for i in range(x_i):
                F1Score[i] = 2 * Recall[i] * Precision[i] / (Recall[i] + Precision[i])
            return F1Score
        else:  # 列表情况下求F1Score，并返回相应列表
            F1ScoreList = []
            for i in range(len(output)):
                F1ScoreList.append(self.calculateF1Score(output[i], label))
            return F1ScoreList

    # UPDATE: 针对给定的小集合，计算预测结果
    # 用于计算给定数据情况下，网络的预测输出
    # 可复用于：中间验证、结尾测试、用户使用
    # 不同模型的计算方式不同，可留给子类实现掉
    # @param data: 输入数据
    # @param label: 目标标签
    # @return output: 预测的结果(网络输出)
    def getOutput(self, data):
        self.layers[0].a = data
        for l in range(0, self.depth - 1):
            self.layers[l + 1].z = np.dot(self.weight[l + 1], self.layers[l].a)
            self.layers[l + 1].a = self.layers[l].activation(self.layers[l + 1].z)
        return self.layers[self.depth - 1].a


if __name__ == '__main__':
    # 定义数据集，数据过少不进行分划
    data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    label = np.array([[1, 1, 1, 0]])
    data = Dataset(allSet=[data.T, label])
    # 只需额外定义 layers
    s = Setting([2, 1])
    # 定义模型
    model = Model(data, s)

    # 测试对 output 的计算
    output = np.random.rand(2, 10)
    label = np.random.rand(2, 10)
    np.set_printoptions(precision=3, suppress=True)
    print("Accuracy:")
    print("{}".format(model.calculateAccuracy(output, label)))
    print("Precision:")
    print("{}".format(model.calculatePrecision(output, label)))
    print("Recall:")
    print("{}".format(model.calculateRecall(output, label)))
    print("F1Score:")
    print("{}".format(model.calculateF1Score(output, label)))
    print("-" * 20)

    # 测试对 outputs 的计算
    outputs = []
    outputs.append(output)
    outputs.append(output)
    print("Accuracy:")
    print("{}".format(model.calculateAccuracy(outputs, label)))
    print("Precision:")
    print("{}".format(model.calculatePrecision(outputs, label)))
    print("Recall:")
    print("{}".format(model.calculateRecall(outputs, label)))
    print("F1Score:")
    print("{}".format(model.calculateF1Score(outputs, label)))
