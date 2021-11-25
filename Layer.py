# 功能：
# 神经网络中的一层神经元
# 包含神经元个数、激活函数
# 以及中间计算结果
# 好处：
# 集成数据，而非散乱分布在Model中
# 编写和使用时更加有结构和逻辑


class Layer():
    # 输入一些预先定义的参数
    # a z delta 等稍后计算获得
    # @param count: 一层神经元的个数
    # @param activation: 该层神经元的激活函数
    def __init__(self, count, activation, dactivation):
        self.count = count
        self.activation = activation
        self.dactivation = dactivation
        self.z = None
        self.a = None
        self.delta = None
