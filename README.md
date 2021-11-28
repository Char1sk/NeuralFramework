# 神经网络框架设计

## 文件结构

- 一个文件里面放一个类(同名)
- Model: 模型基类，包含统一的一些操作
- xxxxx: 派生类，是实际的模型，实现了具体操作(现在不写)
- Setting: 参数设置类，代表模型的配置信息，提供信息保存和修改
- Dataset: 数据集类，提供数据保存，以及基本的预处理功能
- Layer：神经网络中的一层，主要是信息的整合，结构化

## 函数注释说明

- TODO 表明该函数需要完成
- UPDATE 表明将来要做这些工作，现在不用
- DISCUSS 表明此处值得讨论
- DEBUG 表明该函数仅用于测试，不涉及功能
- @param xxx: 说明形参xxx的描述
- @return xxx: 说明返回值xxx的描述
- 单行没有上述标签的，即为该函数的描述

## 需求描述

- Lesson3 模型定义
  - 层数/结构(layers表明层数和结构)
  - 神经元个数(Layer对象的成员)
  - 激活函数(Layer对象的成员)
- Lesson4 性能验证
  - 多处验证(中间训练集和测试集，结束后三个集合)
  - 历史验证(保存训练过程中和过程后的预测结果)
  - 多种性能指标(利用保存的结果计算，减小负担，有选择地计算指标)

## 第一次代码任务列表

- Model.py
  - 实现计算Acc, Recall, Precision, F1的函数
- Setting.py
  - 构造函数中，实现权重的初始化
  - 实现设置的保存save和加载load
- Dataset.py
  - 实现数据集的划分功能，随机划分
  - 【可选】实现数据集的getter, setter
  - 实现数据集的保存save和加载load
- 以上各自部分的单元测试
- Test.py
  - 【可选】一个整体测试

## 第二次代码任务列表

- 优化方法
  - 通过在新类创建新类并覆盖train实现
  - 二层前馈，Hebb*
  - 多层前馈，BP*
  - 可能有：SGD/Momentum/Ada/Adam/退火遗传/等等其他
  - 星号必须实现，其他可以不局限于上面自选
  - 不同优化算法给出对应单元测试（主函数中）
  - 需要用到数学函数时，联系Utility同学或自行加入即可
  - 不同模型需要用到新参数时，在Setting和Model里添加即可
  - 实现的时候尽量贴合我们的框架，用到前面的内容
- 关于模型参数的举例
  - 现有基类Model，以及子类A, B, C
  - A使用参数a, b
  - B使用参数a, c
  - C使用参数a, b, c
  - 则Setting和Model里面包含所有参数a, b, c
  - 出于使用A类进行Setting配置时，只需设置a, b; 而c用默认值
  - 使用子类A时，只访问a, b即可，不会因为c的值产生影响
  - 按照此规则可向Setting和Model添加参数，而不影响其他模型
- Perceptron.py（已实现）
  - 此处只是整理成文件，用于其他优化方法参考
  - 多层感知机，用感知机学习方法优化
- HebbModel.py
  - 网络结构：两层前馈神经网络
  - 优化算法：Hebb规则
- ForwardNetwork.py
  - 网络结构：多层前馈神经网络
  - 优化算法：BP算法
- 其他文件
  - 网络结构：看着写
  - 优化算法：自选
- Utility.py
  - 预定义的数学函数，如激活函数、性能函数等；激活函数也要实现其导数
  - 预定义的枚举类型，用于选项控制（目前还不用写）

## 第三次代码任务列表

- 以下大测试(单开文件)不必执着于调参和性能，仅供测试和对比使用即可
- 关于Setting的json配置，可参考testSetting.json，实际并无太大差异
- 大测试1
  - 【读取文件Setting，初始化各模型并调用train，展现结果即可】
  - 横向比较各模型性能，模型为Forward SGD Momentum Adam
  - 数据集使用MNIST数据集，完成手写体数字识别任务
  - 参数写在json配置文件中，不同模型使用同一个参数配置
  - 提供训练计时，输出训练集和测试集的各项指标
  - 如有必要，可进行画图（如画出cost的图）
- 大测试2
  - 【设置Setting，初始化模型并调用train，展现结果即可】
  - 纵向比较某模型在不同任务的性能，模型任选一个即可
  - 数据集使用SVHN数据集，完成谷歌街景数字识别任务
  - 参数可以写在json配置文件中(建议)，也可以在代码中设置参数
  - 提供训练计时，输出训练集和测试集的各项指标
  - 如有必要，可进行画图（如画出cost的图）
- numba加速3
  - 支持cpu计算加速即可，不必实现gpu加速(内容太多了)
  - 如有必要，可以比较一下加速前后的运算时间
  - 【我之前看的部分信息】
    - [官方简易教程链接](https://numba.pydata.org/numba-doc/latest/user/5minguide.html)
    - 通过预编译，规避实时解析带来的事件损耗
    - 主要应用于numpy计算、循环、数值运算的部分
    - 貌似就是给函数加上修饰器@jit(nopython=True)或@njit即可？
    - 修饰的函数第一次运行自动编译，比较慢；后续运行就很快
- 实验报告和PPT
  - 绪论背景部分4
    - 目的意义：大概吹一吹，我再补一点总体的框架思想
    - 相关工作：大致是介绍已有的框架，使用情况或者特性
  - 方法实现的模块实现：各部分实现同学编写
  - 实验结果与结果分析：由两个大测试的同学编写，单元测试可能简短叙述
  - 其他部分（项目安排、总体设计、总的结论）我来写
  - 课堂展示（大概是PPT）：报告写完后我再来搞
