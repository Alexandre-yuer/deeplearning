卷积神经网络原理
    卷积神经网络介绍
    卷积神经网络 - 结构
        卷积层
        激活函数
        池化层
        全连接层
    Mnist数据集 - 使用卷积神经网络
验证码识别

4.1 卷积神经网络简介
    1）与传统多层神经网络对比
        输入层
        隐藏层
            卷积层
            激活层
            池化层
                pooling layer
                subsample
            全连接层
        输出层
    2）发展历史
    3）卷积网络ImageNet比赛错误率
4.2 卷积神经网络原理
    卷积神经网络 - 结构
        卷积层
            通过在原始图像上平移来提取特征
        激活层
            增加非线性分割能力
        池化层
            减少学习的参数，降低网络的复杂度（最大池化和平均池化）
        全连接层
4.2.2 卷积层（Convolutional Layer）
    卷积核 - filter - 过滤器 - 卷积单元 - 模型参数
        个数
        大小 1*1 3*3 5*5
            卷积如何计算？
            输入
                5*5*1 filter 3*3*1 步长 1
            输出
                3*3*1
        步长
            输入
                5*5*1 filter 3*3*1 步长2
            输出
                2*2*1
        零填充的大小
    6 总结-输出大小计算公式
    7 多通道图片如何观察
        输入图片
            5*5*3 filter 3*3*3 + bias 2个filter 步长2
            H1=5
            D1=3
            K=2
            F=3
            S=2
            P=1
            H2=(5-3+2)/2+1=3
            D2=2

        输出
            3*3*2
    卷积网络API
        tf.nn.conv2d(input, filter, strides=, padding=)
        input：输入图像
            要求：形状[batch,heigth,width,channel]
            类型为float32,64
        filter:
            weights
            变量initial_value=random_normal(shape=[F, F, 3/1, K])
        strides:
            步长 1
             [1, 1, 1, 1]
        padding: “SAME”
            “SAME”：越过边缘取样
            “VALID”：不越过边缘取样
    1）掌握filter要素的相关计算公式
    2）filter大小
        1x1，3x3，5x5
      步长 1
    3）每个过滤器会带有若干权重和1个偏置
    4.2.3 激活函数
        sigmoid
        1/(1+e^-x)
            1）计算量相对大
            2）梯度消失
            3）输入的值的范围[-6, 6]
        Relu的好处
            1）计算速度快
            2）解决了梯度消失
            3）图像没有负的像素值
        tf.nn.relu(features)
    4.2.4 池化层(Polling)
        利用了图像上像素点之间的联系
        tf.nn.max_pool(value, ksize=, strides=, padding=)
            value:
                4-D Tensor形状[batch, height, width, channels]
            ksize：
               池化窗口大小，[1, 2, 2, 1]
            strides:
                步长大小，[1, 2, 2, 1]
            padding：“SAME”
4.3 案例：CNN-Mnist手写数字识别
    4.3.1 网络设计
        第一个卷积大层：
            卷积层：
                32个filter 大小5*5 步长：1 padding="SAME"
                 tf.nn.conv2d(input, filter, strides=, padding=)
                 input：输入图像 [None, 28, 28, 1]
                     要求：形状[batch,heigth,width,channel]
                     类型为float32,64
                 filter:
                     weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 1, 32]))
                     bias = tf.Variable(initial_value=tf.random_normal(shape=[32]))
                     变量initial_value=random_normal(shape=[F, F, 3/1, K])
                 strides:
                     步长 1
                      [1, 1, 1, 1]
                 padding: “SAME”
                     “SAME”：越过边缘取样
                     “VALID”：不越过边缘取样
                 输出形状：
                 [None, 28, 28, 32]
            激活：
                Relu
                tf.nn.relu(features)
            池化：
                输入形状：[None, 28, 28, 32]
                大小2*2 步长2
                输出形状：[None, 14, 14, 32]
        第二个卷积大层：
            卷积层：
                64个filter 大小5*5 步长：1 padding="SAME"
                输入：[None, 14, 14, 32]
                tf.nn.conv2d(input, filter, strides=, padding=)
                input：[None, 14, 14, 32]
                    要求：形状[batch,heigth,width,channel]
                    类型为float32,64
                filter:
                    weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64]))
                    bias = tf.Variable(initial_value=tf.random_normal(shape=[64]))
                    变量initial_value=random_normal(shape=[F, F, 3/1, K])
                strides:
                    步长 1
                     [1, 1, 1, 1]
                padding: “SAME”
                    “SAME”：越过边缘取样
                    “VALID”：不越过边缘取样
                输出形状：
                [None, 14, 14, 64]
            激活：
                Relu
                tf.nn.relu(features)
            池化：
                输入形状：[None, 14, 14, 64]
                大小2*2 步长2
                输出形状：[None, 7, 7, 64]
        全连接
            tf.reshape()
            [None, 7, 7, 64]->[None, 7*7*64]
            [None, 7*7*64] * [7*7*64, 10] = [None, 10]
            y_predict = tf.matmul(pool2, weithts) + bias
        调参->提高准确率？
        1）学习率
        2）随机初始化的权重、偏置的值
        3）选择好用的优化器
        4）调整网络结构
4.4 网络结构与优化
    4.4.1 网络的优化和改进
    4.4.2 卷积神经网络的拓展了解
        1 常见网络模型
        2 卷积网络其它用途
4.5 实战：验证码图片识别
    验证码识别实战
        1）数据集
            图片1 -> NZPP 一个样本对应4个目标值 -> sigmoid交叉熵
            一张手写数字的图片 -> 0~9之间的某一个数 一个样本对应一个目标值 -> softmax交叉熵
            切割 -> 不具备通用性
            [0,0,1,0……]
            NZPP -> [13, 25, 15, 15]
            [4, 26]
            -> [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        2）对数据集中
            特征值 目标值 怎么用
        3）如何分类？
            如何比较输出结果和真实值的正确性？
            如何衡量损失？
                手写数字识别案例
                    softmax+交叉熵
                        [4, 26] -> [4*26]
                sigmoid交叉熵
            准确率如何计算？
                核心：对比真实值和预测值最大值所在位置
                手写数字识别案例
                y_predict[None, 10]
                tf.argmax(y_predict, axis=1)
                y_predict[None, 4, 26]
                tf.argmax(y_predict, axis=2/-1)
                [True,
                True,
                True,
                False] -> tf.reduce_all() -> False
        4）流程分析
            1）读取图片数据
                filename -> 标签值
            2）解析csv文件，将标签值NZPP->[13, 25, 15, 15]
            3）将filename和标签值联系起来
            4）构建卷积神经网络->y_predict
            5）构造损失函数
            6）优化损失
            7）计算准确率
            8）开启会话、开启线程
        5）代码实现
