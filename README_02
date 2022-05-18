深度学习 day02

数据IO操作
    三种
     占位符 & feed_dict搭配使用
     QueueRunner
        通用文件读取流程
            图片
            二进制数据
            TFRecords
神经网络基础
    神经网络原理
    手写数字识别案例

3.1 文件读取流程
    多线程 + 队列
    3.1.1 文件读取流程
        1）构造文件名队列
            file_queue = tf.train.string_input_producer(string_tensor,shuffle=True)
        2）读取与解码
            文本：
                读取：tf.TextLineReader()
                解码：tf.decode_csv()
            图片：
                读取：tf.WholeFileReader()
                解码：
                    tf.image.decode_jpeg(contents)
                    tf.image.decode_png(contents)
            二进制：
                读取：tf.FixedLengthRecordReader(record_bytes)
                解码：tf.decode_raw()
            TFRecords
                读取：tf.TFRecordReader()
            key, value = 读取器.read(file_queue)
            key：文件名
            value：一个样本
        3）批处理队列
            tf.train.batch(tensors, batch_size, num_threads = 1, capacity = 32, name=None)
        手动开启线程
            tf.train.QueueRunner()
            开启会话：
                tf.train.start_queue_runners(sess=None, coord=None)
3.2 图片数据
    3.2.1 图像基本知识
        文本  特征词 -> 二维数组
        字典  one-hot -> 二维数组
        图片  像素值
        1 图片三要素
            黑白图、灰度图
                一个通道
                    黑[0, 255]白
            彩色图
                三个通道
                    一个像素点 三个通道值构成
                    R [0, 255]
                    G [0, 255]
                    B [0, 255]
        2 TensorFlow中表示图片
            Tensor对象
                指令名称、形状、类型
                shape = [height, width, channel]
        3 图片特征值处理
            [samples, features]
            为什么要缩放图片到统一大小？
            1）每一个样本特征数量要一样多
            2）缩小图片的大小
            tf.image.resize_images(images, size)
        4 数据格式
            存储：uint8
            训练：float32
    3.2.4 案例：狗图片读取
        1）构造文件名队列
            file_queue = tf.train.string_input_producer(string_tensor,shuffle=True)
        2）读取与解码
            读取：
                reader = tf.WholeFileReader()
                key, value = reader.read(file_queue)
            解码：
                image_decoded = tf.image.decode_jpeg(value)
        3）批处理队列
            image_decoded = tf.train.batch([image_decoded], 100, num_threads = 2, capacity=100)
        手动开启线程
3.3 二进制数据
    tensor对象
        shape:[height, width, channel] -> [32, 32, 3] [0, 1, 2] -> []
        [[32 * 32的二维数组],
        [32 * 32的二维数组],
        [32 * 32的二维数组]]
            --> [3, 32, 32] [channel, height, width] 三维数组的转置 [0, 1, 2] -> [1, 2, 0]
            [3, 2] -转置-> [2, 3]
        1）NHWC与NCHW
        T = transpose 转置

    3.3.2 CIFAR10 二进制数据读取
        流程分析：
            1）构造文件名队列
            2）读取与解码
            3）批处理队列
            开启会话
            手动开启线程
3.4 TFRecords
    3.4.1 什么是TFRecords文件
    3.4.2 Example结构解析
        cifar10
            特征值 - image - 3072个字节
            目标值 - label - 1个字节
        example = tf.train.Example(features=tf.train.Features(feature={
        "image":tf.train.Feature(bytes_list=tf.train. BytesList(value=[image])
        "label":tf.train.Feature(int64_list=tf.train. Int64List(value=[label]))
        }))
        example.SerializeToString()
    3.4.3 案例：CIFAR10数据存入TFRecords文件
        流程分析
    3.4.4 读取TFRecords文件API
        1）构造文件名队列
        2）读取和解码
            读取
            解析example
            feature = tf.parse_single_example(value, features={
            "image":tf.FixedLenFeature([], tf.string),
            "label":tf.FixedLenFeature([], tf.int64)
            })
            image = feature["image"]
            label = feature["label"]
            解码
            tf.decode_raw()
        3）构造批处理队列
3.5 神经网络基础
    3.5.1 神经网络
        输入层
            特征值和权重 线性加权
            y = w1x1 + w2x2 + …… + wnxn + b
            细胞核-激活函数
                sigmoid
                sign
        隐藏层
        输出层
    单个神经元 - 感知机
    感知机(PLA: Perceptron Learning Algorithm))
        x1, x2
        w1x1 + w2x2 + b = 常数
        w2x2 = -w1x1 - b + 常数
        x2 = kx1 + b
        x2 = kx1 + b
        x1 x2
        与问题
        0   0 0
        0   1 0
        1   0 0
        1   1 1
        异或问题
        0   0 0
        0   1 1
        1   0 1
        1   1 0
        单个神经元不能解决一些复杂问题
        1）多层神经元
        2）增加激活函数
3.6 神经网络原理
    逻辑回归
        y = w1x1 + w2x2 + …… + wnxn + b
        sigmoid -> [0, 1] -> 二分类问题
        损失函数：对数似然损失
    用神经网络进行分类
        假设函数
            y_predict =
            softmax - 多分类问题
        构造损失函数
            loss = 交叉熵损失
        优化损失
            梯度下降
        3.6.1 softmax回归 - 多分类问题
            假设要进行三分类
            2.3, 4.1, 5.6
        3.6.2 交叉熵损失
3.7 案例：Mnist手写数字识别
    3.7.1 数据集介绍
        1 特征值
            [None, 784] * W[784, 10] + Bias = [None, 10]
            构建全连接层：
            y_predict = tf.matmul(x, W) + Bias
            构造损失：
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict,name=None)
            如何计算准确率?
            np.argmax(y_predict, axis=1)
            tf.argmax(y_true, axis=1)
                y_predict [None, 10]
                y_true [None, 10]
            tf.equal()
            如何提高准确率？
                1）增加训练次数
                2）调节学习率
                3）调节权重系数的初始化值
                4）改变优化器
