import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.examples.tutorials.mnist import input_data

def full_connection():
    """
    用全连接层来对手写数字进行识别
    :return:
    """
    # 1.准备数据
    mnist = input_data.read_data_sets("./minist_data",one_hot=True)
    x = tf.placeholder(dtype=tf.float32,shape=[None,784])
    y_true = tf.placeholder(dtype=tf.float32,shape=[None,10])

    # #查看训练数据大小
    # print(mnist.train.images.shape)
    # print(mnist.train.labels.shape)
    # #查看验证数据大小
    # print(mnist.validation.images.shape)
    # print(mnist.validation.labels.shape)
    # #查看测试数据大小
    # print(mnist.test.images.shape)
    # print(mnist.test.labels.shape)
    #
    # print("x:\n",x)
    # print("y_true:\n",y_true)

    # 2.构建模型
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784,10]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict = tf.matmul(x,weights) + bias

    # 3.构造损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))

    # 4.优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)


    # 5.准确率计算
    bool_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(bool_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:

        sess.run(init)
        image,label = mnist.train.next_batch(100)

        print("训练之前，损失为:%f" % sess.run(error,feed_dict={x:image,y_true:label}))

        # 开始训练
        for i in range(3000):

            _, loss_value, accuracy_value = sess.run([optimizer, error, accuracy], feed_dict={x: image, y_true: label})

            print("第%d次的损失为%f，准确率为%f" % (i + 1, loss_value, accuracy_value))

    return None


if __name__ == "__main__":
    full_connection()
