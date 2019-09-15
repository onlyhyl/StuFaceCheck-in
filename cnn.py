import tensorflow as tf
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

SIZE = 64
x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

# 权重
def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

# 偏移量
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 最大池化层
def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# dropout层
def dropout(x, keep):
    return tf.nn.dropout(x, keep)

# 搭建cnn网络
def cnnLayer(classnum):
    # 使用卷积神经网络的层层特点，进行图像识别完成分类
    W1 = weightVariable([3, 3, 3, 32])  # 卷积核大小（3,3），输入通道（3），输出通道即核的个数（32）
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob_5)

    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层（可以不用。
    # 卷积取局部特征（根据局部特征值也能分辨出某一类东西如A的人眼、嘴的特征值和B的人眼、嘴的特征值不同），
    # 全连接则把局部特征重新通过权值组装成完整的图（把A的人眼、嘴和B的人眼、嘴组合成各自完整A、B图的特征值） 实现最终分类（分出哪些图是属于A，哪些图是属于B））
    Wf = weightVariable([8 * 16 * 32, 512])
    bf = biasVariable([512])
    # drop3 = [8,8,64]
    drop3_flat = tf.reshape(drop3, [-1, 8 * 16 * 32])
    # drop3_flat = [8*16*32]
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, classnum])
    bout = weightVariable([classnum])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    print(out)
    return out
    # 经cnn算法分类，最终返回所有照片中属于的类别（图片1,2,4属于A，图片3,5,6属于B）


def train(train_x, train_y, tfsavepath):
    log.debug('train')
    out = cnnLayer(train_y.shape[1])
    # 使用softmax
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
    # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=y_data))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较经cnn分类出来的图片标签和真正的标签之间的差距（图片1：cnn分类出属于A、而实际上属于C），即精度（只是拿得到的标签做对比）
    # tf.argmax(input, axis) axis=0/1，返回每列/每行最大元素的索引
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 测试批次量及轮数对模型的影响
        # 每批（每个batch）有128（batch_size）个样本，即数据集被分为num_batch批。训练每批中128（batch_size）个样本后模型权重更新。
        for batch_size in [64, 128]:
            print("-" * 20)
            print("本次训练batch_size为：", batch_size)
            num_batch = len(train_x) // batch_size  # 200+132(train_x) = 332//10 = 33
            # 此处其实为定义epoch，即10个epoch。一个epoch将训练完所有批次（共num_batch批，每批训练完成后权重都会更新）后权重也都会更新。

            nes = []
            losses = []
            acces = []
            train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)
            for num_epoch in range(10, 26):
                print("-" * 10)
                print("本次将进行%s个epoch的训练。" % num_epoch)
                for n in range(num_epoch):
                    print("第%d个epoch    :" % (n + 1))
                    # r = [332个<332的随机数]
                    r = np.random.permutation(len(train_x))
                    # 随机打乱数据排序
                    train_x = train_x[r, :]
                    train_y = train_y[r, :]

                    for i in range(num_batch):
                        print("第%d批    :" % (i + 1))
                        batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                        batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                        _, loss = sess.run([train_step, cross_entropy],
                                           feed_dict={x_data: batch_x, y_data: batch_y,
                                                      keep_prob_5: 0.75, keep_prob_75
                                                      : 0.75})
                        acc = accuracy.eval({x_data: test_x, y_data: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                        print(n * num_batch + i, loss)

                    print("-" * 5)
                print("loss:%s, acc:%s (batch_size:%s, num_epoch:%s)" % (loss, acc, batch_size, num_epoch))
                # 当损失<0.1精度>0.9时训练结束
                if loss < 0.1 and acc > 0.9:
                    print("stop!")
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x_data: test_x, y_data: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    print('after %d times run: accuracy is %d' %(num_epoch, acc))
                    saver.save(sess, tfsavepath)
                    print("-" * 10)
                    break
                print("-" * 10)

            # plt查看模型训练结果
            # plt.plot(nes, losses, 'bo', label='Training Loss')
            # plt.title('Training loss(batch_size:%s)' % batch_size)
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.show()
            # plt.clf()
            # plt.savefig('loss.png')
            # plt.savefig('./tensorflow2-460/loss(batch_size:%s, num_epoch:%s).png' % (batch_size, num_epoch))
            # plt.plot(nes, acces, 'b', label='Training Acc')
            # plt.title('Training acc(batch_size:%s)' % batch_size)
            # plt.xlabel('Epochs')
            # plt.ylabel('Acc')
            # plt.legend()
            # plt.show()
            # plt.savefig('acc.png')
            # plt.savefig('tensorflow2-460/acc(batch_size:%s, num_epoch:%s).png' % (batch_size, num_epoch))


def validate(test_x, tfsavepath):
    """
    验证结果
    :param test_x: 测试数据
    :param tfsavepath: 模型存储路径
    :return:
    """
    output = cnnLayer(4)
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tfsavepath)
        res = sess.run([predict, tf.argmax(output, 1)],
                       feed_dict={x_data: test_x,
                                  keep_prob_5: 1.0, keep_prob_75: 1.0})

        return res


if __name__ == '__main__':
    pass
