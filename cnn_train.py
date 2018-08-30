import os
import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import cnn_inference
from parameters import Parameters

para = Parameters()
RUN_COUNT1 = para.run_count1  #single
RUN_COUNT2 = para.run_count2  #2
RUN_COUNT3 = para.run_count3  #3

DATA_SIZE = RUN_COUNT1 + RUN_COUNT2 + RUN_COUNT3
# DATA_SIZE = 6272

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.96
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000000000000000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"

input_X = 'data/train/trainX_6321.txt'
input_Y = 'data/train/trainY_6321.txt'


def train():
    x = tf.placeholder(tf.float32, [None, cnn_inference.IMAGE_SIZE2], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, cnn_inference.OUTPUT_NODE], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = cnn_inference.inference(x, 0, None)
    # y = tf.nn.softmax(logits)
    # y_ = tf.nn.softmax(y_)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    # cross_entropy_mean = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DATA_SIZE / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    trainx = np.loadtxt(input_X)
    trainy = np.loadtxt(input_Y)
    for i in range(len(trainy)):
        trainy[i] = trainy[i] / len(np.where(trainy[i] > 0.9)[0])

    # x_a = tf.expand_dims(trainx, 1)
    # trainx = tf.expand_dims(x_a, -1)  # -1表示最后一维

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        # trainx = trainx.eval(session=sess)
        for i in range(TRAINING_STEPS):
            start = (i * BATCH_SIZE) % DATA_SIZE
            end = min(start + BATCH_SIZE, DATA_SIZE)

            # 每次选取batch_size个样本进行训练

            # _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: trainx[start: end], y_: trainy[start: end]})
            _, step = sess.run([train_op, global_step], feed_dict={x: trainx[start: end], y_: trainy[start: end]})

            # 通过选取样本训练神经网络并更新参数
            #sess.run(train_step, feed_dict={x: trainx[start:end], y_: trainy[start:end]})
            # 每迭代1000次输出一次日志信息
            if i % 1000 == 0:
                # 计算所有数据的交叉熵
                total_cross_entropy = sess.run(loss, feed_dict={x: trainx, y_: trainy})
                # total_mse = sess.run(loss, feed_dict={x: trainx, y_: trainy})
                # train_accuracy = sess.run(accuracy, feed_dict={x: X, y_: Y})
                # 输出交叉熵之和
                #print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
                print("After %d training step(s), loss on training "
                      "batch is %g" % (i, total_cross_entropy))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print("save the model")


def main(argv=None):
    # mnist = input_data.read_data_sets("path/to/mnist_data", one_hot=True)
    train()


if __name__ == '__main__':
    main()
