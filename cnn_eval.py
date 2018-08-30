import time
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cnn_inference
import cnn_train
from parameters import Parameters

EVAL_INTERVAL_SECS = 10

para = Parameters()

RUN_COUNT1 = para.run_count1   #single
RUN_COUNT2 = para.run_count2  #2
RUN_COUNT3 = para.run_count3  #3

n = RUN_COUNT1 + RUN_COUNT2 + RUN_COUNT3

input_X = 'data/val/valX_6321.txt'
input_Y = 'data/val/valY_6321.txt'

valx = np.loadtxt(input_X)
valy = np.loadtxt(input_Y)

# x_a = np.expand_dims(valx, 1)
# valx = np.expand_dims(x_a, -1)  # -1表示最后一维

# X = np.loadtxt(input_X)
# Y = np.loadtxt(input_Y)
# BATCH_SIZE = 1024


def evaluate():
    with tf.Graph().as_default() as g:
        # x = tf.placeholder(tf.float32, [
        #    6321,
        #    cnn_inference.IMAGE_SIZE1,
        #    cnn_inference.IMAGE_SIZE2,
        #    cnn_inference.NUM_CHANNELS],
        #                   name="x-input")
        x = tf.placeholder(tf.float32, [None, cnn_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, cnn_inference.OUTPUT_NODE], name='y-input')
        # validate_feed = {x: valx, y_: valy}
        logits = cnn_inference.inference(x, 0, None)
        y = tf.nn.softmax(logits)  
        # percentage = tf.constant(0.1)
        # correct_prediction = tf.equal(tf.where(y > 0.1), tf.where(valy > percentage))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(
            cnn_train.MOVING_AVERAGE_DECAY
        )
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                aaa = 0
                bbb = 0
                ccc = 0
                pred_label = []
                val_label = []
                ckpt = tf.train.get_checkpoint_state(
                    cnn_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path\
                        .split('/')[-1].split('-')[-1]
                    # accuracy_score = sess.run(accuracy,
                    #                           feed_dict=validate_feed)
                    # i = 0
                    # while True:
                        # start = (i * BATCH_SIZE) % len(valx)
                        # end = min(start + BATCH_SIZE, len(valx))
                    Y_ = sess.run(y, feed_dict={x: valx})  # predict
                    
                    Y_pred = Y_
                    # print(len(Y_))
                    # print(len(Y_pred))

                    for i in range(len(valx)):
                        # Y_pred[i] = (Y_[i] - min(Y_[i])) / (max(Y_[i]) - min(Y_[i]))
                        pred_eff_idx = np.where(Y_pred[i] > 0.1)
                        y_pred_idx = (pred_eff_idx[0] + 1).tolist()
                        pred_label.append(y_pred_idx)

                        val_eff_idx = np.where(valy[i] > 0.8)
                        y_val_idx = (val_eff_idx[0] + 1).tolist()
                        # y_val_idx = np.add(y_val_idx + 1)
                        val_label.append(y_val_idx)
                    # print(Y_pred[0])
                    # print(Y_pred[321])
                    print(Y_pred[1921][np.where(Y_pred[1921]>0.1)[0]])
                    
                    # print(pred_label)
                    print(pred_label[1921])
                    # print(pred_label[6320])
                    for j in range(len(valx)):
                        if val_label[j] == pred_label[j]:
                            if len(val_label[j]) == 1:
                                aaa += 1
                            if len(val_label[j]) == 2:
                                bbb += 1
                            if len(val_label[j]) == 3:
                                ccc += 1
                        else:
                            print(j)

                    accuracy_1 = aaa / RUN_COUNT1
                    accuracy_2 = bbb / RUN_COUNT2
                    accuracy_3 = ccc / RUN_COUNT3
                    print("After %s training step(s), validation "
                          "accuracy = %g" % (global_step, accuracy_1))
                    print("After %s training step(s), validation "
                          "accuracy = %g" % (global_step, accuracy_2))
                    print("After %s training step(s), validation "
                          "accuracy = %g" % (global_step, accuracy_3))
                else:
                    print('No checkpoint file found')
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    # mnist = input_data.read_data_sets("/path/to/mnist_data", one_hot=True)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
