import tensorflow as tf
from cnn import CNN
import numpy as np

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [np.asarray(data[i]).flatten() for i in idx] 
    labels_shuffle = [np.eye(10, dtype=int)[labels[i][0]] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

class NetManager():
    def __init__(self, num_input, num_classes, learning_rate, cifar10,
                 max_step_per_action=5500*3,
                 bathc_size=100,
                 dropout_rate=0.85):

        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.cifar10 = cifar10

        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size
        self.dropout_rate = dropout_rate

    def get_reward(self, action, step, pre_acc):
        action = [action[0][0][x:x+4] for x in range(0, len(action[0][0]), 4)]
        cnn_drop_rate = [c[3] for c in action]
        with tf.Graph().as_default() as g:
            with g.container('experiment'+str(step)):
                model = CNN(self.num_input, self.num_classes, action)
                loss_op = tf.reduce_mean(model.loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_op = optimizer.minimize(loss_op)

                with tf.Session() as train_sess:
                    init = tf.global_variables_initializer()
                    train_sess.run(init)

                    for step in range(self.max_step_per_action):
                        batch_x, batch_y = next_batch(self.bathc_size, self.cifar10["x_train"], self.cifar10["y_train"])                        
                        feed = {model.X: batch_x,
                                model.Y: batch_y,
                                model.dropout_keep_prob: self.dropout_rate,
                                model.cnn_dropout_rates: cnn_drop_rate}
                        _ = train_sess.run(train_op, feed_dict=feed)

                        if step % 100 == 0:
                            # Calculate batch loss and accuracy
                            loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                            print("Step " + str(step) +
                                  ", Minibatch Loss= " + "{:.4f}".format(loss) +
                                  ", Current accuracy= " + "{:.3f}".format(acc))
                    batch_x, batch_y = next_batch(10000, self.cifar10["x_test"], self.cifar10["y_test"])
                    loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                    print("!!!!!!acc:", acc, pre_acc)
                    if acc - pre_acc <= 0.01:
                        return acc, acc 
                    else:
                        return 0.01, acc
                    
