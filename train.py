import datetime

import os

from src.analysis import compute_accuracy
from src.forward_propagation import *
from src.initialization import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(learning_rate=0.001, mini_batch=False, minibatch_size=512, number_of_steps=100):
    tf.reset_default_graph()

    x_train, y_train = get_training_set(1)
    x_train, y_train = x_train[:500], y_train[:500]
    x_dev, y_dev = get_dev_set()
    x_dev, y_dev = x_dev[:200], y_dev[:200]
    x_test, y_test = get_test_set()
    x_test, y_test = x_test[:200], y_test[:200]

    _, h, w, c = x_train.shape

    x, y = create_placeholders(h, w, c)
    parameters = initialize_parameters()

    z = forward_prop(x, parameters)

    cost = compute_cost(z, y)
    tf.summary.scalar("cost", cost)

    accuracy = compute_accuracy(z, y)
    tf.summary.scalar("accuracy", accuracy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        merged_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log/plot_1')
        dev_writer = tf.summary.FileWriter('log/plot_2')
        test_writer = tf.summary.FileWriter('log/plot_3')
        sess.run(tf.global_variables_initializer())

        start_time = datetime.datetime.now()
        print('Start time:', start_time)
        for step in range(number_of_steps):
            if mini_batch:
                mini_batches = random_mini_batches(x_train, y_train, minibatch_size)
                for mini_batch in mini_batches:
                    this_x, this_y = mini_batch
                    sess.run(optimizer, feed_dict={x: this_x, y: this_y})
            else:
                sess.run(optimizer, feed_dict={x: x_train, y: y_train})

            if step % 10 == 0:
                train_error = sess.run(merged_summary_op, feed_dict={x: x_train, y: y_train})
                train_writer.add_summary(train_error, step)
                train_writer.flush()

                dev_error = sess.run(merged_summary_op, feed_dict={x: x_dev, y: y_dev})
                dev_writer.add_summary(dev_error, step)
                dev_writer.flush()

                test_error = sess.run(merged_summary_op, feed_dict={x: x_test, y: y_test})
                test_writer.add_summary(test_error, step)
                test_writer.flush()

                print('step:', step)
                if step:
                    print('seconds per step: ', (datetime.datetime.now() - start_time).total_seconds() / step)
                print('=' * 100)
    print('Finish time:', datetime.datetime.now())


if __name__ == '__main__':
    train(learning_rate=0.001, mini_batch=False, number_of_steps=100)
