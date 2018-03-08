import datetime

import os

from src.analysis import compute_accuracy, clean_log_folder
from src.forward_propagation import *
from src.initialization import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(id_of_batch=1, learning_rate=0.001, mini_batch_size=512, number_of_steps=100, load_parameters=False):
    tf.reset_default_graph()

    x_train, y_train = get_training_set(id_of_batch)
    x_train, y_train = x_train, y_train
    x_dev, y_dev = get_dev_set()
    x_dev, y_dev = x_dev, y_dev
    x_test, y_test = get_test_set()
    x_test, y_test = x_test, y_test

    _, h, w, c = x_train.shape

    x, y = create_placeholders(h, w, c)
    parameters = initialize_parameters()
    parameters['start_step'] = tf.Variable(0)

    z = forward_prop(x, parameters)

    cost = compute_cost(z, y)
    tf.summary.scalar("cost", cost)

    accuracy = compute_accuracy(z, y)
    tf.summary.scalar("accuracy", accuracy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if load_parameters:
            saver.restore(sess, 'tmp/model.ckpt')
        else:
            clean_log_folder()
        start_step = parameters['start_step'].eval()
        merged_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log/train')
        dev_writer = tf.summary.FileWriter('log/dev')
        test_writer = tf.summary.FileWriter('log/test')

        step_time = datetime.datetime.now()
        print('Start time:', step_time)
        for step in range(start_step, start_step + number_of_steps):
            if mini_batch_size:
                mini_batches = random_mini_batches(x_train, y_train, mini_batch_size)
                for mini_batch in mini_batches:
                    this_x, this_y = mini_batch
                    sess.run(optimizer, feed_dict={x: this_x, y: this_y})
                train_summary, c, ac = sess.run([merged_summary_op, cost, accuracy],
                                                feed_dict={x: x_train, y: y_train})
            else:
                train_summary, c, ac, _ = sess.run([merged_summary_op, cost, accuracy, optimizer],
                                                   feed_dict={x: x_train, y: y_train})
            train_writer.add_summary(train_summary, step - 1)
            train_writer.flush()

            if step % 10 == 0:
                if step:
                    print('step:', step)
                    print('seconds per step:', (datetime.datetime.now() - step_time).total_seconds() / 10)
                step_time = datetime.datetime.now()
                print('train cost:', c, 'train accuracy:', ac)

                dev_summary, c, ac = sess.run([merged_summary_op, cost, accuracy], feed_dict={x: x_dev, y: y_dev})
                print('dev cost:', c, 'dev accuracy:', ac)
                dev_writer.add_summary(dev_summary, step)
                dev_writer.flush()

                test_summary, c, ac = sess.run([merged_summary_op, cost, accuracy], feed_dict={x: x_test, y: y_test})
                print('test cost:', c, 'test accuracy:', ac)
                test_writer.add_summary(test_summary, step)
                test_writer.flush()
                print('=' * 100)

        parameters['start_step'] = parameters['start_step'].assign_add(number_of_steps)
        sess.run(list(parameters.values()))
        saver.save(sess, 'tmp/model.ckpt')
        if not load_parameters:
            train_writer.add_graph(sess.graph)
    print('Finish time:', datetime.datetime.now())


if __name__ == '__main__':
    train(id_of_batch=1, learning_rate=0.001, mini_batch_size=0, number_of_steps=100, load_parameters=True)
