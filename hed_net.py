# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import yaml


class HED(object):
    def __init__(self, height, width, channel):
        self.height = height
        self.width = width
        self.x = tf.placeholder(tf.float32, (None, height, width, channel))
        with open('cfg.yml') as file:
            self.cfg = yaml.load(file)

    def vgg_hed(self):
        bn1, relu1 = self.block(input_tensor=self.x, filters=64, iteration=2, dilation_rate=[(4, 4), (1, 1)], name='block1')
        mp1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool1')

        bn2, relu2 = self.block(input_tensor=mp1, filters=128, iteration=2, name='block2')
        mp2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool2')

        bn3, relu3 = self.block(input_tensor=mp2, filters=256, iteration=3, name='block3')
        mp3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool3')

        bn4, relu4 = self.block(input_tensor=mp3, filters=512, iteration=3, name='block4')
        mp4 = tf.layers.max_pooling2d(inputs=relu4, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool4')

        bn5, relu5 = self.block(input_tensor=mp4, filters=512, iteration=3, name='block5')

        self.side1 = self.side(input_tensor=bn1, stride=(1, 1), name='side1', deconv=False)
        self.side2 = self.side(input_tensor=bn2, stride=(2, 2), name='side2')
        self.side3 = self.side(input_tensor=bn3, stride=(4, 4), name='side3')
        self.side4 = self.side(input_tensor=bn4, stride=(8, 8), name='side4')
        self.side5 = self.side(input_tensor=bn5, stride=(16, 16), name='side5')
        # side1 = tf.layers.conv2d(inputs=bn1, filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same',
        #                          activation=tf.nn.relu, name='side1')
        # side2 = tf.layers.conv2d_transpose(inputs=bn2, filters=1, kernel_size=(3, 3),
        #                                    strides=(2, 2), padding='same', activation=tf.nn.relu, name='side2')
        # side3 = tf.layers.conv2d_transpose(inputs=bn3, filters=1, kernel_size=(3, 3),
        #                                    strides=(4, 4), padding='same', activation=tf.nn.relu, name='side3')
        # side4 = tf.layers.conv2d_transpose(inputs=bn4, filters=1, kernel_size=(3, 3),
        #                                    strides=(8, 8), padding='same', activation=tf.nn.relu, name='side4')
        # side5 = tf.layers.conv2d_transpose(inputs=bn5, filters=1, kernel_size=(3, 3),
        #                                    strides=(16, 16), padding='same', activation=tf.nn.relu, name='side5')
        sides = tf.concat(values=[self.side1, self.side2, self.side3, self.side4, self.side5], axis=3)
        self.fused_side = tf.layers.conv2d(inputs=sides, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                           use_bias=False, kernel_initializer=tf.constant_initializer(0.2), name='fused_side')
        return self.side1, self.side2, self.side3, self.side4, self.side5, self.fused_side

    def block(self, input_tensor, filters, iteration, dilation_rate=None, name=None):
        if dilation_rate is None:
            dilation_rate = [(1, 1)]
        if len(dilation_rate) == 1:
            dilation_rate *= iteration

        regularizer = tf.contrib.layers.l2_regularizer(self.cfg['weight_decay_ratio'])
        with tf.variable_scope(name):
            relu = input_tensor
            for it in range(iteration):
                tp_dilation_rate = dilation_rate.pop(0)
                print(tp_dilation_rate)
                conv = tf.layers.conv2d(inputs=relu, filters=filters,
                                        kernel_size=(3, 3), strides=(1, 1), padding='same',
                                        activation=None, use_bias=True,
                                        kernel_regularizer=regularizer,
                                        dilation_rate=tp_dilation_rate,
                                        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.5),
                                        name='conv{:d}'.format(it))
                # bn = tf.layers.batch_normalization(inputs=conv, axis=-1, name='bn{:d}'.format(it))
                bn = conv
                relu = tf.nn.relu(bn, name='relu{:d}'.format(it))
        return relu, relu

    def side(self, input_tensor, stride, name, deconv=True):
        with tf.variable_scope(name):
            side = tf.layers.conv2d(inputs=input_tensor, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                    padding='same',
                                    activation=None,
                                    bias_initializer=tf.constant_initializer(value=0),
                                    kernel_initializer=tf.constant_initializer(value=0),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
            if deconv:
                side = tf.layers.conv2d_transpose(inputs=side, filters=1, kernel_size=(2*stride[0], 2*stride[1]),
                                                  strides=stride, padding='same',
                                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  bias_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.cfg['weight_decay_ratio']),
                                                  activation=None)
            side = tf.image.resize_images(images=side, size=(self.height, self.width),
                                          method=tf.image.ResizeMethod.BILINEAR)
        return side

    def evaluate(self):
        # evaluation criteria
        # accuracy

        # precision

        # recall

        # F1 score
        pass

    def summary(self):
        max_outputs = 1
        tf.summary.image(name='orig_image_sm', tensor=self.x, max_outputs=max_outputs)
        tf.summary.image(name='side1_im', tensor=tf.sigmoid(self.side1), max_outputs=max_outputs, )
        tf.summary.image(name='side2_im', tensor=tf.sigmoid(self.side2), max_outputs=max_outputs, )
        tf.summary.image(name='side3_im', tensor=tf.sigmoid(self.side3), max_outputs=max_outputs, )
        tf.summary.image(name='side4_im', tensor=tf.sigmoid(self.side4), max_outputs=max_outputs, )
        tf.summary.image(name='side5_im', tensor=tf.sigmoid(self.side5), max_outputs=max_outputs, )
        tf.summary.image(name='fused_side_im', tensor=tf.sigmoid(self.fused_side), max_outputs=max_outputs, )

        tf.summary.histogram(name='side1_hist', values=tf.sigmoid(self.side1))
        tf.summary.histogram(name='side2_hist', values=tf.sigmoid(self.side2))
        tf.summary.histogram(name='side3_hist', values=tf.sigmoid(self.side3))
        tf.summary.histogram(name='side4_hist', values=tf.sigmoid(self.side4))
        tf.summary.histogram(name='side5_hist', values=tf.sigmoid(self.side5))
        tf.summary.histogram(name='fused_side_hist', values=tf.sigmoid(self.fused_side))

    def assign_init_weights(self, sess=None):
        with open(self.cfg['init_weights'], 'rb') as file:
            weights = np.load(file, encoding='latin1').item()
        with tf.variable_scope('block1', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv1_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv1_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv1_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv1_2'][1]))
        print('assign first block done !')
        with tf.variable_scope('block2', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv2_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv2_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv2_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv2_2'][1]))
        print('assign second block done !')
        with tf.variable_scope('block3', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv3_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv3_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv3_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv3_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv3_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv3_3'][1]))
        print('assign third block done !')
        with tf.variable_scope('block4', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv4_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv4_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv4_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv4_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv4_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv4_3'][1]))
        print('assign fourth block done !')
        with tf.variable_scope('block5', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv5_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv5_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv5_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv5_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv5_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv5_3'][1]))
        weights = None  # gc
        print('assign fifth block done !')
        print('net initializing successfully with vgg16 weights trained by imagenet data')


if __name__ == "__main__":
    hed = HED(112, 112, 3)
    ipt = np.ones((12, 112, 112, 3))

    s1 = hed.vgg_hed()
    hed.summary()
    merged_summary = tf.summary.merge_all()
    for s in s1:
        print(s)
    print(tf.trainable_variables())
    with tf.variable_scope('block1', reuse=True):
        k = tf.get_variable(name='conv0/kernel')
        print(k)
    # # tf.get_default_graph()
    # with tf.Session() as sess:
    #     # `sess.graph` provides access to the graph used in a `tf.Session`.
    #     summary_writer = tf.summary.FileWriter("/Users/whyguu/Desktop/log/", sess.graph)
    #
    #     # Perform your computation...
    #     # for i in range(1000):
    #     #     sess.run(train_op)
    #     #     # ...
    #     summary_writer.add_summary(merged_summary)
    #     summary_writer.close()
    # print(s1)
    # with tf.variable_scope('block1', reuse=True):
    #     print(tf.get_variable('conv1/kernel'))
    # summary = tf.summary.image(name='hah', tensor='block1/side1/kernel')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     a = sess.run(s1, feed_dict={hed.x: ipt})
    #     print(a.shape)
    #     print(tf.trainable_variables())
    #     with tf.variable_scope('block1', reuse=True):
    #         print(sess.run(tf.get_variable('conv1/kernel')))


