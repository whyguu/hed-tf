# -*- coding: UTF-8 -*-
from __future__ import print_function
import tensorflow as tf
import yaml
import numpy as np


class HedLoss(object):
    def __init__(self, sides):
        self.sides = sides
        self.loss = 0.0
        self.floss = 0.0
        with open('cfg.yml') as file:
            self.cfg = yaml.load(file)
        self.label = tf.placeholder(tf.float32, (None, self.cfg['height'], self.cfg['width'], 1))
        # self.calc_loss()

    def calc_loss(self):
        if self.cfg['is_deep_supervised']:
            for n in range(len(self.sides)-1):
                tp_loss = self.cfg['sides_weights'][n] * tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.sides[n], pos_weight=self.cfg['pos_weights'])
                self.loss += tf.reduce_mean(tp_loss)
        self.loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.sides[-1], pos_weight=self.cfg['pos_weights']))

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if self.cfg['use_weight_regularizer']:
            self.loss = tf.add_n(reg_loss) + self.loss

        return self.loss  # 1.0*tf.shape(self.label)[0]

    def focal_loss(self):
        if self.cfg['is_deep_supervised']:
            for n in range(len(self.sides) - 1):
                sg_p = tf.nn.sigmoid(self.sides[n])
                sg_n = 1.0 - sg_p
                sg_p += 1e-5
                sg_n += 1e-5
                pos_num = tf.reduce_sum(tf.cast(self.label > 0.99, tf.float32))
                neg_num = tf.reduce_sum(tf.cast(self.label < 0.01, tf.float32))

                pos = -self.label*sg_n*sg_n*tf.log(sg_p)
                pos = tf.reduce_sum(pos) / (pos_num+1e-5)

                neg = -(1.0-self.label)*sg_p*sg_p*tf.log(sg_n)
                neg = tf.reduce_sum(neg) / (neg_num+1e-5)
                self.floss = self.floss + 0.25*pos + neg*0.75

        sg_p = tf.nn.sigmoid(self.sides[-1])
        sg_n = 1.0 - sg_p
        sg_p += 1e-5
        sg_n += 1e-5
        pos_num = tf.reduce_sum(tf.cast(self.label > 0.99, tf.float32))
        neg_num = tf.reduce_sum(tf.cast(self.label < 0.01, tf.float32))

        pos = -self.label * sg_n * sg_n * tf.log(sg_p)
        pos = tf.reduce_sum(pos) / (pos_num+1e-5)

        neg = -(1.0 - self.label) * sg_p * sg_p * tf.log(sg_n)
        neg = tf.reduce_sum(neg) / (neg_num+1e-5)
        self.floss = self.floss + 0.25*pos + neg*0.75
        if self.cfg['use_weight_regularizer']:
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.floss += tf.add_n(reg_loss)
        return self.floss

    def summary(self):
        tf.summary.scalar(name='loss_sm', tensor=self.loss)
        tf.summary.scalar(name='floss_sm', tensor=self.floss)
        max_outputs = 1
        tf.summary.image(name='label_sm', tensor=self.label, max_outputs=max_outputs, )


if __name__ == "__main__":
    with open('cfg.yml') as yaml_file:
        cfg = yaml.load(yaml_file)
    print(cfg)
    print(cfg['pos_weights'])

    a = tf.random_uniform((2, 2), 0, 5)
    # b = (a-0.5) * a

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print(sess.run(tf.reduce_sum(tf.cast(a>2, tf.float32))))
