# -*- coding: UTF-8 -*-
import tensorflow as tf
import yaml
import numpy as np


class HedLoss(object):
    def __init__(self, sides):
        self.sides = sides
        self.loss = 0.0
        with open('cfg.yml') as file:
            self.cfg = yaml.load(file)
        self.label = tf.placeholder(tf.float32, (None, self.cfg['height'], self.cfg['width'], 1))
        # self.calc_loss()

    def calc_loss(self):
        if self.cfg['is_deep_supervised']:
            for n in range(len(self.sides)-1):
                tp_loss = self.cfg['sides_weights'][n] * tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.sides[n], pos_weight=self.cfg['pos_weights'])
                self.loss = tf.reduce_mean(tp_loss)
        self.loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.sides[-1], pos_weight=self.cfg['pos_weights']))

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        use_weight_regularizer = self.cfg['use_weight_regularizer']
        if use_weight_regularizer:
            self.loss = tf.add_n(reg_loss) + self.loss

        return self.loss  # 1.0*tf.shape(self.label)[0]

    def summary(self):
        tf.summary.scalar(name='loss_sm', tensor=self.loss)
        max_outputs = 1
        tf.summary.image(name='label_sm', tensor=self.label, max_outputs=max_outputs, )


if __name__ == "__main__":
    with open('cfg.yml') as yaml_file:
        cfg = yaml.load(yaml_file)
    print(cfg)
    print(cfg['pos_weights'])
