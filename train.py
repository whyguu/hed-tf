# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import yaml
from hed_net import HED
from loss import HedLoss
import os
import cv2
import argparse
import random


class DataSet(object):
    def __init__(self):
        with open('cfg.yml') as file:
            self.cfg = yaml.load(file)
        self.pointer = 0
        self.imgs = None
        self.labels = None
        self.samples_num = 0
        # read data
        self.read_data()

    def read_data(self):
        img_names = []
        label_names = []
        with open(self.cfg['file_name']) as file:
            while True:
                il = file.readline(1000)
                if not il:
                    break
                a = il.split(sep=' ')
                print(a)
                img_names.append(a[0])
                label_names.append(a[1][0:-1])  # remove '\n'
        self.samples_num = len(img_names)
        print('total image num: ', self.samples_num)
        self.imgs = np.zeros((len(img_names), self.cfg['height'], self.cfg['width'], self.cfg['channel']), np.float32)
        self.labels = np.zeros((len(img_names), self.cfg['height'], self.cfg['width'], 1), np.float32)
        for it in range(len(self.labels)):
            # read as bgr
            tp_img = cv2.imread(os.path.join(self.cfg['image_path'], img_names[it]))
            tp_label = cv2.imread(os.path.join(self.cfg['image_path'], label_names[it]), cv2.IMREAD_GRAYSCALE)
            self.imgs[it, :, :, :] = tp_img.astype(np.float32)
            self.labels[it, :, :, 0] = (tp_label/255).astype(np.float32)

        self.imgs -= self.cfg['mean']
        print('images and labels reading done!')

    def batch_iterator(self, shuffle=False):
        batch_size = self.cfg['batch_size']
        num_examples = len(self.imgs)
        idx = list(range(num_examples))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, num_examples, batch_size):
            imgs = self.imgs[idx[i:min(i+batch_size, num_examples)], :, :, :]
            labels = self.labels[idx[i:min(i+batch_size, num_examples)], :, :, :]
            print('batch_size: ', labels.shape[0])
            yield imgs, labels
        # if self.pointer == self.samples_num-1:
        #     self.pointer = 0
        # if self.pointer+batch_size > self.samples_num-1:
        #     imgs = self.imgs[self.pointer:, :, :, :]
        #     labels = self.labels[self.pointer:, :, :, :]
        #     self.pointer = 0
        # else:
        #     imgs = self.imgs[self.pointer:self.pointer+batch_size, :, :, :].copy()
        #     labels = self.labels[self.pointer:self.pointer+batch_size, :, :, :].copy()
        #     self.pointer += batch_size
        #
        # return imgs, labels


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='1')
    args = parser.parse_args()
    return args


def sess_config(args=None):
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用 GPU 0
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            gpu_options=gpu_options)

    return config


if __name__ == "__main__":
    with open('cfg.yml') as file:
        cfg = yaml.load(file)
    args = arg_parser()
    config = sess_config(args)

    # train data
    dataset = DataSet()
    # hed net
    hed_class = HED(height=cfg['height'], width=cfg['width'], channel=cfg['channel'])
    sides = hed_class.vgg_hed()
    # loss
    loss_class = HedLoss(sides)
    loss = loss_class.calc_loss()
    # optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=1e-5,
                                               global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.1,
                                               staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    # summary
    tf.summary.scalar(name='lr', tensor=learning_rate)
    hed_class.summary()
    loss_class.summary()
    merged_summary_op = tf.summary.merge_all()

    # train
    with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
        saver = tf.train.Saver()

        # sess.run(tf.global_variables_initializer())
        # # initialize with vgg16 weights trained on imagenet
        # hed_class.assign_init_weights(sess)
        #
        saver.restore(sess, cfg['model_weights_path']+'vgg16_hed-150')
        sess.run(tf.assign(global_step, 0))  # lr is relative to init lr and global_step, so initializing global_step makes lr initialized by your assignment

        summary_writer = tf.summary.FileWriter(cfg['log_dir'], graph=sess.graph, flush_secs=15)
        step = 0
        for epoch in range(1, cfg['max_epochs']+1):
            for imgs, labels in dataset.batch_iterator():
                merged_summary, _ = sess.run([merged_summary_op, train_op],
                                                feed_dict={hed_class.x: imgs, loss_class.label: labels})
                if not (step % 1):
                    summary_writer.add_summary(merged_summary, global_step=step)
                    print('save a merged summary !')
                step += 1

                print('global_step:', sess.run(global_step), 'epoch: ', epoch)

            if not epoch % cfg['snapshot_epochs']:
                saver.save(sess=sess, save_path=os.path.join(cfg['model_weights_path'], 'vgg16_hed'), global_step=epoch)
                print('save a snapshoot !')
        summary_writer.close()
        saver.save(sess=sess, save_path=os.path.join(cfg['model_weights_path'], 'vgg16_hed'), global_step=epoch)
        print('save final model')



