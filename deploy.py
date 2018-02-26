# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import yaml
from hed_net import HED
from loss import HedLoss
import os
import cv2
import argparse
import gc


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='0')
    parser.add_argument('-img_path', type=str, required=True, default=None)
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


def img_pre_process(img, **kwargs):
    img -= kwargs['mean']
    return img


def predict_big_map(img_path, out_shape=(448, 448), inner_shape=(224, 224), out_channel=1, pred_fun=None, **kwargs):
    """
    :param img_path: big image path
    :param out_shape: (height, width)
    :param inner_shape: (height, width)
    :param out_channel: predicted results' channel num
    :param pred_fun: forward model
    :return: predicted image
    """
    image = cv2.imread(img_path, )
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        gc.collect()
    pd_up_h, pd_lf_w = np.int64((np.array(out_shape)-np.array(inner_shape)) / 2)

    print(image.shape)
    ori_shape = image.shape
    pd_bm_h = (out_shape[0]-pd_up_h) - (image.shape[0] % inner_shape[0])
    pd_rt_w = (out_shape[1]-pd_lf_w) - (image.shape[1] % inner_shape[1])

    it_h = np.int64(np.ceil(image.shape[0] / inner_shape[0]))
    it_w = np.int64(np.ceil(image.shape[1] / inner_shape[1]))

    image = np.pad(image, ((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w), (0, 0)), mode='reflect').astype(np.float32)  # the image is default a color one
    print(image.shape)
    print((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w))
    gc.collect()

    tp1 = np.array(inner_shape[0] - ori_shape[0] % inner_shape[0])
    tp2 = np.array(inner_shape[1] - ori_shape[1] % inner_shape[1])
    out_img = np.zeros((ori_shape[0]+tp1, ori_shape[1]+tp2, out_channel), np.float32)

    for ith in range(0, it_h):
        h_start = ith * inner_shape[0]
        count = 1
        for itw in range(0, it_w):
            w_start = itw*inner_shape[1]
            tp_img = image[h_start:h_start+out_shape[0], w_start:w_start+out_shape[1], :]

            tp_img = img_pre_process(tp_img.copy(), **kwargs)
            # print('tp_img', tp_img.shape)

            tp_out = pred_fun(tp_img[np.newaxis, :])
            tp_out = np.squeeze(tp_out, axis=0)
            out_img[h_start:h_start+inner_shape[0], w_start:w_start+inner_shape[1], :] = tp_out[pd_up_h:pd_up_h+inner_shape[0], pd_lf_w:pd_lf_w+inner_shape[1], :]

            print('hahahaha!', h_start, w_start, count)
            count += 1
    return out_img


if __name__ == '__main__':
    args = arg_parser()
    config = sess_config(args)
    with open('cfg.yml') as file:
        cfg = yaml.load(file)
    path = args.img_path

    ipt_img = cv2.imread(path, )
    height = cfg['height']
    width = cfg['width']
    channel = cfg['channel']
    mean = cfg['mean']

    hed_class = HED(height=height, width=width, channel=channel)
    hed_class.vgg_hed()
    sides = [tf.sigmoid(hed_class.side1),
             tf.sigmoid(hed_class.side2),
             tf.sigmoid(hed_class.side3),
             tf.sigmoid(hed_class.side4),
             tf.sigmoid(hed_class.side5),
             tf.sigmoid(hed_class.fused_side)]
    sides = tf.add_n(sides) / len(sides)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    # load weights
    saver.restore(sess, cfg['model_weights_path'] + 'vgg16_hed-88')

    output_img = predict_big_map(img_path=path, out_shape=(448, 448), inner_shape=(224, 224), out_channel=1,
                                 pred_fun=(lambda ipt: sess.run(sides, feed_dict={hed_class.x: ipt})), mean=cfg['mean'])
    output_img = np.squeeze((output_img*255).astype(np.uint8))
    cv2.imwrite('gray_img.png', output_img)
    cv2.imwrite('black_img.png', 255*(output_img > 127))
    sess.close()


# if __name__ == '__main__':
#     args = arg_parser()
#     config = sess_config(args)
#     with open('cfg.yml') as file:
#         cfg = yaml.load(file)
#     path = args.img_path
#
#     ipt_img = cv2.imread(path, )
#     height = cfg['height']
#     width = cfg['width']
#     channel = cfg['channel']
#
#     ori_shape = ipt_img.shape
#     print(ori_shape)
#     pd_h = ipt_img.shape[0] % height
#     pd_w = ipt_img.shape[1] % width
#
#     if pd_h != 0:
#         pd_h = height - pd_h
#     if pd_w != 0:
#         pd_w = width - pd_w
#
#     pd_img = np.pad(ipt_img, ((0, pd_h), (0, pd_w), (0, 0)), mode='reflect').astype(np.float32)
#     pd_img -= cfg['mean']
#
#     ipt_img = None
#     gc.collect()
#     out_img = np.zeros(pd_img.shape[0:2], np.uint8)
#
#     hed_class = HED(height=height, width=width, channel=channel)
#     hed_class.vgg_hed()
#
#     sides = [tf.sigmoid(hed_class.side1),
#              tf.sigmoid(hed_class.side2),
#              tf.sigmoid(hed_class.side3),
#              tf.sigmoid(hed_class.side4),
#              tf.sigmoid(hed_class.side5),
#              tf.sigmoid(hed_class.fused_side)]
#     sides = tf.add_n(sides) / len(sides)
#     sess = tf.Session(config=config)
#     saver = tf.train.Saver()
#     # load weights
#     saver.restore(sess, cfg['model_weights_path'] + 'mw_bak/vgg16_hed-150')
#     for ith in range(0, pd_img.shape[0], height):
#         for itw in range(0, pd_img.shape[1], width):
#             tp_img = pd_img[ith:ith+height, itw:itw+width, :]
#             tp_img = np.expand_dims(tp_img, axis=0)
#             np_sides = sess.run(sides, feed_dict={hed_class.x: tp_img.astype(np.float32)})
#
#             tp_img = np.squeeze(np_sides,)
#             out_img[ith:ith+height, itw:itw+width] = (tp_img*255).astype(np.uint8)
#
#             print('hahahaha!', ith, itw)
#
#     cv2.imwrite('gray_img.png', out_img)
#     cv2.imwrite('black_img.png', 255*(out_img > 127))
#     sess.close()
