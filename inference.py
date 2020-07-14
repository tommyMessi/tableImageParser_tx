import cv2
import numpy as np
import tensorflow as tf

import logging
log = logging.getLogger(__name__)

import model
import time

import os
import random

class Detector(object):
    def __init__(self,model_dir):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        config.gpu_options.allow_growth = True
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        self.session = tf.Session(config=config)
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.score_nrow, self.score_ncol, self.score_row, self.score_col = model_tx.model(self.input_images, is_training=False)
        self.variable_averages = tf.train.ExponentialMovingAverage(0.997, self.global_step)
        self.saver = tf.train.Saver(self.variable_averages.variables_to_restore())
        self.ckpt_state = tf.train.get_checkpoint_state(model_dir)
        print(self.ckpt_state)
        self.model_path = os.path.join(model_dir, os.path.basename(self.ckpt_state.model_checkpoint_path))
        print(self.model_path)
        self.saver.restore(self.session,self.model_path)


    def main_detection(self, image):
        # img_e_c = image[:,:,::-1]
        img_e = np.expand_dims(image, axis=2)
        img_e_c = np.concatenate((img_e, img_e, img_e), axis=-1)
        im_resized, (ratio_h, ratio_w) = resize_image(img_e_c)
        score_nrow, score_ncol, score_row, score_col = self.session.run([self.score_nrow, self.score_ncol, self.score_row, self.score_col], feed_dict={self.input_images: [im_resized]})
        return score_nrow[0], score_ncol[0], score_row[0], score_col[0] ,ratio_h, ratio_w

def resize_image(im):
    h, w, _ = im.shape
    size = (int(512), int(512))
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    # la_p = cv2.resize(label_im, size, interpolation=cv2.INTER_AREA)

    ratio_h = 512 / float(h)
    ratio_w = 512 / float(w)

    return im, (ratio_h, ratio_w)

def iou_count(list1, list2):
    xx1 = np.maximum(list1[0], list2[0])
    yy1 = np.maximum(list1[1], list2[1])
    xx2 = np.minimum(list1[4], list2[4])
    yy2 = np.minimum(list1[5], list2[5])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)

    inter = w * h
    area1 = (list1[4] - list1[0] + 1) * (list1[5] - list1[1] + 1)
    area2 = (list2[4] - list2[0] + 1) * (list2[5] - list2[1] + 1)
    iou = inter / min(area1, area2)
    return iou

if __name__ == '__main__':

    result_path = './result/'
    instance = Detector('./model/')
    images = os.listdir('./image/')

    row_root = './tx_infer_data/row'
    col_root = './tx_infer_data/col'
    nrow_root = './tx_infer_data/nrow'
    ncol_root = './tx_infer_data/ncol'


    i_l = []
    for x in range(len(images)):
        print(images[x])
        image_path = os.path.join('./image/',images[x])
        image_name = images[x]
        txt_name = image_name.replace('.jpg','.txt')
        row_path = os.path.join(row_root, image_name)
        col_path = os.path.join(col_root, image_name)
        nrow_path = os.path.join(nrow_root, image_name)
        ncol_path = os.path.join(ncol_root, image_name)

        image = cv2.imread(image_path, 0)
        # image = cv2.imread(image_path)
        image_color = cv2.imread(image_path)
        # instance.table_detection(image, image_color)
        score_nrow, score_ncol, score_row, score_col, ratio_h, ratio_w = instance.main_detection(image)

        score_nrow = np.where(score_nrow > 0.9, score_nrow, 0)
        score_nrow = np.where(score_nrow < 0.9, score_nrow, 1)

        score_ncol = np.where(score_ncol > 0.9, score_ncol, 0)
        score_ncol = np.where(score_ncol < 0.9, score_ncol, 1)

        score_row = np.where(score_row > 0.9, score_row, 0)
        score_row = np.where(score_row < 0.9, score_row, 1)

        score_col = np.where(score_col > 0.9, score_col, 0)
        score_col = np.where(score_col < 0.9, score_col, 1)

        nmap = cv2.bitwise_and(score_nrow, score_ncol)
        lmap = cv2.bitwise_and(score_row, score_col)
        pre_map = cv2.bitwise_and(nmap, lmap)

        result = os.path.join(result_path, images[x])
        score_nrow_map = cv2.resize(score_nrow, dsize=None, fx=1/ratio_w, fy=1/ratio_h, interpolation=cv2.INTER_AREA)
        score_ncol_map = cv2.resize(score_ncol, dsize=None, fx=1 / ratio_w, fy=1 / ratio_h, interpolation=cv2.INTER_AREA)
        score_row_map = cv2.resize(score_row, dsize=None, fx=1 / ratio_w, fy=1 / ratio_h, interpolation=cv2.INTER_AREA)
        score_col_map = cv2.resize(score_col, dsize=None, fx=1 / ratio_w, fy=1 / ratio_h, interpolation=cv2.INTER_AREA)
        pre_map = cv2.resize(pre_map, dsize=None, fx=1 / ratio_w, fy=1 / ratio_h, interpolation=cv2.INTER_AREA)
        # mask_result = os.path.join(result_path, 'mask_'+images[x])
        # print(mask_result)
        cv2.imwrite(row_path, score_row_map*255)
        cv2.imwrite(col_path, score_col_map*255)
        cv2.imwrite(nrow_path, score_nrow_map * 255)
        cv2.imwrite(ncol_path, score_ncol_map * 255)
        cv2.imwrite(result, pre_map*255)
