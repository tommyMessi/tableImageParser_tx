# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
from shapely.geometry import Polygon
import random

import tensorflow as tf

from data_util import GeneratorEnqueuer

tf.app.flags.DEFINE_string('training_data_path', './data_tx/raw_img/',
                           'training dataset to use')


FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(FLAGS.training_data_path, '*.{}'.format(ext))))
    return files


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def resize_train(im, label_row, label_col, label_nrow, label_ncol):
    h, w, _ = im.shape

    if h<450:
        h_new = 512*h/w
        pad = random.randint(10,512 - int(h_new))

        im = cv2.copyMakeBorder(im, 0,pad,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
        label_row = cv2.copyMakeBorder(label_row, 0, pad, 0, 0, cv2.BORDER_CONSTANT,value=1)
        label_col = cv2.copyMakeBorder(label_col, 0, pad, 0, 0, cv2.BORDER_CONSTANT,value=1)
        label_nrow = cv2.copyMakeBorder(label_nrow, 0, pad, 0, 0, cv2.BORDER_CONSTANT,value=1)
        label_ncol = cv2.copyMakeBorder(label_ncol, 0, pad, 0, 0, cv2.BORDER_CONSTANT,value=1)


    size = (int(512), int(512))
    im_1 = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    label_row = cv2.resize(label_row, size, interpolation=cv2.INTER_AREA)
    label_col = cv2.resize(label_col, size, interpolation=cv2.INTER_AREA)
    label_nrow = cv2.resize(label_nrow, size, interpolation=cv2.INTER_AREA)
    label_ncol = cv2.resize(label_ncol, size, interpolation=cv2.INTER_AREA)

    return im_1,label_row,label_col,label_nrow,label_ncol


def crop_area(im, label_row,label_col, label_nrow, label_ncol):

    im_p,la_row,la_col,la_nrow,la_ncol = resize_train(im, label_row, label_col, label_nrow, label_ncol)

    return im_p, la_row, la_col, la_nrow, la_ncol


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle


def generator_label(label_im, label_str):
    label_name = label_str.split('/')[-1]
    h, w = label_im.shape
    score_map = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if label_im[i][j] == 0:
                score_map[i][j] = 0
            else:
                score_map[i][j] = 1

    return score_map

def generator(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=True):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(
        image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps_nrow = []

        score_maps_ncol = []

        score_maps_row = []

        score_maps_col = []
        training_masks = []
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                if '.png' in im_fn:
                    im_fn = im_fn.replace('.png','.jpg')

                h, w, _ = im.shape
                label_fn_nrow = im_fn.replace('raw', 'nrow')
                label_fn_ncol = im_fn.replace('raw', 'ncol')
                label_fn_row = im_fn.replace('raw', 'row')
                label_fn_col = im_fn.replace('raw', 'col')

                if not os.path.exists(label_fn_nrow):
                    print('text file {} does not exists'.format(label_fn_nrow))
                    continue
                if not os.path.exists(label_fn_ncol):
                    print('text file {} does not exists'.format(label_fn_ncol))
                    continue
                if not os.path.exists(label_fn_row):
                    print('text file {} does not exists'.format(label_fn_row))
                    continue
                if not os.path.exists(label_fn_col):
                    print('text file {} does not exists'.format(label_fn_col))
                    continue
                label_im_nrow = cv2.imread(label_fn_nrow, cv2.IMREAD_GRAYSCALE)
                label_im_ncol = cv2.imread(label_fn_ncol, cv2.IMREAD_GRAYSCALE)
                label_im_row = cv2.imread(label_fn_row, cv2.IMREAD_GRAYSCALE)
                label_im_col = cv2.imread(label_fn_col, cv2.IMREAD_GRAYSCALE)

                score_map_nrow = generator_label(label_im_nrow, label_fn_nrow)
                score_map_ncol = generator_label(label_im_ncol, label_fn_ncol)
                score_map_row = generator_label(label_im_row, label_fn_row)
                score_map_col = generator_label(label_im_col, label_fn_col)

                im, score_map_row,score_map_col,score_map_nrow,score_map_ncol = crop_area(im, score_map_row, score_map_col,score_map_nrow,score_map_ncol)

                training_mask = np.ones((input_size, input_size), dtype=np.uint8)

                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)

                score_maps_nrow.append(score_map_nrow[::2, ::2, np.newaxis].astype(np.float32))

                score_maps_ncol.append(score_map_ncol[::2, ::2, np.newaxis].astype(np.float32))

                score_maps_row.append(score_map_row[::2, ::2, np.newaxis].astype(np.float32))

                score_maps_col.append(score_map_col[::2, ::2, np.newaxis].astype(np.float32))

                training_masks.append(training_mask[::2, ::2, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, image_fns, score_maps_nrow, score_maps_ncol, \
                          score_maps_row, score_maps_col, training_masks
                    images = []
                    image_fns = []
                    score_maps_nrow = []

                    score_maps_ncol = []

                    score_maps_row = []

                    score_maps_col = []

                    training_masks = []
            except Exception as e:
                import traceback
                print(im_fn)
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



if __name__ == '__main__':
    pass
