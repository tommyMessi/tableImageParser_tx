# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
from shapely.geometry import Polygon

import tensorflow as tf

from data_util import GeneratorEnqueuer

tf.app.flags.DEFINE_string('training_data_path', './tx_data/image',
                           'training dataset to use')


FLAGS = tf.app.flags.FLAGS


def get_images():
    """
    Returns a list of images.

    Args:
    """
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



def crop_area(im, label_im,crop_background=False, max_tries=150):
    """
    Crops an image to a specified size.

    Args:
        im: (int): write your description
        label_im: (str): write your description
        crop_background: (todo): write your description
        max_tries: (int): write your description
    """
    size = (int(512), int(512))
    im_p = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    la_p = cv2.resize(label_im, size, interpolation=cv2.INTER_AREA)
    return im_p,la_p


def point_dist_to_line(p1, p2, p3):
    """
    Calculate the distance between two points.

    Args:
        p1: (todo): write your description
        p2: (todo): write your description
        p3: (todo): write your description
    """
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    """
    Fit line to line

    Args:
        p1: (array): write your description
        p2: (array): write your description
    """
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    """
    Compute the line between two line points.

    Args:
        line1: (array): write your description
        line2: (array): write your description
    """
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
    """
    Adds a line to a vertical vertical circle.

    Args:
        line: (str): write your description
        point: (int): write your description
    """
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
    """
    Generate label label from label_im.

    Args:
        label_im: (str): write your description
        label_str: (str): write your description
    """
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
    """
    Generate image generator.

    Args:
        input_size: (int): write your description
        batch_size: (int): write your description
        background_ratio: (todo): write your description
        random_scale: (int): write your description
        np: (todo): write your description
        array: (array): write your description
        vis: (todo): write your description
    """
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
                label_fn_nrow = im_fn.replace('image', 'label_nrow')
                label_fn_ncol = im_fn.replace('image', 'label_ncol')
                label_fn_row = im_fn.replace('image', 'label_row')
                label_fn_col = im_fn.replace('image', 'label_col')

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

                im, score_map_nrow = crop_area(im, score_map_nrow, crop_background=True)
                im, score_map_ncol = crop_area(im, score_map_ncol, crop_background=True)
                im, score_map_row = crop_area(im, score_map_row, crop_background=True)
                im, score_map_col = crop_area(im, score_map_col, crop_background=True)
                im = cv2.resize(im, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)

                score_map_nrow = cv2.resize(score_map_nrow, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
                score_map_ncol = cv2.resize(score_map_ncol, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
                score_map_row = cv2.resize(score_map_row, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
                score_map_col = cv2.resize(score_map_col, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)

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
    """
    Iterate batches of the given number of workers.

    Args:
        num_workers: (int): write your description
    """
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
