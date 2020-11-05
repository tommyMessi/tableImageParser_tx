import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    """
    Unpool a list.

    Args:
        inputs: (array): write your description
    """
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score_nrow = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

            F_score_ncol = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

            F_score_row = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

            F_score_col = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

    return F_score_nrow, F_score_ncol, F_score_row, F_score_col


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls*training_mask) + tf.reduce_sum(y_pred_cls*training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss

def focal_loss(y_true_cls, y_pred_cls):
    '''
    :param y_true_cls:
    :param y_pred_cls:
    :return:
    '''
    gamma = 2
    alpha = 0.5

    dim = tf.reduce_prod(tf.shape(y_true_cls)[1:])
    flat_y_true_cls = tf.reshape(y_true_cls, [-1, dim])
    flat_y_pred_cls = tf.reshape(y_pred_cls, [-1, dim])
    pt = flat_y_true_cls*flat_y_pred_cls+(1.0-flat_y_true_cls)*(1.0 - flat_y_pred_cls)
    weight_map = alpha*tf.pow((1.0-pt),gamma)
    weighted_loss = tf.multiply(((flat_y_true_cls * tf.log(flat_y_pred_cls + 1e-9)) + ((1 - flat_y_true_cls) * tf.log(1 - flat_y_pred_cls + 1e-9))), weight_map)
    cross_entropy = -tf.reduce_sum(weighted_loss,axis = 1)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('classification_focal_loss', cross_entropy_mean)
    return cross_entropy_mean


def loss(y_true_cls_nrow, y_pred_cls_nrow,
            y_true_cls_ncol, y_pred_cls_ncol,
            y_true_cls_row, y_pred_cls_row,
            y_true_cls_col, y_pred_cls_col,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss_nrow = dice_coefficient(y_true_cls_nrow, y_pred_cls_nrow, training_mask)
    classification_loss_ncol = dice_coefficient(y_true_cls_ncol, y_pred_cls_ncol, training_mask)
    classification_loss_row = dice_coefficient(y_true_cls_row, y_pred_cls_row, training_mask)
    classification_loss_col = dice_coefficient(y_true_cls_col, y_pred_cls_col, training_mask)


    return tf.reduce_mean(classification_loss_row+classification_loss_ncol+classification_loss_nrow+classification_loss_col)
