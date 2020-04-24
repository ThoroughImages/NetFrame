"""
    DeepLab v3 models based on slim library.
    Ref: https://github.com/rishizek/tensorflow-deeplab-v3/blob/master/deeplab_model.py
"""

"""DeepLab v3 models based on slim library."""

import tensorflow as tf

from model.resnet import resnet_v2_50, resnet_v2_101, resnet_arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
    """Atrous Spatial Pyramid Pooling.
    Args:
        inputs: A tensor of size [batch, height, width, channels].
        output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
        batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
        is_training: A boolean denoting whether the input is for training.
        depth: The depth of the ResNet unit output.
    Returns:
        The atrous spatial pyramid pooling output.
    """
    with tf.variable_scope("aspp"):
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')

        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2*rate for rate in atrous_rates]

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with arg_scope([layers.batch_norm], is_training=is_training):
                inputs_size = tf.shape(inputs)[1:3]
                # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                # the rates are doubled when output stride = 8.
                conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
                conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
                conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

                # (b) the image-level features
                with tf.variable_scope("image_level_features"):
                    # global average pooling
                    image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                    # 1x1 convolution with 256 filters( and batch normalization)
                    image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
                    # bilinearly upsample features
                    image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

                net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
                net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

                return net


def DeepLabV3(inputs,
              num_classes,
              is_training,
              output_stride=16,
              base_architecture='resnet_v2_50',
              batch_norm_decay=0.9997,
              reuse=None):
                  
    """Generator for DeepLab v3 plus models.

    Args:
        num_classes: The number of possible classes for image classification.
        output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
            the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
        base_architecture: The architecture of base Resnet building block.
        batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
    Returns:
        The model function that takes in `inputs` and `is_training` and
            returns the output tensor of the DeepLab v3 model.
    """
    if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
        raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'.")

    if base_architecture == 'resnet_v2_50':
        base_model = resnet_v2_50
    else:
        base_model = resnet_v2_101

    # encoder
    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        logits, end_points = base_model(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride,
                                      reuse=reuse)

    inputs_size = tf.shape(inputs)[1:3]
    net = end_points[base_architecture + '/block4']
    net = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

    with tf.variable_scope("upsampling_logits"):
        net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
        logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')

    return logits
