import random
from functools import partial

import cv2
import tensorflow as tf

from data.data_augmentation import augmentation


class DataDealer(object):
    """Class that makes loading data and data augmentation easy.

    Public methods:
    - get_batch_from_queue: gets data batch tuple using ImageReader
    - data_loader: loads data from a file with data locations
    - get_img_input_from_img_batch: gets image input from data batch (output of get_batch_from_queue)
    - get_labels_seg_from_label_batch: gets label segment from data batch (output of get_batch_from_queue)

    """

    def __init__(self, cfg, mode):
        # Data preparation
        self.cfg = cfg
        self.mode = mode

    def _read_list(self, record):
        image_path, label_path, class_label = tf.decode_csv(
            record,
            record_defaults=[[""],[""],[0]],
            field_delim=' ',
        )
        return image_path, label_path, class_label, record

    def _read_images_from_disk(self, image_path, label_path, class_label, sample_name):
        """Reads image and mask from disk, decodes it into dense tensors, 
        and resizes it to the fixed shape.
        Args:
            image_path: path to the image.
            label_path: path to the label mask.
            class_label: class label of type integer.
            sample_name: name of the sample.

        Returns:
            Tuple of image_content, image, label, class_label
        """
        try:
            image_content = tf.read_file(image_path)
            image = tf.image.decode_png(image_content, channels=3)
            # image = tf.image.decode_jpeg(image_content, channels=3)
            image = tf.image.resize_images(image, [self.cfg.patch_size, self.cfg.patch_size])
            image = tf.cast(image, dtype=tf.float32)

            label_content = tf.read_file(label_path)
            label = tf.image.decode_png(label_content, channels=1)
            label = tf.image.resize_images(label, [self.cfg.patch_size, self.cfg.patch_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            label = tf.cast(label, dtype=tf.int32)
        except:
            image = tf.zeros(shape=[self.cfg.patch_size, self.cfg.patch_size, 3], dtype=tf.float32)
            label = tf.zeros(shape=[self.cfg.patch_size, self.cfg.patch_size, 1], dtype=tf.int32)

        class_label = tf.cast(class_label, dtype=tf.int32)
        sample_name = tf.convert_to_tensor(sample_name, dtype=tf.string)
        
        return image, label, class_label, sample_name

    def _get_augmentation_func(self, mode):
        if mode == 'train':
            random_scale, random_mirror, color_jitter, random_blur, size_clip = self.cfg.random_scale, self.cfg.random_mirror, self.cfg.color_jitter, self.cfg.random_blur, False
        else:
            random_scale, random_mirror, color_jitter, random_blur, size_clip = False, False, False, False, True

        augmentation_with_params = partial(augmentation, self.cfg.patch_size, self.cfg.input_size, 
                                        random_scale, random_mirror, color_jitter, random_blur, size_clip)

        def augmentation_func(image, label, class_label, sample_name):
            image, label = augmentation_with_params(image, label, class_label)
            return image, label, class_label, sample_name
        
        return augmentation_func

    def _preprocessing(self, dataset, mode):
        """Implement pre-processing methods."""
        # Apply data augmentation
        augmentation_func = self._get_augmentation_func(mode=mode)
        dataset = dataset.map(augmentation_func, num_parallel_calls=12)
        return dataset

    def build_dataset(self, data_list_file, mode='train', batch_size=None, num_epoch=1000):
        """Create Tensorflow Dataset instance. 
        Args:
            data_list_file: file of input data list
            mode: train or test

        Return:
            Tensorflow Dataset instance for input
        """
        # filename = tf.placeholder(tf.string, shape=[])

        dataset = tf.data.TextLineDataset(data_list_file)
        dataset = dataset.map(self._read_list, num_parallel_calls=8)
        dataset = dataset.map(self._read_images_from_disk, num_parallel_calls=18)

        dataset = self._preprocessing(dataset, mode)
        dataset = dataset.shuffle(buffer_size=30)
        
        batch_size = self.cfg.batch_size if batch_size is None else batch_size
        # dataset = dataset.batch(batch_size=batch_size * len(self.cfg.gpu))
        dataset = dataset.batch(batch_size=batch_size * len(self.cfg.gpu), drop_remainder=True)
        dataset = dataset.prefetch(10)
        dataset = dataset.repeat(num_epoch)

        ds_size = 0
        for _ in enumerate(open(data_list_file, 'r')):
            ds_size += 1

        iters_per_epoch = ds_size // (batch_size * len(self.cfg.gpu))
        print('Building dataset from {}, with {} records and {} iters each epoch'.format(
            data_list_file, ds_size, iters_per_epoch))

        return dataset, iters_per_epoch
