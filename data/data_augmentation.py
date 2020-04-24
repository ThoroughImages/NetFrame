import cv2
import numpy as np
import tensorflow as tf
from scipy.stats import halfnorm
from scipy.ndimage import rotate
from sklearn.utils import shuffle


def _random_gaussian_blur(img, kernel_size=5):
    choice = np.random.choice([0, 1], replace=True, p=[0.666, 0.334])
    kernel_size = int(halfnorm.rvs(loc=kernel_size, scale=kernel_size + 1, size=1)[0])
    kernel_size = kernel_size if kernel_size % 2 else kernel_size - 1
    
    if choice:
        output = cv2.blur(img, (kernel_size, kernel_size))
    
    else:
        output = img
        
    return output.clip(0.0, 255.0)


def _random_motion_blur(img, kernel_size=7):
    choice = np.random.choice([0, 1], replace=True, p=[0.666, 0.334])
    kernel_size = int(halfnorm.rvs(loc=kernel_size, scale=kernel_size + 1, size=1)[0])
    kernel_size = kernel_size if kernel_size % 2 else kernel_size - 1
        
    if choice:
        # generating the kernel
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size

        deg = np.random.uniform(0, 180)
        kernel_motion_blur_rotated = rotate(kernel_motion_blur, deg, reshape=False)
        kernel_motion_blur_rotated /= kernel_motion_blur_rotated.sum()

        # applying the kernel to the input image
        output = cv2.filter2D(img, -1, kernel_motion_blur_rotated)
    
    else:
        output = img
        
    return output.clip(0.0, 255.0)


def _image_blur(img):
    choice = np.random.choice([0, 1], replace=True, p=[0.5, 0.5])

    if choice:
        return _random_motion_blur(img)
    else:
        return _random_gaussian_blur(img)


def image_blur(img):
    return tf.py_func(_image_blur, [img], tf.float32)


def random_adjust_brightness(image, max_delta=0.2, seed=None):
    """Randomly adjusts brightness. """
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_brightness(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25, seed=None):
    """Randomly adjusts contrast. """
    contrast_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


# def random_adjust_hue(image, max_delta=0.02, seed=None):
#     """Randomly adjusts hue. """
#     delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
#     image = tf.image.adjust_hue(image / 255, delta) * 255
#     image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
#     return image


def random_adjust_hue(image, max_delta=0.02, seed=None):
    """Randomly adjusts hue. Only Red and Blue channel """
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    image_hue = tf.image.adjust_hue(image / 255, delta) * 255
    image_new = tf.stack([image_hue[:, :, 0], image[:, :, 1], image_hue[:, :, 2]], axis=2)
    image_new = tf.clip_by_value(image_new, clip_value_min=0.0, clip_value_max=255.0)
    return image_new


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25, seed=None):
    """Randomly adjusts saturation. """
    saturation_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def image_jittering(image):
    """Randomly distorts color.

    Randomly distorts color using a combination of brightness, hue, contrast and
    saturation changes. Makes sure the output image is still between 0 and 255.

    Args:
        image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
                     with pixel values varying between [0, 255].

    Returns:
        image: image which is the same shape as input image.
    """

    image = random_adjust_brightness(image, max_delta=0.2)
    image = random_adjust_saturation(image, min_delta=0.8, max_delta=1.2)
    image = random_adjust_hue(image, max_delta=0.08)
    image = random_adjust_contrast(image, min_delta=0.8, max_delta=1.2)

    return image
    
    
def image_scaling(img, label, patch_size, input_size):
    """
    Randomly scales the images between 0.5 to 1.25 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    rand = tf.random_uniform([2], 0, patch_size - input_size + 1)

    offset = tf.cast(tf.floor(rand), dtype=tf.int32)

    img = tf.image.crop_to_bounding_box(img, offset[0], offset[1], input_size, input_size)
    label = tf.image.crop_to_bounding_box(label, offset[0], offset[1], input_size, input_size)

    return img, label


def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    random_var1 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    img = tf.cond(pred=tf.equal(random_var1, 0),true_fn=lambda: tf.image.flip_left_right(img),false_fn=lambda: img)
    label = tf.cond(pred=tf.equal(random_var1, 0),true_fn=lambda: tf.image.flip_left_right(label),false_fn=lambda: label)

    random_var2 = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    img = tf.cond(pred=tf.equal(random_var2, 0),true_fn=lambda: tf.image.flip_up_down(img),false_fn=lambda: img)
    label = tf.cond(pred=tf.equal(random_var2, 0),true_fn=lambda: tf.image.flip_up_down(label),false_fn=lambda: label)
        
    return img, label


def augmentation(patch_size, input_size, random_scale, random_mirror, color_jitter, random_blur, size_clip, img, label, class_label):
    # Randomly mirror the images and labels.
    if size_clip is True and patch_size > input_size:
        img = tf.image.crop_to_bounding_box(img, patch_size // 4, patch_size // 4, input_size, input_size)
        label = tf.image.crop_to_bounding_box(label, patch_size // 4, patch_size // 4, input_size, input_size)

    # if random_scale:
    #     img, label = image_scaling(img, label, patch_size, input_size)

    if random_mirror:
        img, label = image_mirroring(img, label)
    
    if color_jitter:
        img = image_jittering(img)

    if random_blur:
        img = image_blur(img)
        img.set_shape([input_size, input_size, 3])

    return img, label


