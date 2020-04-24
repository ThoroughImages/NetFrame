import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

from model.deeplab_v3 import DeepLabV3
from utils.evaluation import Evaluation


class NetFrame(object):
    """Basic framework for constructing model pipeline.
    """

    def __init__(self, cfg, data, mode):
        self.cfg = cfg
        self.mode = mode
        self.data = data
        self.eval_obj = Evaluation()

        if mode != 'export':
            self.build_model()
            self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1000)

            
    def create_input(self):
        """ Datasets construction
        """
        if self.mode == 'train':
            self.training_dataset, self.train_size = self.data.build_dataset(self.cfg.train_path, mode='train')
            self.validation_dataset, self.valid_size = self.data.build_dataset(self.cfg.valid_path, mode='test')
            
            self.training_iterator = self.training_dataset.make_initializable_iterator()
            self.validation_iterator = self.validation_dataset.make_initializable_iterator()
            
            _types, _shapes = self.training_dataset.output_types, self.training_dataset.output_shapes
        else:
            self.test_dataset, self.test_size = self.data.build_dataset(self.cfg.test_path, mode='test', num_epoch=1)
            self.test_iterator = self.test_dataset.make_initializable_iterator()
            _types, _shapes = self.test_dataset.output_types, self.test_dataset.output_shapes

        print('Input type: {}, input shape: {}'.format(_types, _shapes))
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.handle, _types, _shapes)
        next_batch = iterator.get_next()

        return next_batch


    @staticmethod
    def average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
            over the devices. The inner list ranges over the different variables.
        Returns:
                List of pairs of (gradient, variable) where the gradient has been averaged
                across all towers.
        Ref: http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = [g for g, _ in grad_and_vars if g is not None]
            grad = tf.reduce_mean(grads, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    @property
    def snapshots_dir(self):
        return '{}/{}/{}'.format(self.cfg.model_dir, self.cfg.project, self.cfg.version)


    def model_path(self, iters):
        return self.snapshots_dir + '/model-{}.ckpt-{}'.format(iters, iters)
        # return self.snapshots_dir + '/model-{}.ckpt'.format(iters)

    
    def load(self, sess, model_path):
        restore_var = [v for v in tf.global_variables() if 'lr' not in v.name]
        loader = tf.train.Saver(var_list=restore_var)
        loader.restore(sess, model_path)
        print("Restored model parameters from: {}".format(model_path))

    @staticmethod
    def save(saver, sess, snapshots_dir, step):
        '''Save weights.

        Args:
            saver: TensorFlow Saver object.
            sess: TensorFlow session.
            snapshots_dir: Path to the snapshots directory.
            step: Current training step.
        '''
        model_name = 'model-{}.ckpt'.format(step)
        checkpoint_path = os.path.join(snapshots_dir, model_name)
        
        if not os.path.exists(snapshots_dir):
          os.makedirs(snapshots_dir)
        
        saver.save(sess, checkpoint_path, global_step=step)
        print('The checkpoint at step {} has been created.'.format(step))



