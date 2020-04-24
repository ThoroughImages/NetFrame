import os
import time

import numpy as np
import tensorflow as tf
from collections import defaultdict

from model.deeplab_v3 import DeepLabV3
from model.net_frame import NetFrame


class SegModel(NetFrame):
    """Segmentation network base on DeepLab v3
    """

    def __init__(self, cfg, data, mode):
        super(SegModel, self).__init__(cfg, data, mode)

    def _eval_per_sample(self, prob_flatten, cross_entropy_flatten, label_flatten):
        # Calculate the loss of each sample for hard negative mining
        loss_per_pixel = cross_entropy_flatten
        loss_per_pixel_zeros = tf.zeros_like(loss_per_pixel)
        loss_per_pixel_valid = tf.where(
            tf.less_equal(label_flatten, self.cfg.num_classes - 1),
            loss_per_pixel, 
            loss_per_pixel_zeros
        )
        loss_per_pixel_reshape = tf.reshape(loss_per_pixel_valid, [self.cfg.batch_size, self.cfg.input_size * self.cfg.input_size])
        loss_per_sample = tf.reduce_mean(tf.cast(loss_per_pixel_reshape, tf.float32), axis=1)
        probs_per_pixel_reshape = tf.reshape(prob_flatten[:, :], [self.cfg.batch_size, self.cfg.input_size * self.cfg.input_size, self.cfg.num_classes])
        probs_per_sample = tf.reduce_mean(tf.cast(probs_per_pixel_reshape, tf.float32), axis=1)

        return loss_per_sample, probs_per_sample
    
    def build_model(self):
        # Create network.
        with tf.device("/cpu:0"):
            with tf.name_scope("create_inputs"):
                image_batch, label_batch, self.class_label_batch, self.name_batch = self.create_input()

        self.lr_tf = tf.Variable(self.cfg.learning_rate, name='lr', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_tf)

        gpu_num = len(self.cfg.gpu)
        tower_dict = defaultdict(lambda: [None] * gpu_num)

        for i in range(gpu_num):
            with tf.device("/gpu:%d" % i):
                re_use = False if i == 0 else True
                with tf.variable_scope(tf.get_variable_scope(), reuse=re_use):
                    image_input = image_batch[i * self.cfg.batch_size : (i + 1) * self.cfg.batch_size]
                    image_input_general = tf.placeholder_with_default(image_input, shape=[self.cfg.batch_size, self.cfg.input_size, self.cfg.input_size, 3])

                    label_seg = label_batch[i * self.cfg.batch_size : (i + 1) * self.cfg.batch_size]
                    # assert image_input.get_shape() == [self.cfg.batch_size, self.cfg.input_size, self.cfg.input_size, 3]
                    self.is_train = tf.placeholder_with_default(True, shape=[])

                    self.logit = DeepLabV3(inputs=image_input_general,
                                     num_classes=self.cfg.num_classes,
                                     is_training=self.is_train,
                                     output_stride=16,
                                     base_architecture='resnet_v2_{}'.format(self.cfg.resnet_layer),
                                     batch_norm_decay=0.9,
                                     reuse=re_use)

                    self.logit = tf.reshape(self.logit, [self.cfg.batch_size, self.cfg.input_size, self.cfg.input_size, self.cfg.num_classes])
                    assert self.logit.get_shape() == [self.cfg.batch_size, self.cfg.input_size, self.cfg.input_size, self.cfg.num_classes]

                    # Flatten the logits and labels for calculating loss
                    logit_flatten = tf.reshape(self.logit, [-1, self.cfg.num_classes])
                    label_flatten = tf.reshape(label_seg, [-1,])

                    # Ignore 255-valued pixels in label (ground truth)
                    valid_indices = tf.squeeze(tf.where(tf.less_equal(label_flatten, self.cfg.num_classes - 1)), 1)

                    label_flatten_clipped = tf.clip_by_value(label_flatten, 0, self.cfg.num_classes - 1)
                    cross_entropy_flatten = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_flatten, labels=label_flatten_clipped)

                    if hasattr(self.cfg, 'class_weight'):
                        weight_flatten = tf.gather(self.cfg.class_weight, label_flatten_clipped)
                        cross_entropy_flatten = cross_entropy_flatten * weight_flatten
                    cross_entropy = tf.reduce_mean(tf.gather(cross_entropy_flatten, valid_indices))

                    all_trainable = [v for v in tf.trainable_variables() if not 'postnorm' in v.name]
                    l2_loss = self.cfg.l2_loss_lambda * tf.reduce_sum([
                                tf.nn.l2_loss(v) for v in all_trainable if not "bias" in v.name.lower()])

                    loss = cross_entropy + l2_loss
                    gradient = optimizer.compute_gradients(loss, var_list=all_trainable)

                    # Logging metrics
                    self.prob_flatten = tf.nn.softmax(logit_flatten) # shape [-1, num_classes]
                    predict_flatten = tf.cast(tf.argmax(self.prob_flatten, 1), tf.int32)
                    # Convert to binary
                    positive_flatten = tf.ones_like(label_flatten)
                    negative_flatten = tf.zeros_like(label_flatten)
                    predict_flatten_binary = tf.where(tf.equal(predict_flatten, self.cfg.log_label), positive_flatten, negative_flatten)
                    label_flatten_binary = tf.where(tf.equal(label_flatten, self.cfg.log_label), positive_flatten, negative_flatten)
                    # Calculate the confusion metrix
                    tp = tf.count_nonzero(tf.gather(predict_flatten_binary * label_flatten_binary, valid_indices))
                    tn = tf.count_nonzero(tf.gather((predict_flatten_binary - 1) * (label_flatten_binary - 1), valid_indices))
                    fp = tf.count_nonzero(tf.gather(predict_flatten_binary * (label_flatten_binary - 1), valid_indices))
                    fn = tf.count_nonzero(tf.gather((predict_flatten_binary - 1) * label_flatten_binary, valid_indices))
                    
                    loss_per_sample, probs_per_sample = self._eval_per_sample(self.prob_flatten, cross_entropy_flatten, label_flatten)

                    # Added tensors to tower dictionary
                    tower_dict['gradient'][i] = gradient
                    tower_dict['loss'][i] = loss
                    tower_dict['l2_loss'][i] = l2_loss
                    tower_dict['tp'][i] = tp
                    tower_dict['tn'][i] = tn
                    tower_dict['fp'][i] = fp
                    tower_dict['fn'][i] = fn
                    tower_dict['loss_per_sample'][i] = loss_per_sample
                    tower_dict['probs_per_sample'][i] = probs_per_sample

        with tf.device("/cpu:0"):
            grads = self.average_gradients(tower_dict['gradient'])
            # grads = [(tf.clip_by_norm(grad, 0.2), var) for grad, var in grads]
            grads = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in grads]

            self.train_op = optimizer.apply_gradients(grads)
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # Apply reduce_mean to the values in tower_dict
            self.reduced_tower_key = ['loss', 'l2_loss', 'tp', 'tn', 'fp', 'fn']
            self.reduced_dict = {
                'loss': tf.reduce_mean(tower_dict['loss']),
                'l2_loss': tf.reduce_mean(tower_dict['l2_loss']),
                'tp': tf.reduce_sum(tower_dict['tp']),
                'tn': tf.reduce_sum(tower_dict['tn']),
                'fp': tf.reduce_sum(tower_dict['fp']),
                'fn': tf.reduce_sum(tower_dict['fn']),
            }            
            self.loss_per_sample = tf.reshape(tower_dict['loss_per_sample'], (-1, ))
            self.probs_per_sample = tf.reshape(tower_dict['probs_per_sample'], (-1, self.cfg.num_classes))

    def _test_dataset(self, sess, ds_handle, num_iters, step, ds_name=''):
        """Test on the whole dataset
        Args:
            sess: Tensorflow Session
            ds_handle: Handle of dataset for input
            num_iters: Number of iterations of one epoch
            step: Current training step
            ds_name: name of the dataset
        
        Return:
            name_res_label: List of tuple (sample_name, sample_eval_result, class_label)
            reduced_dict_list: List of reduced dictionary
        """
        name_label_loss_probs = []
        reduced_dict_list = []
        
        for i in range(num_iters):
            start_time = time.time()
            try:
                names, losses, probs, class_labels, reduced_dict = sess.run([
                    self.name_batch, self.loss_per_sample, self.probs_per_sample, self.class_label_batch, self.reduced_dict], feed_dict={
                        self.handle: ds_handle})
            except tf.errors.OutOfRangeError:
                print('Reach the end of dataset {}'.format(ds_name))
                break
            else:
                name_label_loss_probs += zip(names, class_labels, losses, probs)
                mean_loss = np.mean([x[1] for x in name_label_loss_probs])
                reduced_dict_list.append(reduced_dict)
                print('Test on {} ({}/{}): loss = {:.3f} ({:.2f} sec/step)'.format(ds_name, i+1, num_iters, mean_loss, time.time() - start_time))

        loss, precision, recall, acc = self.eval_obj.whole_dataset_metrics(reduced_dict_list)
        print('Version: {} Test iters {} on dataset {}, loss = {:.3f}, acc = {:.3f}, recall = {:.3f}, precision = {:.3f} ({:.2f} seconds)'.format(
            self.cfg.version, step, ds_name, loss, acc, recall, precision, time.time() - start_time))
        
        return name_label_loss_probs


    def train(self, sess):
        # Create dataset handle, for dataset selection.
        training_handle = sess.run(self.training_iterator.string_handle())
        validation_handle = sess.run(self.validation_iterator.string_handle())

        # Initialize validation dataset, only execute once.
        sess.run(self.training_iterator.initializer)
        sess.run(self.validation_iterator.initializer)

        step = self.cfg.restore_iters + 1

        while True:
            try:
                start_time = time.time()
                _, _, reduced_dict = sess.run([
                    self.train_op, self.update_ops, self.reduced_dict], feed_dict={
                        self.handle: training_handle, self.is_train: True})

                loss, precision, recall, acc = self.eval_obj.training_metrics(reduced_dict)
                print('step: {:d}, loss = {:.3f}, acc = {:.3f}, recall = {:.3f}, precision = {:.3f} ({:.2f} sec/step)'.format(
                    step, loss, acc, recall, precision, time.time() - start_time))
                # Learning rate decay
                if step % self.cfg.lr_decay_step == 0:
                    sess.run(tf.assign(self.lr_tf, self.lr_tf / 2))
                    print('LEARNING_RATE:', sess.run(self.lr_tf))

                # Test on validation set and save the snapshots
                if step % self.cfg.save_step == 0:
                    # Test on Validation set
                    self.save(self.saver, sess, self.snapshots_dir, step)
                    _ = self._test_dataset(sess, validation_handle, num_iters=self.valid_size, step=step, ds_name='validation')

                step += 1

            except tf.errors.OutOfRangeError:
                print('Finished training step {:d}'.format(step))
                break

    def test(self, sess):
        test_handle = sess.run(self.test_iterator.string_handle())
        # Initialize validation dataset, only execute once.
        sess.run(self.test_iterator.initializer)
        name_label_loss_probs = self._test_dataset(sess, test_handle, num_iters=self.test_size, step=self.cfg.restore_iters, ds_name='test')
        name_label_loss_probs = sorted(name_label_loss_probs, key=lambda x: x[2], reverse=True)
        # Save the test results to csv file
        name_label_loss_probs = ['{},{},{},{}'.format(name, label, loss, ','.join(list(map(str, probs)))) for name, label, loss, probs in name_label_loss_probs]
        csv_dir = '../test_results/{}/{}'.format(self.cfg.project, self.cfg.version)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        with open('{}/{}_{}.csv'.format(csv_dir, self.cfg.version, self.cfg.restore_iters), 'w+') as f:
            # f.write(('\n'.join(name_label_loss_probs)).encode('utf-8'))
            f.write('\n'.join(name_label_loss_probs))

