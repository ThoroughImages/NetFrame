import random
import numpy as np


class Evaluation(object):
    """Public evaluation methods"""

    def __init__(self):
        super(Evaluation, self).__init__()
        self.loss_arr = []
        self.precision_arr = []
        self.recall_arr = []
        self.accuracy_arr = []

    @staticmethod
    def _get_metrics(tp, tn, fp, fn):
        precision = 1. * tp / max((tp + fp), 1)
        recall = 1. * tp / max((tp + fn), 1)
        accuracy = 1. * (tp + tn) / max((tp + tn + fp + fn), 1)
        return precision, recall, accuracy

    def training_metrics(self, reduced_dict):
        tp, tn, fp, fn = reduced_dict['tp'], reduced_dict['tn'], reduced_dict['fp'], reduced_dict['fn']
        precision, recall, accuracy = self._get_metrics(tp, tn, fp, fn)
        # f1 = 2 * precision * recall / (precision + recall)        
        self.precision_arr.append(precision)
        self.recall_arr.append(recall)
        self.accuracy_arr.append(accuracy)
        self.loss_arr.append(reduced_dict['loss'])

        return np.array(self.loss_arr)[-100:].mean(), np.array(self.precision_arr)[-100:].mean(), \
                np.array(self.recall_arr)[-100:].mean(), np.array(self.accuracy_arr)[-100:].mean()

    def whole_dataset_metrics(self, reduced_dict_list):
        tp = np.sum([r['tp'] for r in reduced_dict_list])
        tn = np.sum([r['tn'] for r in reduced_dict_list])
        fp = np.sum([r['fp'] for r in reduced_dict_list])
        fn = np.sum([r['fn'] for r in reduced_dict_list])
        precision, recall, accuracy = self._get_metrics(tp, tn, fp, fn)
        loss = np.mean([r['loss'] for r in reduced_dict_list])
        return loss, precision, recall, accuracy

