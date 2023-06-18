import numpy as np
from sklearn.metrics import confusion_matrix as cm_fn
import matplotlib.pyplot as plt
import pandas


# import torch


class Evaluator(object):
    """
    Code is modified based on  https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py#L33
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # (TP+TN)/(ALL)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        MIoU = np.nanmean(MIoU)

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, f"gt shape is {gt_image.shape}, while pre shape is {pre_image.shape}"

        # gt_image=gt_image.astype(np.int8)
        # pre_image = pre_image.astype(np.int8)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.TP = self.confusion_matrix[1, 1]
        self.FN = self.confusion_matrix[1, 0]
        self.FP = self.confusion_matrix[0, 1]
        self.TN = self.confusion_matrix[0, 0]


    def F1(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)

    def Precision(self):
        return self.TP / (self.TP + self.FP)

    def Recall(self):
        return self.TP / (self.TP + self.FN)

    def OA(self):
        return (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN)
    def mIoU(self):
        return (self.TP + self.TN) / (self.TP + self.FP + + self.FN)
    def IoU(self):
        return self.TP / (self.TP + self.FP + self.FN)

    def print_stat(self):
        print("IoU: ", self.IoU())
        print("mIoU", self.Mean_Intersection_over_Union())
        print("OA:", self.OA())
        print("F1:", self.F1())
        print("Recall:", self.Recall())
        print("Precision:", self.Precision())
        print("TP:", self.TP / (self.TP + self.FP + self.FN + self.TN))
        print("FP:", self.FP / (self.TP + self.FP + self.FN + self.TN))
        print("TN:", self.TN / (self.TP + self.FP + self.FN + self.TN))
        print("FN:", self.FN / (self.TP + self.FP + self.FN + self.TN))


    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Evaluator2(object):
    def __init__(self, num_class=2):
        self.num_class = num_class
        self.cm = np.zeros((self.num_class,) * 2)
        self.FP = 0
        self.FN = 0
        self.TP = 0
        self.TN = 0

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # gt_image = gt_image.astype("int")
        # pre_image = pre_image.astype("int")
        gt_image = gt_image.flatten()
        pre_image = pre_image.flatten()
        self.cm += cm_fn(gt_image, pre_image)  # confusion_matrix(gt,y_pred)
        tn, fp, fn, tp = cm_fn(gt_image, pre_image).ravel()
        self.FP += fp
        self.FN += fn
        self.TP += tp
        self.TN += tn
        """
        for multi classes
        self.FP = self.cm.sum(axis=0) - np.diag(self.cm)
        self.FN = self.cm.sum(axis=1) - np.diag(self.cm)
        self.TP = np.diag(self.cm)
        self.TN = self.cm.sum() - (self.FP + self.FN + self.TP)
        print(self.TN, self.TP, self.FP, self.FN)
        print(self.TP / (self.TP + self.FP))
        """

    def F1_score(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)

    def Precision(self):
        return self.TP / (self.TP + self.FP)

    def Recall(self):
        return self.TP / (self.TP + self.FN)

    def IoU(self):
        return self.TP / (self.TP + self.FP + self.FN)

    def OA(self):
        return (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN)

    def show_confusion_matrix(self):
        plt.matshow(self.cm, cmap=plt.cm.Reds)
        for i in range(len(self.cm)):
            for j in range(len(self.cm)):
                plt.annotate(self.cm[j, i] / self.cm.sum(), xy=(i, j), horizontalalignment='center',
                             verticalalignment='center')
        plt.show()

    def reset(self):
        self.FP = 0
        self.FN = 0
        self.TP = 0
        self.TN = 0

