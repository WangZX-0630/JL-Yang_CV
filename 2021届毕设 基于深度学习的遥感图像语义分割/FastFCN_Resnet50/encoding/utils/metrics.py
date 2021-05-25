##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import threading
import numpy as np

import torch

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes
    """
    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union, pred, ans = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
                self.total_pred += pred
                self.total_ans += ans
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        pe = sum(np.multiply(self.total_pred, self.total_ans)) / sum(self.total_ans) ** 2
        kappa = (pixAcc - pe) / (1 - pe)
        return pixAcc, mIoU, kappa
 
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_pred = 0
        self.total_ans = 0
        return


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    #output维度是B*C*H*W，target维度是B*H*W
    _, predict = torch.max(output, 1) #在dim=1维度即C维度取最大值，predict是最大值的索引
    #predict维度是B*H*W

    predict = predict.cpu().numpy().astype('int64') + 1 #所有元素+1，从原来的0-5变成1-6
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0) #应该是从1开始才算类别，target是答案
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1 #astype强制类型转换
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target) #既有类别又预测正确
    # intersection是个矩阵，预测正确的地方是类别数（如1，2，3等），否则是0
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    # np.histogram用来统计位于区间的数字个数，bins是区间个数（类别数），range是统计范围，
    # area_inter是对每类预测正确的统计
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    # area_pred是对每个种类预测的统计
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    # area_lab是对答案每个种类的统计
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union, area_pred, area_lab
