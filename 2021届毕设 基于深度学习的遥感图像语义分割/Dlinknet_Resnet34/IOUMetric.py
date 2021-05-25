import numpy as np
class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist)) # np.diag输出矩阵的对角线元素
        #iu是一个列表，包含每一类的iu
        pe=np.dot(self.hist.sum(axis=0), self.hist.sum(axis=1)) / self.hist.sum()**2
        kappa=(acc-pe)/(1-pe)
        mean_iu = np.nanmean(iu) #忽略nan求平均
        # freq = self.hist.sum(axis=1) / self.hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        self.hist = np.zeros((self.num_classes, self.num_classes)) #清0
        return mean_iu, acc, kappa
