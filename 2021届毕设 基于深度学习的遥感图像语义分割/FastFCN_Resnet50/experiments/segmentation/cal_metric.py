import os
import numpy as np
import cv2
from tqdm import tqdm

gtdir = '/f2020/shengyifan/data/postdam1024/val/gtblack'
preddir = './expRes50S1024/gray'
total_intersection, total_union, total_pred, total_ans = 0, 0, 0, 0


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target, area_output


gtlist = os.listdir(gtdir)
tbar = tqdm(gtlist)
for item in tbar:
    # item = top_potsdam_2_13_rgb-0_label.png
    # pred名称top_potsdam_2_13_rgb-0_pred.png
    predname = item[:-9] + 'pred.png'
    gtpath = os.path.join(gtdir, item)
    predpath = os.path.join(preddir, predname)
    gt = cv2.imread(gtpath, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(predpath, cv2.IMREAD_GRAYSCALE)
    gt = np.array(gt)
    pred = np.array(pred)
    intersection, union, target, output = intersectionAndUnion(pred, gt, K=6)
    total_intersection += intersection
    total_union += union
    total_pred += output
    total_ans += target

iou_class = total_intersection / total_union
mIoU = np.mean(iou_class)
oa = sum(total_intersection) / sum(total_ans)
pe = sum(np.multiply(total_pred, total_ans)) / sum(total_ans) ** 2
kappa = (oa - pe) / (1 - pe)
print('OA: {:.4f} mIoU: {:.4f} kappa: {:.4f}'.format(oa, mIoU, kappa))

