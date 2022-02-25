import argparse
import glob
import multiprocessing as mp
import os

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from collections import Counter
import xlwt as excel
from decimal import Decimal
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask_former import add_mask_former_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "MaskFormer demo"

def cal_confu_matrix(label, predict, class_num):
    confu_list = []
    for z in range(class_num):
        c = Counter(predict[np.where(label == z)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)
def metric_evaluate(confu_mat_total, save_path, save_name):
    class_num = confu_mat_total.shape[0]
    file = excel.Workbook(encoding='utf-8')
    table_name = save_name
    pic_name = table_name + ' metrics:'
    table = file.add_sheet(table_name)
    table_raw = 0
    table.write(table_raw, 0, pic_name)
    table_raw += 2

    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)
    raw_sum = np.sum(confu_mat, axis=0)
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)
    TP = []
    table.write(table_raw, 0, 'confusion_matrix:')
    table_raw = table_raw + 1
    #name_str = ['Clutter/background', 'Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car']
    name_str = ['Clutter/background', 'Building', 'Tree', 'Car', 'Low vegetation', 'Impervious surfaces']
    for i in range(class_num):
        table.write(table_raw, 1+i, name_str[i])
    for i in range(class_num):
        table_raw = table_raw + 1
        table.write(table_raw, 0, name_str[i])
        TP.append(confu_mat[i, i])
        for j in range(class_num):
            table.write(table_raw, j + 1, int(confu_mat_total[i, j]))
    TP = np.array(TP)
    FN = raw_sum - TP
    FP = col_sum - TP
    table_raw = table_raw + 2
    table.write(table_raw, 0, 'precision:')
    for i in range(class_num):
        table.write(table_raw, i + 1, Decimal(float(TP[i]/raw_sum[i])).quantize(Decimal("0.0000")))
    table_raw += 1
    table.write(table_raw, 0, 'Recall:')
    for i in range(class_num):
        table.write(table_raw, i + 1, Decimal(float(TP[i]/col_sum[i])).quantize(Decimal("0.0000")))
    f1_m = []
    iou_m = []
    table_raw += 1
    table.write(table_raw, 0, 'f1-score:')
    for i in range(class_num):
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)
        table.write(table_raw, i + 1, Decimal(f1).quantize(Decimal("0.0000")))
    table_raw += 1
    table.write(table_raw, 0, 'OA:')
    table.write(table_raw, 1, Decimal(float(oa)).quantize(Decimal("0.0000")))
    table_raw += 1
    table.write(table_raw, 0, 'Kappa:')
    table.write(table_raw, 1, Decimal(float(kappa)).quantize(Decimal("0.0000")))
    f1_m = np.array(f1_m)
    table_raw += 1
    table.write(table_raw, 0, 'f1-m:')
    table.write(table_raw, 1, Decimal(float(np.mean(f1_m))).quantize(Decimal("0.0000")))
    iou_m = np.array(iou_m)
    table_raw += 1
    table.write(table_raw, 0, 'mIOU:')
    table.write(table_raw, 1, Decimal(float(np.mean(iou_m))).quantize(Decimal("0.0000")))
    file.save(save_path + '/' + table_name + '.xls')
    
    #混淆矩阵归一化及可视化
    confusion_mat = confu_mat.astype(np.float32)
    confusion_mat_N = confusion_mat.copy()
    # 归一化
    for i in range(len(name_str)):#类别数
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
 
    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()
 
    # 设置文字
    xlocations = np.array(range(len(name_str)))
    #xlocations = np.array(range(len(name_str))
    plt.xticks(xlocations, name_str, rotation=60)
    #plt.xticks(xlocations, name_str)#rotation=60
    plt.yticks(xlocations, name_str)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + 'postdam_maskformer_base')
 
    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=Decimal(float(confusion_mat_N[i, j])).quantize(Decimal("0.0000")), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(save_path, 'Confusion_Matrix_' + save_name + '.png'))
    plt.close()

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if True:
        path10 = '/root/Mask/datasets/V_test/val/'#6000val原图文件夹
        path20 = '/root/Mask/datasets/V_test/val_png/'#val_label_png文件夹
        rootdir1=sorted(os.listdir(path10))
        rootdir2=sorted(os.listdir(path20))
        print(len(rootdir1))
        print(len(rootdir2))
        save_path = "/root/Mask/res/DC_V_0000"#精度计算结果保存文件夹
        RGB_path = "/root/Mask/res/DC_V_0000/png_fr/"#可视化结果保存文件夹
        save_name = "DC_V_0000_fr"
        
        path1= path10 + rootdir1[0]
        path2= path20 + rootdir1[0].split('.tif')[0] + '.png'#_noBoundary.png'
        print(path1)
        print(path2)
        img = read_image(path1, format="BGR")
        h,w,c=img.shape
        start_time = time.time()
        
        imgf = np.empty([h,w,3], dtype = int)

        visualized_output = demo.run_on_image(img)

        visualized_output=np.uint8(visualized_output)
        print(np.unique(visualized_output))
        
#         imgf[visualized_output==0,0:3]=[0,0,255]
#         imgf[visualized_output==1,0:3]=[255,0,0]
#         imgf[visualized_output==2,0:3]=[0,255,0]
#         imgf[visualized_output==3,0:3]=[0,255,255]
#         imgf[visualized_output==4,0:3]=[255,255,0]
#         imgf[visualized_output==5,0:3]=[255,255,255]
        rgb_path = RGB_path + rootdir1[0].split('.')[0] + '.png'
        cv2.imwrite(rgb_path, visualized_output)
        
        lab=cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
        print(path2)
        print(np.unique(lab))
        lab=np.uint8(lab)
        #print(lab.shape)
        confu_mat_total = cal_confu_matrix(lab,visualized_output, 6)
        for i in range(1,len(rootdir1)):
            path1= path10 + rootdir1[i]
            path2= path20 + rootdir1[i].split('.tif')[0] + '.png'#_noBoundary.png'
            img = read_image(path1, format="BGR")
            print(path1)
            print(path2)
            
            h,w,c=img.shape
            start_time = time.time()
        
            imgf = np.empty([h,w,3], dtype = int)
            
            visualized_output = demo.run_on_image(img)

            visualized_output=np.uint8(visualized_output)
            print(np.unique(visualized_output))
        
#             imgf[visualized_output==0,0:3]=[0,0,255]
#             imgf[visualized_output==1,0:3]=[255,0,0]
#             imgf[visualized_output==2,0:3]=[0,255,0]
#             imgf[visualized_output==3,0:3]=[0,255,255]
#             imgf[visualized_output==4,0:3]=[255,255,0]
#             imgf[visualized_output==5,0:3]=[255,255,255]
            rgb_path = RGB_path + rootdir1[i].split('.')[0] + '.png'
            cv2.imwrite(rgb_path, visualized_output)
        
            lab=cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
            print(lab.shape)
            print(visualized_output.shape)
            lab=np.uint8(lab)
            confu_mat_total = confu_mat_total + cal_confu_matrix(lab,visualized_output, 6)
        metric_evaluate(confu_mat_total, save_path, save_name)
