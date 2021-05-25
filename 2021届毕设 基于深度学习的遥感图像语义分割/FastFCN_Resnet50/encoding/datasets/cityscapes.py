###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2018
###########################################################################

import os
import random
import numpy as np

import torch

from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter

from .base import BaseDataset

class CitySegmentation(BaseDataset):
    NUM_CLASS = 19
    def __init__(self, root="/f2020/shengyifan/data/cityscapes", split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CitySegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.images, self.mask_paths = get_city_pairs(self.root, self.split) # 路径列表
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")
        self._indices = np.array(range(-1, 19)) # [-1,0,1...18]共20个数
        self._classes = np.array([0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 31, 32, 33]) # 20个元素
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1,  0,  1, -1, -1, 
                              2,   3,  4, -1, -1, -1,
                              5,  -1,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18]) # 35个数
        self._mapping = np.array(range(-1, len(self._key)-1)).astype('int32') # [-1,0,1,2...33]，35个数

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask) # unique去除重复元素，并按从小到大返回一个列表
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        # ravel()将mask展成一维，digitize返回mask中的元素在_mapping中所处的区间，right为True则包含区间右端点
        # 所达到的效果应该就是将mask中所有元素+1
        return self._key[index].reshape(mask.shape) # 将mask换成19类，从-1到18

    def _preprocess(self, mask_file):
        if os.path.exists(mask_file):
            masks = torch.load(mask_file)
            return masks
        masks = []
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        tbar = tqdm(self.mask_paths)
        for fname in tbar:
            tbar.set_description("Preprocessing masks {}".format(fname))
            mask = Image.fromarray(self._class_to_index(
                np.array(Image.open(fname))).astype('int8'))
            masks.append(mask)
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index]) # test只有图片，没有答案
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror，随机镜像
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        # base_size=520, crop_size=480
        w, h = img.size # 原始尺寸
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._indices)
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)


def get_city_pairs(folder, split='train'): # folder=~/.encoding/data
    # folder=/f2020/shengyifan/data/cityscapes
    def get_path_pairs(img_folder, mask_folder):
        img_paths = [] #空列表
        mask_paths = []  
        for root, directories, files in os.walk(img_folder): #路径，子文件夹，子文件
            # root应该是/f2020/shengyifan/data/cityscapes/leftImg8bit/train/aachen
            for filename in files:
                # filename应该是aachen_000000_000019_leftImg8bit.png
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    # imgpath是/f2020/shengyifan/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    # dirname去掉最后的文件名，返回目录，如imgpath是/f2020/shengyifan/data/cityscapes/leftImg8bit/train/aachen
                    # 再经basename变成aachen
                    maskname = filename.replace('leftImg8bit','gtFine_labelIds')
                    # 将filename中的leftImg8bit换成gtFine_labelIds
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath) #向路径列表中添加路径
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths # 返回路径列表

    if split == 'train' or split == 'val' or split == 'test':
        # folder=/f2020/shengyifan/data/cityscapes
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        # folder=/f2020/shengyifan/data/cityscapes
        mask_folder = os.path.join(folder, 'gtFine/'+ split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths
