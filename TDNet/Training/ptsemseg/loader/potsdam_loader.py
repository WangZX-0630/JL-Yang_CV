import os
import torch
import numpy as np
import imageio
from glob import glob

from torch.utils import data
import random
from Training.ptsemseg.utils import recursive_glob
from Training.ptsemseg.augmentations.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class PotsdamLoader(data.Dataset):
    colors = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0]
    ]

    label_colours = dict(zip(range(6), colors))

    def __init__(
            self,
            root,
            split="train",
            augmentations=None,
            test_mode=False,
            model_name=None,
            interval=2,
            path_num=2,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.path_num = path_num
        self.interval = interval
        self.root = root        # 数据集根目录
        self.split = split
        self.augmentations = augmentations
        self.test_mode = test_mode
        self.model_name = model_name
        self.n_classes = 6
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, "img-png")
        # self.videos_base = os.path.join(self.root, "leftImg8bit_sequence", self.split)
        self.annotations_base = os.path.join(self.root, self.split, "label_gray")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = []
        self.valid_classes = [0, 1, 2, 3, 4, 5]
        self.class_names = [
            "ground",
            "building",
            "vegetation",
            "tree",
            "car",
            "background"
        ]

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(6)))  # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if not self.test_mode:
            img_path = self.files[self.split][index].rstrip()

            # Potsdam的img与lbl对应关系示例:
            # top_potsdam_2_10_rgb-0.png
            # top_potsdam_2_10_rgb-0.bmp-2019-10-03-21-58-03-495.png
            basename = os.path.split(img_path)[1].replace('.png', '')  # 获取top_potsdam_2_10_rgb-0
            lbl_filename = glob(pathname=os.path.join(self.annotations_base, f'{basename}.bmp-*'))
            lbl_path = os.path.join(self.annotations_base, lbl_filename[0])

            lbl = imageio.imread(lbl_path)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

            f1_img = imageio.imread(img_path)
            f1_img = np.array(f1_img, dtype=np.uint8)
            f2_img, f3_img, f4_img = f1_img

            if self.augmentations is not None:
                [f4_img, f3_img, f2_img, f1_img], lbl = self.augmentations([f4_img, f3_img, f2_img, f1_img], lbl)

            f4_img = f4_img.float()
            f3_img = f3_img.float()
            f2_img = f2_img.float()
            f1_img = f1_img.float()
            lbl = torch.from_numpy(lbl).long()

            if self.path_num == 4:
                return [f1_img, f2_img, f3_img, f4_img], lbl
            else:
                return [f3_img, f4_img], lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_pred(self, mask):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            mask[mask == _predc] = self.valid_classes[_predc]
        return mask.astype(np.uint8)
