import glob
import math
import os
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import imageio
import tqdm
import cv2
import matplotlib.pyplot as plt
import random


class Parser:
    def parse(self):
        parser = ArgumentParser()
        parser.add_argument('--file_dir', help='Directory with images for process')
        parser.add_argument('--dest_dir', help='Directory for store')
        parser.add_argument('--model', help='Image convert type')
        parser.add_argument('--if_delete', help='if delete directory')
        args, _ = parser.parse_known_args()
        self.args = vars(args)

    def __getitem__(self, key):
        return self.args[key]

    def __str__(self):
        return str(self.args)


def npy2png(file_path, dest_path=None):
    if dest_path is None:
        dest_path = file_path
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for fileName in tqdm.tqdm(os.listdir(file_path)):
        arr = np.load(file_path + "\\" + fileName)
        for i in range(6):
            newFileName = fileName[0:fileName.find(".npy")] + "_{0}_gt.png".format(i)
            imageio.imsave(dest_path + "\\" + newFileName, arr[i, ...])
    print("Finish..")


def bmp2png(file_path, dest_path=None):
    if dest_path is None:
        dest_path = file_path
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for fileName in tqdm.tqdm(os.listdir(file_path)):
        newFileName = fileName[0:fileName.find(".bmp")] + "_im.jpg"
        im = Image.open(file_path + "\\" + fileName)
        im.save(dest_path + "\\" + newFileName)
    print("Finish..")


def deleteImages(file_path, imageFormat):
    for infile in glob.glob(os.path.join(file_path, imageFormat)):
        os.remove(infile)


def dilateImages(file_path, dest_path=None):
    if dest_path is None:
        dest_path = file_path
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    i = 0
    for fileName in tqdm.tqdm(os.listdir(file_path)):
        # newFileName = fileName[0:fileName.find("_seg.png")] + ".png"
        im = Image.open(file_path + "\\" + fileName).convert("L")
        im.save(dest_path + "\\" + fileName)
        i += 1
        if i == 144:
            break
    print("Finish..")


def renameImages(file_path, dest_path=None, if_delete=False):
    if dest_path is None:
        dest_path = file_path
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for fileName in tqdm.tqdm(os.listdir(file_path)):
        newFileName = fileName[0:fileName.find(".png")] + "_im.jpg"
        im = Image.open(file_path + "\\" + fileName)
        im.save(dest_path + "\\" + newFileName)
        if if_delete:
            os.remove(file_path + "\\" + fileName)
    print("Finish..")


def splitLabel(classNo, file_path, dest_path=None, ):
    """
    map to gary: 0 1 2 3 4 5 -> 255 29 179(178) 150(149) 226(225) 76
    根据原图修改后缀_gt.png _seg.png
    """
    if dest_path is None:
        dest_path = file_path
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if classNo == 0:
        for fileName in tqdm.tqdm(os.listdir(file_path)):
            # image = cv2.imread(file_path + "\\" + fileName)
            image = np.array(Image.open(os.path.join(file_path, fileName)).convert("L"))
            height, width = image.shape
            for i in range(height):
                for j in range(width):
                    # b, g, r = image[i, j]
                    if image[i, j] == 255:
                        pass
                    else:
                        image[i, j] = 0
            newFileName = fileName[0:fileName.find(".png")] + "_0_seg.png"
            # cv2.imwrite(dest_path + "\\" + newFileName, image)
            im = Image.fromarray(image)
            im.save(dest_path + "\\" + newFileName)
    elif classNo == 1:
        for fileName in tqdm.tqdm(os.listdir(file_path)):
            image = np.array(Image.open(os.path.join(file_path, fileName)).convert("L"))
            height, width = image.shape
            for i in range(height):
                for j in range(width):
                    # b, g, r = image[i, j]
                    if image[i, j] == 29:
                        image[i, j] = 255
                    else:
                        image[i, j] = 0
            newFileName = fileName[0:fileName.find(".png")] + "_1_seg.png"
            # cv2.imwrite(dest_path + "\\" + newFileName, image)
            im = Image.fromarray(image)
            im.save(dest_path + "\\" + newFileName)
    elif classNo == 2:
        for fileName in tqdm.tqdm(os.listdir(file_path)):
            image = np.array(Image.open(os.path.join(file_path, fileName)).convert("L"))
            height, width = image.shape
            for i in range(height):
                for j in range(width):
                    # b, g, r = image[i, j]
                    if image[i, j] == 178 or image[i, j] == 179:
                        image[i, j] = 255
                    else:
                        image[i, j] = 0
            newFileName = fileName[0:fileName.find(".png")] + "_2_seg.png"
            # cv2.imwrite(dest_path + "\\" + newFileName, image)
            im = Image.fromarray(image)
            im.save(dest_path + "\\" + newFileName)
    elif classNo == 3:
        for fileName in tqdm.tqdm(os.listdir(file_path)):
            image = np.array(Image.open(os.path.join(file_path, fileName)).convert("L"))
            height, width = image.shape
            for i in range(height):
                for j in range(width):
                    # b, g, r = image[i, j]
                    if image[i, j] == 149 or image[i, j] == 150:
                        image[i, j] = 255
                    else:
                        image[i, j] = 0
            newFileName = fileName[0:fileName.find(".png")] + "_3_seg.png"
            # cv2.imwrite(dest_path + "\\" + newFileName, image)
            im = Image.fromarray(image)
            im.save(dest_path + "\\" + newFileName)
    elif classNo == 4:
        for fileName in tqdm.tqdm(os.listdir(file_path)):
            image = np.array(Image.open(os.path.join(file_path, fileName)).convert("L"))
            height, width = image.shape
            for i in range(height):
                for j in range(width):
                    # b, g, r = image[i, j]
                    if image[i, j] == 225 or image[i, j] == 226:
                        image[i, j] = 255
                    else:
                        image[i, j] = 0
            newFileName = fileName[0:fileName.find(".png")] + "_4_seg.png"
            # cv2.imwrite(dest_path + "\\" + newFileName, image)
            im = Image.fromarray(image)
            im.save(dest_path + "\\" + newFileName)
    elif classNo == 5:
        for fileName in tqdm.tqdm(os.listdir(file_path)):
            image = np.array(Image.open(os.path.join(file_path, fileName)).convert("L"))
            height, width = image.shape
            for i in range(height):
                for j in range(width):
                    # b, g, r = image[i, j]
                    if image[i, j] == 76:
                        image[i, j] = 255
                    else:
                        image[i, j] = 0
            newFileName = fileName[0:fileName.find(".png")] + "_5_seg.png"
            # cv2.imwrite(dest_path + "\\" + newFileName, image)
            im = Image.fromarray(image)
            im.save(dest_path + "\\" + newFileName)


def fuseGT(file_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    gt_list = []
    idx = 0
    for file in tqdm.tqdm(os.listdir(file_dir)):
        if "_seg.png" in file:
            gt_list.append(str(file)[:-10])
    gt_list = np.unique(gt_list)
    for gt in tqdm.tqdm(gt_list):
        '''if idx > 144:
            break'''
        fusion = np.zeros((1024, 1024))
        single_gt = np.array(Image.open(file_dir + "/" + gt + "_2_seg.png").convert("L")) / 255
        fusion[single_gt != 0] = 179  # Low vegetation
        single_gt = np.array(Image.open(file_dir + "/" + gt + "_0_seg.png").convert("L")) / 255
        fusion[single_gt != 0] = 255  # Surface
        single_gt = np.array(Image.open(file_dir + "/" + gt + "_1_seg.png").convert("L")) / 255
        fusion[single_gt != 0] = 29  # Building
        single_gt = np.array(Image.open(file_dir + "/" + gt + "_3_seg.png").convert("L")) / 255
        fusion[single_gt != 0] = 150  # Tree
        single_gt = np.array(Image.open(file_dir + "/" + gt + "_4_seg.png").convert("L")) / 255
        fusion[single_gt != 0] = 226  # Car
        single_gt = np.array(Image.open(file_dir + "/" + gt + "_5_seg.png").convert("L")) / 255
        fusion[single_gt != 0] = 76  # Clutters/background
        cv2.imwrite(dest_dir + "/" + gt + ".png", fusion)
        idx += 1


def filterDataset(file_dir):
    gt_list = []
    idx = 0
    for file in tqdm.tqdm(os.listdir(file_dir)):
        if "_gt.png" in file:
            gt_list.append(str(file)[:-7])
    for gt_name in tqdm.tqdm(gt_list):
        gt = np.array(Image.open(file_dir + "\\" + gt_name + "_gt.png").convert("L")) / 255
        if 1 not in gt:
            os.remove(file_dir + "\\" + gt_name + "_gt.png")
            os.remove(file_dir + "\\" + gt_name + "_seg.png")


def image_flip(im_path, gt_path, dest_path):
    # 翻转图像 data augmentation process
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    all_im = []
    for im_name in os.listdir(im_path):
        all_im.append(im_name[:-4])
    print("total image: ", len(all_im))
    for im_name in tqdm.tqdm(all_im):
        for i in range(6):
            gt = Image.open(os.path.join(gt_path, im_name + "_{0}_gt.png".format(i)))
            im = Image.open(os.path.join(im_path, im_name + ".png"))
            if random.randint(0, 1) == 1:
                gt_flip = gt.transpose(Image.FLIP_LEFT_RIGHT)
                im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                gt_flip = gt.transpose(Image.FLIP_TOP_BOTTOM)
                im_flip = im.transpose(Image.FLIP_TOP_BOTTOM)
            gt_flip.save(os.path.join(dest_path, im_name + "_{0}_f_gt.png".format(i)))
            im_flip.save(os.path.join(dest_path, im_name + "_{0}_f_im.jpg".format(i)))
            gt.save(os.path.join(dest_path, im_name + "_{0}_gt.png".format(i)))
            im.save(os.path.join(dest_path, im_name + "_{0}_im.jpg".format(i)))


def mask_montage(mask_path, save_path, crop_size):
    """
    对于1024size 裁剪为6x6 最后一行的最上和最后一列的最左各有144x1024 和 1024x144的overlap
    对于512size 裁剪为12x12 最后一行的最上和最后一列的最左各有144x1024 和 1024x144的overlap
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_im_list = []
    for name in os.listdir(mask_path):
        full_im_list.append(name[:20])
    full_im_list = np.unique(full_im_list)
    for im_name in tqdm.tqdm(full_im_list):
        full_image = np.zeros((6000, 6000, 3))
        x, y = 0, 0
        if crop_size == 1024:
            for i in range(36):
                split_image = np.array(
                    Image.open(os.path.join(mask_path, im_name + "-{0}.png".format(i))).convert("RGB"))
                if (i + 1) % 6 == 0:
                    split_image = split_image[:, 144:, :]
                if i >= 30:
                    split_image = split_image[144:, :, ]
                if i < 5:
                    full_image[0:1024, x:x + 1024, ] = split_image
                    x += 1024
                elif i == 5:
                    full_image[0:1024, x:6000, ] = split_image
                    x = 0
                elif i < 11:
                    full_image[1024:2048, x:x + 1024, ] = split_image
                    x += 1024
                elif i == 11:
                    full_image[1024:2048, x:6000, ] = split_image
                    x = 0
                elif i < 17:
                    full_image[2048:3072, x:x + 1024, ] = split_image
                    x += 1024
                elif i == 17:
                    full_image[2048:3072, x:6000, ] = split_image
                    x = 0
                elif i < 23:
                    full_image[3072:4096, x:x + 1024, ] = split_image
                    x += 1024
                elif i == 23:
                    full_image[3072:4096, x:6000, ] = split_image
                    x = 0
                elif i < 29:
                    full_image[4096:5120, x:x + 1024, ] = split_image
                    x += 1024
                elif i == 29:
                    full_image[4096:5120, x:6000, ] = split_image
                    x = 0
                elif i < 35:
                    full_image[5120:6000, x:x + 1024, ] = split_image
                    x += 1024
                elif i == 35:
                    full_image[5120:6000, x:6000, ] = split_image
                    x = 0
        elif crop_size == 512:
            for i in range(144):
                split_image = np.array(
                    Image.open(os.path.join(mask_path, im_name + "-{0}_mask.png".format(i))).convert("RGB"))
                if (i + 1) % 12 == 0:
                    split_image = split_image[:, 144:, :]
                if i >= 132:
                    split_image = split_image[144:, :, ]
                if i < 11:
                    full_image[0:512, x:x + 512, ] = split_image
                    x += 512
                elif i == 11:
                    full_image[0:512, x:6000, ] = split_image
                    x = 0
                elif i < 23:
                    full_image[512:1024, x:x + 512, ] = split_image
                    x += 512
                elif i == 23:
                    full_image[512:1024, x:6000, ] = split_image
                    x = 0
                elif i < 35:
                    full_image[1024:1536, x:x + 512, ] = split_image
                    x += 512
                elif i == 35:
                    full_image[1024:1536, x:6000, ] = split_image
                    x = 0
                elif i < 47:
                    full_image[1536:2048, x:x + 512, ] = split_image
                    x += 512
                elif i == 47:
                    full_image[1536:2048, x:6000, ] = split_image
                    x = 0
                elif i < 59:
                    full_image[2048:2560, x:x + 512, ] = split_image
                    x += 512
                elif i == 59:
                    full_image[2048:2560, x:6000, ] = split_image
                    x = 0
                elif i < 71:
                    full_image[2560:3072, x:x + 512, ] = split_image
                    x += 512
                elif i == 71:
                    full_image[2560:3072, x:6000, ] = split_image
                    x = 0
                elif i < 83:
                    full_image[3072:3584, x:x + 512, ] = split_image
                    x += 512
                elif i == 83:
                    full_image[3072:3584, x:6000, ] = split_image
                    x = 0
                elif i < 95:
                    full_image[3584:4096, x:x + 512, ] = split_image
                    x += 512
                elif i == 95:
                    full_image[3584:4096, x:6000, ] = split_image
                    x = 0
                elif i < 107:
                    full_image[4096:4608, x:x + 512, ] = split_image
                    x += 512
                elif i == 107:
                    full_image[4096:4608, x:6000, ] = split_image
                    x = 0
                elif i < 119:
                    full_image[4608:5120, x:x + 512, ] = split_image
                    x += 512
                elif i == 119:
                    full_image[4608:5120, x:6000, ] = split_image
                    x = 0
                elif i < 131:
                    full_image[5120:5632, x:x + 512, ] = split_image
                    x += 512
                elif i == 131:
                    full_image[5120:5632, x:6000, ] = split_image
                    x = 0
                elif i < 143:
                    full_image[5632:6000, x:x + 512, ] = split_image
                    x += 512
                elif i == 143:
                    full_image[5632:6000, x:6000, ] = split_image
                    x = 0
        else:
            print("Invalid crop size")
            assert ValueError
        full_image = Image.fromarray(np.uint8(full_image))
        full_image.save(os.path.join(save_path, im_name + ".png"))


def image_slice(img_path, save_path, crop_size):
    """
    除最后一行一列除不尽时有重叠，其余无overlap
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    im_list = os.listdir(img_path)
    for im_name in tqdm.tqdm(im_list):
        im = np.array(Image.open(os.path.join(img_path, im_name)).convert("RGB"))
        batch_num = math.ceil(6000 / crop_size)
        x_beacon, y_beacon = 0, 0
        for i in range(int(math.pow(batch_num, 2))):
            if y_beacon < batch_num-1:
                if i >= math.pow(batch_num, 2) - batch_num:
                    slice_im = Image.fromarray(im[6000 - crop_size:6000,
                                               y_beacon * crop_size:(y_beacon + 1) * crop_size, ])
                else:
                    slice_im = Image.fromarray(im[x_beacon * crop_size:(x_beacon + 1) * crop_size,
                                               y_beacon * crop_size:(y_beacon + 1) * crop_size, ])
                y_beacon += 1
            elif y_beacon == batch_num-1:
                if i >= math.pow(batch_num, 2) - batch_num:
                    slice_im = Image.fromarray(im[6000 - crop_size:6000,
                                               6000 - crop_size:6000, ])
                else:
                    slice_im = Image.fromarray(im[x_beacon * crop_size:(x_beacon + 1) * crop_size,
                                               6000 - crop_size:6000, ])
                x_beacon += 1
                y_beacon = 0
            # slice_im.save(os.path.join(save_path, im_name[:-4]+"_{0}.png".format(i)))
            slice_im.save(os.path.join(save_path, im_name[:-9] + "_{0}".format(i) + im_name[-9:]))


para = Parser()
para.parse()
print(para)
if para['model'] == 'npy2png':
    print("1")
    npy2png(para['file_dir'], para['dest_dir'])
if para['model'] == 'bmp2png':
    bmp2png(para['file_dir'], para['dest_dir'])

if __name__ == "__main__":
    '''gt = np.array(Image.open("Potsdam512/test/gt_small/top_potsdam_2_13_rgb-0.png").convert("P"))
    print(gt)
    seg = np.array(Image.open("Potsdam512/test/seg_small/top_potsdam_2_13_rgb-0.png").convert("P"))
    print(seg)
    total_classes = np.union1d(np.unique(gt), np.unique(seg))
    seg[gt == 0] = 0
    print(total_classes)
    total_classes = total_classes[1:]
    print(total_classes)
    total_classes -= 1 /
    print(total_classes)'''
    # image_flip("../Potsdam1024/train/im", "../Potsdam1024/train/gt_split", "../Potsdam1024/train/train_dataset")
    mask_montage("../Output/maskFusion/FastFCN_50_512_DA", "../Output/maskFusion/FastFCN_50_512_DA_6000", 512)
    # image_slice("../Potsdam1024/val/gt_split_montage", "../Larger_Potsdam/2000/gt_split", 2000)
    # renameImages("../Larger_Potsdam/6000/img", "../Larger_Potsdam/6000/6000_test", if_delete=False)

