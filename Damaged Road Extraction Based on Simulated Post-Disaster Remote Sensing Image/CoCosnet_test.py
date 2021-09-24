# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os,cv2,random


#User should store the original remote sensing images and their semantic segmentation masks in imgs/ade20k/validation and keep the file names in pair
#like a.jpg for original remote sensing images and a.png for its semantic segmentation mask
#clear the file folder if you do not need the images, but do not delete the file folder
#the semantic segmentation mask should be one channel

#change the name of the original remote sensing images
path="imgs/ade20k/training"

fileList=os.listdir(path)

n=0
j=1
for i in fileList:
    if (i[-3:]=="jpg"):
        oldname_jpg=path+ os.sep + fileList[n]
        oldname_png=path+ os.sep + fileList[n].replace("jpg","png")
        newname_jpg=path + os.sep +"ADE_train_"+str(j)+".jpg"
        newname_png=path + os.sep +"ADE_train_"+str(j)+".png"
        j+=1
        os.rename(oldname_jpg,newname_jpg)
        os.rename(oldname_png,newname_png)
    n+=1
    
#copy the original remote sensing images to another file folder
path = "imgs/ade20k/training"
number_of_images = 0

for root, dirs, files in os.walk(path):
  for afile in files:
    if afile[-3:].lower() == 'jpg':
        print(afile)
        number_of_images += 1
        img = cv2.imread(os.path.join("imgs/ade20k/training",afile),-1)
        cv2.imwrite(os.path.join("imgs/ade20k/validation",afile.replace("ADE_train_","ADE_val_")),img)
        #os.system('cp '+os.path.join("imgs/ade20k/training",afile)+' '+os.path.join("imgs/ade20k/validation",afile.replace("ADE_train_","ADE_val_")))
        
#erase part of road masks
length = 300 #length of erased block
road = 45 #value of road
background = 10  #value of nonroad
numberofErased = 5000 #lower limit of erased road pixels 
path = "imgs/ade20k/training"  
png_list = [x for x in os.listdir(path) if x.endswith(".png")] 
for num, i in enumerate(png_list):
    img = cv2.imread(os.path.join("imgs/ade20k/training",i),cv2.IMREAD_GRAYSCALE) #groundtruth of mask after erasing
    mask_erased = cv2.imread(os.path.join("imgs/ade20k/training",i),cv2.IMREAD_GRAYSCALE) #groundtruth of erased mask

    for x in range(mask_erased.shape[0]):
        for y in range(mask_erased.shape[1]):
            mask_erased[x,y] = background
    pixel_of_road = 0
    x = 0
    y = 0

    while((pixel_of_road<numberofErased)or(img[x-1,y-1]==road)or(img[x+length+1,y-1]==road)or(img[x-1,y+length+1]==road)or(img[x+length+1,y+length+1]==road)):
        pixel_of_road = 0
        x = random.randint(2,img.shape[0]-length-2)
        y = random.randint(2,img.shape[1]-length-2)
        for x_s in range(x,x+length):
            for y_s in range(y,y+length):
                if(img[x_s,y_s] == road):
                    pixel_of_road+=1
    for x_s in range(x,x+length):
        for y_s in range(y,y+length):
            if(img[x_s,y_s]==road):
                img[x_s,y_s] = background
                mask_erased[x_s,y_s] = road
    cv2.imwrite(os.path.join("imgs/ade20k/validation",i.replace("ADE_train_","ADE_val_")), img)
    cv2.imwrite(os.path.join("imgs/ade20k/erased",i.replace("ADE_train_","ADE_compare_")), mask_erased)
    
#create txt for CoCosNet
data = open("data/ade20k_ref_test.txt","w").close()
data = open("data/ade20k_ref_test.txt","w")
for i in range(1,number_of_images+1):
    data.write("ADE_val_%d.jpg," % i)
    data.write("ADE_train_%d.jpg\n" % i)
data.close()

#run CoCosNet
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel

opt = TestOptions().parse()
print("running CoCosNet")
torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')

# test
for i, data_i in enumerate(dataloader):
    print('{} / {}'.format(i, len(dataloader)))
    if i * opt.batchSize >= opt.how_many:
        break
    imgs_num = data_i['label'].shape[0]
    #data_i['stage1'] = torch.ones_like(data_i['stage1'])
    
    out = model(data_i, mode='inference')
    if opt.save_per_img:
        root = save_root + '/test_per_img/'
        if not os.path.exists(root + opt.name):
            os.makedirs(root + opt.name)
        imgs = out['fake_image'].data.cpu()
        try:
            imgs = (imgs + 1) / 2
            for i in range(imgs.shape[0]):
                if opt.dataset_mode == 'deepfashion':
                    name = data_i['path'][i].split('Dataset/DeepFashion/')[-1].replace('/', '_')
                else:
                    name = os.path.basename(data_i['path'][i])
                vutils.save_image(imgs[i:i+1], root + opt.name + '/' + name,  
                        nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)
    else:
        if not os.path.exists(save_root + '/test/' + opt.name):
            os.makedirs(save_root + '/test/' + opt.name)

        if opt.dataset_mode == 'deepfashion':
            label = data_i['label'][:,:3,:,:]
        elif opt.dataset_mode == 'celebahqedge':
            label = data_i['label'].expand(-1, 3, -1, -1).float()
        else:
            label = masktorgb(data_i['label'].cpu().numpy())
            label = torch.from_numpy(label).float() / 128 - 1

        imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu()), 0)
        try:
            imgs = (imgs + 1) / 2
            vutils.save_image(imgs, save_root + '/test/' + opt.name + '/' + str(i) + '.png',  
                    nrow=imgs_num, padding=0, normalize=False)
        except OSError as err:
            print(err)

#copy images
path = "imgs/ade20k/training"

for root, dirs, files in os.walk(path):
  for afile in files:
    if afile[-3:].lower() == 'jpg':
        print(afile)
        img = cv2.imread(os.path.join("imgs/ade20k/training",afile),-1)
        cv2.imwrite(os.path.join("dataset/pre_disaster",afile),img)
        #os.system('cp '+os.path.join("imgs/ade20k/training",afile)+' '+os.path.join("../D-LinkNet/dataset/pre_disaster",afile))
        
path = "output/test_per_img/ade20k"

for root, dirs, files in os.walk(path):
  for afile in files:
    if afile[-3:].lower() == 'jpg':
        print(afile)
        img = cv2.imread(os.path.join("output/test_per_img/ade20k",afile),-1)
        cv2.imwrite(os.path.join("dataset/post_disaster",afile),img)
        #os.system('cp '+os.path.join("output/test_per_img/ade20k",afile)+' '+os.path.join("../D-LinkNet/dataset/post_disaster",afile))
        
path = "imgs/ade20k/erased"

for root, dirs, files in os.walk(path):
  for afile in files:
    if afile[-3:].lower() == 'png':
        print(afile)
        img = cv2.imread(os.path.join("imgs/ade20k/erased",afile),-1)
        cv2.imwrite(os.path.join("submits/label",afile),img)
        #os.system('cp '+os.path.join("imgs/ade20k/erased",afile)+' '+os.path.join("../D-LinkNet/submits/label",afile))
        
#change size of pre-disaster images
path = "dataset/pre_disaster"
tif_list = [x for x in os.listdir(path) if x.endswith(".jpg")]
for num, i in enumerate(tif_list):
    img = cv2.imread(os.path.join("dataset/pre_disaster",i),-1)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(os.path.join("dataset/pre_disaster",i), img)
        
#change color of masks
path = "submits/label"
tif_list = [x for x in os.listdir(path) if x.endswith(".png")]
for num, i in enumerate(tif_list):
    img = cv2.imread(os.path.join("submits/label",i),-1)
    img = cv2.resize(img, (256, 256))
    print(num,'  ',i)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if ((img[x,y]==45)):
                img[x,y]=255
            else:
                img[x,y]=0
    cv2.imwrite(os.path.join("submits/label",i), img)

from master import master_function

master_function()