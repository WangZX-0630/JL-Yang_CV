import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import torch.nn.functional as F
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import ss
import time
from tqdm import tqdm
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

BATCHSIZE_PER_CARD = 4

class TTAFrame():
    def __init__(self, net, source_dir, transform):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.source_dir = source_dir
        self.transform = transform
        
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        # return self.test_one_img(path)
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_1(path)

    def test_one_img(self, path):
        picname = path.split('/')[-1] # top_potsdam_2_13_rgb-0.png
        gtname = picname[:-4] + '_label.png'
        gtpath = os.path.join(self.source_dir, 'gtblack', gtname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        img = torch.Tensor(img)
        mask = cv2.imread(gtpath, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask, dtype=np.int32)


        img = img.unsqueeze(0)
        start = time.time()
        pred = self.net.forward(img).squeeze()
        use_time = time.time() - start
        predindex = pred.max(dim=0)[1]
        score = pred.max(dim=0)[0]
        predindex = np.array(predindex.cpu())
        score = np.array(score.cpu().detach())

        return predindex, mask, use_time, score

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32) #/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32) #/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        picname = path.split('/')[-1]
        gtname = picname[:-4] + '_label.png'
        gtpath = os.path.join(self.source_dir, 'gtblack', gtname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = np.float32(img)
        mask = cv2.imread(gtpath, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask, dtype=np.int32)
        
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32) /255.0
        img5 = V(torch.Tensor(img5).cuda())
        
        pred = self.net.forward(img5).squeeze().cpu().data#.squeeze(1)
        pred = F.softmax(pred, dim=1)
        pred = pred.numpy()
        mask1 = pred[:4] + pred[4:,:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,:,::-1]
        mask4 = mask2[1].transpose(1, 2, 0)
        mask4 = np.rot90(mask4).transpose(2, 0, 1)
        mask3 = mask2[0] + mask4[:, ::-1, ::-1]
        # mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        pred = np.argmax(mask3, axis=0)
        score = np.max(mask3, axis=0)

        return pred, mask, 0, score / 8

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

def verifyexist(path):
    if not os.path.exists(path):
        os.makedirs(path)

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


save_path = './Res34S1024'
model_path = './expRes34S1024'
gray_dir = os.path.join(save_path, 'gray')
color_dir = os.path.join(save_path, 'color')
score_dir = os.path.join(save_path, 'score')
verifyexist(gray_dir)
verifyexist(color_dir)
verifyexist(score_dir)
# data transform
value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]
test_transform = ss.Compose([
    ss.ToTensor(),
    ss.Normalize(mean=mean, std=std)
])

source_dir = '/f2020/shengyifan/data/postdam1024/val'
# source = '/f2020/shengyifan/data/postdam512/val/img/'
source = source_dir + '/img/'
val = os.listdir(source)
tbar = tqdm(val)
solver = TTAFrame(DinkNet34, source_dir=source_dir, transform=test_transform)
solver.load(model_path + '/model_best.pth')
tic = time.time()
colors = np.loadtxt('./postdam_colors.txt').astype('uint8')
time_list = []
total_intersection, total_union, total_pred, total_ans = 0, 0, 0, 0
for i,name in enumerate(tbar):
    # name = top_potsdam_2_13_rgb-0.png
    # if i % 10 == 0:
    #     print(i/10, '    ','%.2f'%(time.time()-tic))
    pred, mask, use_time, score = solver.test_one_img_from_path(source+name) # H*W
    # mask = mask.numpy()
    time_list.append(use_time)
    # pred, mask = pred.cpu().numpy(), mask.cpu().numpy()
    intersection, union, target, output = intersectionAndUnion(pred, mask, K=6)
    total_intersection += intersection
    total_union += union
    total_pred += output
    total_ans += target

    grayname = os.path.splitext(name)[0] + '_pred.png'
    graypath = os.path.join(gray_dir, grayname)
    colorname = os.path.splitext(name)[0] + '_color.png'
    colorpath = os.path.join(color_dir, colorname)
    scorename = os.path.splitext(name)[0] + '_pred_score.npy'
    scorepath = os.path.join(score_dir, scorename)

    # cv2.imwrite(graypath, mask.astype(np.uint8))
    cv2.imwrite(graypath, np.array(pred))
    color = colorize(pred, colors)
    color.save(colorpath)
    np.save(scorepath, np.squeeze(score))


iou_class = total_intersection / total_union
mIoU = np.mean(iou_class)
oa = sum(total_intersection) / sum(total_ans)
pe = sum(np.multiply(total_pred, total_ans)) / sum(total_ans) ** 2
kappa = (oa - pe) / (1 - pe)
print('Average time: {}'.format(sum(time_list) / len(time_list)))
print('OA: {:.4f} mIoU: {:.4f} kappa: {:.4f}'.format(oa, mIoU, kappa))
