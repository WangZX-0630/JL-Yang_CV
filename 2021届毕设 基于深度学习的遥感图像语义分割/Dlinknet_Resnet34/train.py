import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from tensorboardX import SummaryWriter
import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
import ss

SHAPE = (1024,1024)
ROOT = '/f2020/shengyifan/data/postdam1024'
trainpath=os.path.join(ROOT,"train/img")
testpath=os.path.join(ROOT,"val/img")

trainlist = os.listdir(trainpath)
testlist = os.listdir(testpath)
NAME = 'log01_dink34'
BATCHSIZE_PER_CARD = 4
savepath = './expRes34S1024'

batchsize, test_batchsize = 16, 3
print('train batchsize: {}'.format(batchsize))
solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
writer1 = SummaryWriter(savepath)

dataset = ImageFolder(trainlist, ROOT, mode='train')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)
testdataset = ImageFolder(testlist, ROOT, mode='val')
testdata_loader = torch.utils.data.DataLoader(testdataset, batch_size=test_batchsize, shuffle=False, num_workers=4, drop_last=False)
mylog = open('logs/'+NAME+'.log','w')
tic = time.time()
no_optim = 0
total_epoch = 121
train_epoch_best_loss = 100.
print('Start time:{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
best_oa = 0.0
for epoch in range(0, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        # train_loss, batch_miou = solver.optimize()
        train_epoch_loss += train_loss

    train_epoch_loss /= len(data_loader_iter)
    writer1.add_scalar('Loss', train_epoch_loss, global_step=epoch)
    print('                ')
    print('epoch:', epoch, '    time:', int(time.time() - tic))
    print('train_loss:', train_epoch_loss)

    if epoch % 2 == 0:
        test_loader_iter=iter(testdata_loader)
        test_epoch_intersection = 0
        test_epoch_union = 0
        test_epoch_target = 0
        test_epoch_pred = 0
        for img, mask in test_loader_iter:
            solver.set_input(img, mask)
            intersection, union, target, output = solver.testmodel()
            test_epoch_intersection += intersection
            test_epoch_union += union
            test_epoch_target += target
            test_epoch_pred += output


        test_epoch_intersection = test_epoch_intersection.cpu().numpy()
        test_epoch_union = test_epoch_union.cpu().numpy()
        test_epoch_target = test_epoch_target.cpu().numpy()
        test_epoch_pred = test_epoch_pred.cpu().numpy()
        iou_class = test_epoch_intersection / test_epoch_union
        mIoU = np.mean(iou_class)
        oa = sum(test_epoch_intersection) / sum(test_epoch_target)
        pe = sum(np.multiply(test_epoch_pred, test_epoch_target)) / sum(test_epoch_target) ** 2
        kappa = (oa - pe) / (1 - pe)
        writer1.add_scalar('OA', oa, global_step=epoch)
        writer1.add_scalar('Kappa', kappa, global_step=epoch)
        writer1.add_scalar('Miou', mIoU, global_step=epoch)
        print('            ')
        print('This is test')
        print('epoch:', epoch, '    time:', int(time.time() - tic))
        print('test_oa:', oa)
        print('test_kappa ', kappa)
        print('test_miou ', mIoU)
        if oa > best_oa:
            solver.save(savepath+'/model_best.pth')
            print('**********************best at {} '.format(epoch))
            best_oa = oa


    # print >> mylog, '********'
    # print >> mylog, 'epoch:',epoch,'    time:',int(time()-tic)
    # print >> mylog, 'train_loss:',train_epoch_loss
    # print >> mylog, 'train_miou:', train_epoch_miou
    # print >> mylog, 'SHAPE:',SHAPE
    
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save(savepath+'/'+NAME+'.pth')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load(savepath+'/'+NAME+'.pth')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()
    
#print >> mylog, 'Finish!'
writer1.close()
print('Finish time:{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
print('Finish!')
