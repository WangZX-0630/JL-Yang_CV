import torch
import torch.nn as nn
from torch.autograd import Variable as V
import IOUMetric
import cv2
import numpy as np
from sklearn.metrics import *

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        self.calIOU = IOUMetric.Metric(6)
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        predindex = pred.max(dim=1)[1]
        batchsize=predindex.size()[0]
        #loss = self.loss(self.mask, pred)
        # self.calIOU.add_batch(predindex.cpu().numpy(), self.mask.cpu().numpy())
        # batch_miou, oa, kappa = self.calIOU.evaluate()


        loss1 = nn.CrossEntropyLoss(ignore_index=255)
        loss = loss1(pred, torch.squeeze(self.mask).long())
        loss.backward()
        self.optimizer.step()
        # self.mask=self.mask.cpu()
        # predindex=predindex.cpu()
        # for i in range(batchsize):
        #     oa += accuracy_score(torch.squeeze(self.mask)[i].cpu().numpy().flatten(), predindex[i].cpu().flatten())
        #     kappa += cohen_kappa_score(torch.squeeze(self.mask)[i].cpu().numpy().flatten(), predindex[i].cpu().flatten())
        # return loss.item(), batch_miou
        return loss.item()

    def testmodel(self):
        self.forward()
        self.net.eval()
        with torch.no_grad():
            pred = self.net.forward(self.img)
        predindex = pred.max(dim=1)[1]
        batchsize=predindex.size()[0]
        intersection, union, target, output = self.intersectionAndUnionGPU(predindex, self.mask.long(), K=6)

        # self.calIOU.add_batch(predindex.cpu().numpy(), self.mask.cpu().numpy())
        # batch_miou, oa, kappa = self.calIOU.evaluate()
        # for i in range(batchsize):
        #     oa += accuracy_score(torch.squeeze(self.mask)[i].cpu().numpy().flatten(), predindex[i].cpu().flatten())
        #     kappa += cohen_kappa_score(torch.squeeze(self.mask)[i].cpu().numpy().flatten(), predindex[i].cpu().flatten())
        return intersection, union, target, output

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # print('update learning rate: %f -> %f' % (self.old_lr, new_lr),file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def intersectionAndUnionGPU(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        # https://github.com/pytorch/pytorch/issues/1382
        area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
        area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
        area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return area_intersection.cuda(), area_union.cuda(), area_target.cuda(), area_output.cuda()