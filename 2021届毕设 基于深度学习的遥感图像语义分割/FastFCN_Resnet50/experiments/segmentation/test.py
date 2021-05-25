###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torchvision.transforms as transform
import cv2
import encoding.utils as utils
import numpy as np
from tqdm import tqdm
import time
from torch.utils import data
from PIL import Image
from encoding.nn import BatchNorm
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options



def test(args):
    # output folder
    graydir = os.path.join(args.save_folder, 'gray')
    colordir = os.path.join(args.save_folder, 'color')
    scoredir = os.path.join(args.save_folder, 'score')
    if not os.path.exists(graydir):
        os.makedirs(graydir)

    if not os.path.exists(colordir):
        os.makedirs(colordir)

    if not os.path.exists(scoredir):
        os.makedirs(scoredir)

    colors = np.loadtxt('./postdam_colors.txt').astype('uint8')

    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    testset = get_segmentation_dataset(args.dataset, split=args.split, mode=args.mode,
                                       transform=input_transform) # split=val mode=test
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=1,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm,
                                       base_size = args.base_size, crop_size = args.crop_size)
        # resuming checkpoint
        if args.model_path is None or not os.path.isfile(args.model_path):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.model_path))
        checkpoint = torch.load(args.model_path)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))

    # print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [ 0.75, 1.0, 1.25, ]
    if not args.ms:
        scales = [1.0]

    # model = torch.nn.DataParallel(model).cuda()
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()
    evaluator.eval()
    metric = utils.SegmentationMetric(testset.num_class)
    average_time = []


    zongji=0
    tbar = tqdm(test_data)
    for i, (image, dst) in enumerate(tbar):
        # 原始测试
        if 'val' in args.mode:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                metric.update(dst, predicts)
                pixAcc, mIoU, kappa = metric.get()
                tbar.set_description('pixAcc: %.4f, mIoU: %.4f, kappa: %.4f' % (pixAcc, mIoU, kappa))

        else:
            with torch.no_grad():
                outputs = evaluator.parallel_forward(image)
                predicts = [torch.max(output, 1)[1].cpu().numpy()
                            for output in outputs]
                scores = [torch.max(output, 1)[0].cpu().numpy()
                          for output in outputs]
            for predict, score, impath in zip(predicts, scores, dst):
                # mask = utils.get_mask_pallete(predict, args.dataset)
                predict = np.squeeze(predict)
                outname = os.path.splitext(impath)[0] + '_pred.png'
                scorename = os.path.splitext(impath)[0] + '_pred_score.npy'
                colorname = os.path.splitext(impath)[0] + '_color.png'
                graypath = os.path.join(graydir, outname)
                scorepath = os.path.join(scoredir, scorename)
                colorpath = os.path.join(colordir, colorname)
                cv2.imwrite(graypath, np.array(predict))
                np.save(scorepath, np.squeeze(score))
                color = colorize(predict, colors)
                color.save(colorpath)
                # mask.save(os.path.join(outdir, outname))



        # model.eval() # 测试模式
        # if 'val' in args.mode:
        #     with torch.no_grad():
        #         start=time.time()
        #         image = image[0].unsqueeze(0).cuda()
        #         predicts = model(image)
        #         use_time = time.time() - start
        #         average_time.append(use_time)
        #         metric.update(dst[0], predicts[0])
        #         pixAcc, mIoU, kappa = metric.get()
        #         tbar.set_description('pixAcc: %.4f, mIoU: %.4f, kappa: %.4f' % (pixAcc, mIoU, kappa))
        # else:
        #     with torch.no_grad():
        #         start = time.time()
        #         image = image[0].unsqueeze(0).cuda()  # 1*C*H*W
        #         outputs = model(image)
        #         use_time = time.time() - start
        #         average_time.append(use_time)
        #         predicts = [torch.max(output, 1)[1].cpu().numpy()
        #                     for output in outputs]
        #         scores = [torch.max(output, 1)[0].cpu().numpy()
        #                   for output in outputs]  # 取概率
        #     for predict, score, impath in zip(predicts, scores, dst):
        #
        #         predict = np.squeeze(predict) # 512*512
        #
        #         outname = os.path.splitext(impath)[0] + '_pred.png'
        #
        #         colorname = os.path.splitext(impath)[0] + '_color.png'
        #         scorename = os.path.splitext(impath)[0] + '_pred_score.npy'
        #         graypath = os.path.join(graydir, outname)
        #         colorpath = os.path.join(colordir, colorname)
        #         scorepath = os.path.join(scoredir, scorename)
        #         cv2.imwrite(graypath, np.array(predict))
        #         color = colorize(predict, colors)
        #         color.save(colorpath)
        #         np.save(scorepath, np.squeeze(score))


    # print('Average_time:{}'.format(sum(average_time) / len(average_time)))
    print('Finish!')


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    # args.test_batch_size = 16
    test(args)
