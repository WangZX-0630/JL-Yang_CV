from __future__ import print_function

import argparse
import json
import os
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from model.models import MODELS
from road_dataset_convert import DeepGlobeDataset, SpacenetDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from utils.loss import CrossEntropyLoss2d, mIoULoss
from utils import util
from utils import viz_util
import csv


__dataset__ = {"spacenet": SpacenetDataset, "deepglobe": DeepGlobeDataset}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", required=True, type=str, help="config file path"
)
parser.add_argument(
    "--model_name",
    required=True,
    choices=sorted(MODELS.keys()),
    help="Name of Model = {}".format(MODELS.keys()),
)
parser.add_argument("--exp", required=True, type=str, help="Experiment Name/Directory")
parser.add_argument(
    "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
)
parser.add_argument(
    "--dataset",
    required=True,
    choices=sorted(__dataset__.keys()),
    help="select dataset name from {}. (default: Spacenet)".format(__dataset__.keys()),
)
parser.add_argument(
    "--model_kwargs",
    default={},
    type=json.loads,
    help="parameters for the model",
)
parser.add_argument(
    "--multi_scale_pred",
    default=True,
    type=util.str2bool,
    help="perform multi-scale prediction (default: True)",
)

args = parser.parse_args()
config = None

if args.resume is not None:
    if args.config is not None:
        print("Warning: --config overridden by --resume")
        config = torch.load(args.resume)["config"]
elif args.config is not None:
    config = json.load(open(args.config))

assert config is not None

util.setSeed(config)

experiment_dir = os.path.join(config["trainer"]["save_dir"], args.exp)
util.ensure_dir(experiment_dir)

###Logging Files
# train_file = "{}/{}_train_loss.txt".format(experiment_dir, args.dataset)
# test_file = "{}/{}_test_loss.txt".format(experiment_dir, args.dataset)

# train_loss_file = open(train_file, "w", 1)
# val_loss_file = open(test_file, "w", 1)

# ### Angle Metrics
# train_file_angle = "{}/{}_train_angle_loss.txt".format(experiment_dir, args.dataset)
# test_file_angle = "{}/{}_test_angle_loss.txt".format(experiment_dir, args.dataset)

# train_loss_angle_file = open(train_file_angle, "w", 1)
# val_loss_angle_file = open(test_file_angle, "w", 1)
################################################################################
# num_gpus = torch.cuda.device_count()

# model = MODELS[args.model_name](
#     config["task1_classes"], config["task2_classes"], **args.model_kwargs
# )

# if num_gpus > 1:
#     print("Training with multiple GPUs ({})".format(num_gpus))
#     model = nn.DataParallel(model).cuda()
# else:
#     print("Single Cuda Node is avaiable")
#     model.cuda()
################################################################################

### Load Dataset from root folder and intialize DataLoader
train_loader = data.DataLoader(
    __dataset__[args.dataset](
        config["train_dataset"],
        seed=config["seed"],
        is_train=True,
        multi_scale_pred=args.multi_scale_pred,
    ),
    batch_size=config["train_batch_size"],
    num_workers=8,
    shuffle=True,
    pin_memory=False,
)

val_loader = data.DataLoader(
    __dataset__[args.dataset](
        config["val_dataset"],
        seed=config["seed"],
        is_train=False,
        multi_scale_pred=args.multi_scale_pred,
    ),
    batch_size=config["val_batch_size"],
    num_workers=8,
    shuffle=True,
    pin_memory=False,
)

print("Training with dataset => {}".format(train_loader.dataset.__class__.__name__))
################################################################################

# best_accuracy = 0
# best_miou = 0
# start_epoch = 1
# total_epochs = config["trainer"]["total_epochs"]
# # optimizer = optim.SGD(
# #     model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9, weight_decay=0.0005
# # )

# if args.resume is not None:
#     print("Loading from existing FCN and copying weights to continue....")
#     checkpoint = torch.load(args.resume)
#     start_epoch = checkpoint["epoch"] + 1
#     best_miou = checkpoint["miou"]
#     # stat_parallel_dict = util.getParllelNetworkStateDict(checkpoint['state_dict'])
#     # model.load_state_dict(stat_parallel_dict)
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
# else:
#     util.weights_init(model, manual_seed=config["seed"])

# viz_util.summary(model, print_arch=False)

# scheduler = MultiStepLR(
#     optimizer,
#     milestones=eval(config["optimizer"]["lr_drop_epoch"]),
#     gamma=config["optimizer"]["lr_step"],
# )


# weights = torch.ones(config["task1_classes"]).cuda()
# if config["task1_weight"] < 1:
#     print("Roads are weighted.")
#     weights[0] = 1 - config["task1_weight"]
#     weights[1] = config["task1_weight"]


# weights_angles = torch.ones(config["task2_classes"]).cuda()
# if config["task2_weight"] < 1:
#     print("Road angles are weighted.")
#     weights_angles[-1] = config["task2_weight"]


# angle_loss = CrossEntropyLoss2d(
#     weight=weights_angles, size_average=True, ignore_index=255, reduce=True
# ).cuda()
# road_loss = mIoULoss(
#     weight=weights, size_average=True, n_classes=config["task1_classes"]
# ).cuda()

f = open('103201_mask_pred.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(['ImageId','WKT_Pix'])

for i, data in enumerate(val_loader, 0):
    linestring, image_name = data
    
    for val in linestring:
#         print(image_name[0][30:-4]+'_mask')
#         print(val)
        csv_writer.writerow([image_name[0][34:-4]+'_mask',val[0]])
    
#     print(linestring)
#     print(image_name[0][30:-4])
    

