# %% CELL (CHECK DATASET)

import glob
import os
import sys
import torch

import matplotlib.pyplot as plt
from matplotlib import ticker
from skimage import io

device = 'cpu'
PARK_ROOT_DIR = '/Users/ericcarlson/Desktop/Projects/park'
KITTI_DIR = '/Users/ericcarlson/Desktop/Datasets/data_semantics/training'

if len(sys.argv) == 4:
    device = sys.argv[1]
    PARK_ROOT_DIR = sys.argv[2]
    KITTI_DIR = sys.argv[3]
else:
    print("Defaults will be used for device, base directory, and dataset directory.")

KITTI_INSTANCE_DIR = os.path.join(KITTI_DIR, 'instance')

count = 0
max_count = 1
for fname in glob.iglob(os.path.join(KITTI_INSTANCE_DIR, '*')):
    instance_semantic = io.imread(fname)
    plt.imshow(instance_semantic)
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
    
    plt.show()

    instance = instance_semantic  % 256
    semantic = instance_semantic // 256
    plt.subplot(1, 2, 1)
    plt.imshow(instance)
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())

    plt.subplot(1, 2, 2)
    plt.imshow(semantic)
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())

    plt.tight_layout()
    plt.show()

    count += 1
    if count > max_count:
        break

# %% CELL (CREATE MODEL TO FINE-TUNE)

import math
import time

from detection import engine as det_engine, utils as det_utils
from models.util.dataset import KittiDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN
# TODO: mobilenet_v3
from torchvision.models import mobilenet_v2
from torch import optim
from torch.utils.data import DataLoader

def mrcnn_to_finetune(n_classes: int) -> torch.nn.Module:
    out_channels = 1280
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = out_channels
    
    for n, p in backbone.named_parameters():
        p.requires_grad = '17' in n or '18' in n

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    # NOTE: default 'anchor_generator' produces an error in 'anchor_utils'
    ret = MaskRCNN(backbone=backbone, num_classes=n_classes, 
                   rpn_anchor_generator=anchor_generator)

    in_features_box_pred = ret.roi_heads.box_predictor.cls_score.in_features
    ret.roi_heads.box_predictor = FastRCNNPredictor(in_features_box_pred, n_classes)

    in_features_mask = ret.roi_heads.mask_predictor.conv5_mask.in_channels
    n_hidden = 256 # TODO: CONFIGURE
    ret.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, n_hidden, n_classes)

    return ret

dataset = KittiDataset(KITTI_DIR)

dataloader_train = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=det_utils.collate_fn)
dataloader_val = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=det_utils.collate_fn)

n_classes = 33
model = mrcnn_to_finetune(n_classes)
model.to(device)

# %% CELL (TRAIN)

lr = 1e-3
weight_decay = 5e-4
optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=weight_decay)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

n_epochs = 1

logged_time = time.time()
model.train()
for epoch in range(n_epochs):
    print('epoch', epoch)
    epoch_loss = 0.0
    
    for i, (inputs, targets) in enumerate(dataloader_train):
        # NOTE: Each iteration will take around one minute on a Macbook with 16 GB RAM!
        print('i', i)
        inputs = list(input.to(device) for input in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(inputs, targets)
        # for k in loss_dict:
        #     print(k, '\t', loss_dict[k].item())
        losses = sum(loss for loss in loss_dict.values())
        loss_val = losses.item()
        epoch_loss += loss_val

        print(loss_val)

        if not math.isfinite(loss_val):
            print(f"Loss is {loss_val}, stopping training")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # if lr_scheduler is not None:
        #     lr_scheduler.step()
    
    print(f'epoch {epoch} loss \t {epoch_loss:.4f}')

logged_time = time.time() - logged_time
print('total_time', logged_time)

# %% CELL (SAVE MODEL)

torch.save(model, 'mobilenet-mrcnn-kitti-transfer.pt')

# %% CELL (EVALUATE)

model.eval()
det_engine.evaluate(model, dataloader_val, device)

# %% CELL (VISUALIZE EVALUATION)

import matplotlib.pyplot as plt

model.eval()
for inputs, targets in dataloader_train:
    inputs = list(input.to(device) for input in inputs)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    outputs = model(inputs)

    for i, (target, output) in enumerate(zip(targets, outputs)):
        print(i)
        tensor = inputs[i]
        plt.imshow(tensor.permute((1, 2, 0)).numpy())
        plt.gca().xaxis.set_major_locator(ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(ticker.NullLocator())

        for bb in target['boxes']:
            highlight_rect = plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color='red', fill=False, lw=2)
            plt.gca().add_patch(highlight_rect)

        max_bb_count = 4
        bb_count = 0
        print(len(output['boxes']), 'boxes')
        for bb in output['boxes']:
            bb = bb.detach()
            print(bb)
            highlight_rect = plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color='blue', fill=False, lw=2)
            plt.gca().add_patch(highlight_rect)

            bb_count += 1
            if bb_count >= max_bb_count:
                break

        plt.show()

    break

# TODO: Is there a similar functionality in pytorch?
# visualize.display_instances(image, p['rois'], p['masks'], p['class_ids'], class_names, p['scores'])
