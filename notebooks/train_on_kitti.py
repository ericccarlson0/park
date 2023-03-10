# %% CELL (CHECK DATASET)

import coco
import glob
import mrcnn.model as mrcnnmodel
import os
import torch

import matplotlib.pyplot as plt
from matplotlib import ticker
from skimage import io
from torchvision.models import ResNet50_Weights

# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

SEMANTICS_DATA_DIR = '/Users/ericcarlson/Desktop/Datasets/data_semantics/training'
INSTANCE_DIR = os.path.join(SEMANTICS_DATA_DIR, 'instance')

count = 0
max_count = 4
for fname in glob.iglob(os.path.join(INSTANCE_DIR, '*')):
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

from detection import engine as det_engine
from detection import transforms as det_transforms
from detection import utils as det_utils
from models.util.dataset import PennFudanDataset
from mrcnn.config import Config
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN
# TODO: mobilenet_v3
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch import optim
from torch.utils.data import DataLoader

train = False # TODO: REMOVE

PARK_ROOT_DIR = '/Users/ericcarlson/Desktop/Projects/park'
RCNN_ROOT_DIR = '/Users/ericcarlson/Desktop/Projects/mask-rcnn'
SAVE_MODEL_DIR = os.path.join(RCNN_ROOT_DIR, 'log')
COCO_RCNN_MODEL_PATH = os.path.join(RCNN_ROOT_DIR, 'mask_rcnn_coco.h5')
# if not os.path.exists(COCO_RCNN_MODEL_PATH):
#     utils.download_trained_weights(COCO_RCNN_MODEL_PATH)
IMAGE_DIR = os.path.join(RCNN_ROOT_DIR, "images")

# trainable_backbone_layers=3
# torch_model_pretrained = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)

def mrcnn_to_finetune(n_classes: int) -> torch.nn.Module:
    out_channels = 1280
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = out_channels
    
    for n, p in backbone.named_parameters():
        p.requires_grad = '17' in n or '18' in n
    # for n, p in backbone.named_parameters():
    #     print(n, p.requires_grad)

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # NOTE: default 'anchor_generator' produces an error in 'anchor_utils'
    ret = MaskRCNN(backbone=backbone, num_classes=n_classes, 
                   rpn_anchor_generator=anchor_generator)
    # ret = maskrcnn_resnet50_fpn_v2(pretrained=True)

    # NOTE: 'box_head' INPUT IS 'box_roi_pool.output_size[0] ** 2'
    #   (or, 'resolution ** 2')
    in_features_box_pred = ret.roi_heads.box_predictor.cls_score.in_features
    # print('in_features_box_pred', in_features_box_pred)
    ret.roi_heads.box_predictor = FastRCNNPredictor(in_features_box_pred, n_classes)

    in_features_mask = ret.roi_heads.mask_predictor.conv5_mask.in_channels
    # print('in_features_mask', in_features_mask)
    n_hidden = 256 # TODO: CONFIGURE
    ret.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, n_hidden, n_classes)

    return ret

PENN_FUDAN_DATASET_DIR = "/Users/ericcarlson/Desktop/Datasets/PennFudanPed" # TODO
dataset = PennFudanDataset(PENN_FUDAN_DATASET_DIR)

indices = torch.randperm(len(dataset)).tolist()

dataloader_train = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=det_utils.collate_fn)
dataloader_val = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=det_utils.collate_fn)

n_classes = 2
model = mrcnn_to_finetune(n_classes)

# %% CELL (LOAD MODEL)

model = torch.load(os.path.join(PARK_ROOT_DIR, 'log/mobilenet-mrcnn-penn-fudan-transfer.pt'))

# %% CELL (TRAIN)

lr = 5e-3
weight_decay = 5e-4
optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=weight_decay)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# for inputs, targets in dataloader_train:
#     print('inputs', len(inputs))
#     for input in inputs:
#         print input
#     break

n_epochs = 4 # TODO

t0 = time.time()
model.train()
for epoch in range(n_epochs):
    print('epoch', epoch)
    # THIS INSTEAD OF det_engine.train_one_epoch
    for i, (inputs, targets) in enumerate(dataloader_train):
        # NOTE: Each iteration will take around one minute on a Macbook with 16 GB RAM!
        print('i', i)

        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_val = losses.item()

        print(loss_val)

        if not math.isfinite(loss_val):
            print(f"Loss is {loss_val}, stopping training")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # if lr_scheduler is not None:
        #     lr_scheduler.step()

total_time = time.time() - t0
print('total_time', total_time)

# %% CELL (SAVE MODEL)

torch.save(model, 'mobilenet-mrcnn-penn-fudan-transfer.pt')

#   model = torch.load(os.path.join(PARK_ROOT_DIR, 'log/mrcnn-penn-fudan-transfer.pt'))

# %% CELL (EVALUATE)

det_engine.evaluate(model, dataloader_val, 'cpu')

# %% CELL (VISUALIZE EVALUATION)

import matplotlib.pyplot as plt

model.eval()
for inputs, targets in dataloader_train:
    outputs = model(inputs)
    outputs = outputs[0]
    print('boxes', len(outputs['boxes']))
    # TODO: outputs.detach()

    for i in range(len(inputs)):
        target = targets[i]
        # 'boxes'
        # 'labels'
        # 'masks'
        # 'image_id'
        # 'iscrowd'
        # 'area'

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
        for bb in outputs['boxes']:
            bb = bb.detach()
            print(bb)
            highlight_rect = plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color='blue', fill=False, lw=2)
            plt.gca().add_patch(highlight_rect)

            bb_count += 1
            if bb_count >= max_bb_count:
                break

        plt.show()

    break

# %% CELL (MRCNN WITH TF)

import random
from mrcnn import visualize

class KittiConfig(Config):
    NAME = "kitti"
    BATCH_SIZE = 16
    NUM_CLASSES = 5 # TODO

config = KittiConfig()
config.display()

class CocoInferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

tf_model_pretrained = mrcnnmodel.MaskRCNN(mode="inference", model_dir=SAVE_MODEL_DIR, config=CocoInferenceConfig())

tf_model_pretrained.load_weights(COCO_RCNN_MODEL_PATH, by_name=True)

# keras.models.save_model(model_coco_pretrained.keras_model, 'mask_rcnn_coco_full.h5')
# model_coco_pretrained.keras_model.save(os.path.join(RCNN_ROOT_DIR, 'mask_rcnn_coco_full.h5'))

# TODO?
# weights = tf_model_pretrained.get_weights()

# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

fnames = next(os.walk(IMAGE_DIR))[2]
image = io.imread(os.path.join(IMAGE_DIR, random.choice(fnames)))

res = tf_model_pretrained.detect([image], verbose=1)
p = res[0]
visualize.display_instances(image, p['rois'], p['masks'], p['class_ids'], class_names, p['scores'])

# %%
