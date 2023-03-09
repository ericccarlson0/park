# %% CELL (CHECK DATASET)

import coco
import glob
import mrcnn.model as mrcnnmodel
import os
import torch

import matplotlib.pyplot as plt
from matplotlib import ticker
from skimage import io
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
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

# %% CELL (CREATE M-RCNN MODEL TO FINE-TUNE)

import math
import time

from detection import engine as det_engine
from detection import transforms as det_transforms
from detection import utils as det_utils
from models.util.dataset import PennFudanDataset
from mrcnn.config import Config
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
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
# for s, p in torch_model_pretrained.named_parameters():
#     if 'rpn.head.cls_logits' in s or 'rpn.head.bbox_pred' in s or 'roi_heads.box_head.5' in s \
#             or 'roi_heads.box_predictor' in s or 'roi_heads.mask_predictor.mask_fcn_logits' in s:
#         print(s)
#         print(p.shape)

def mrcnn_to_finetune(n_classes: int) -> torch.nn.Module:
    # torch_model_pretrained = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    ret = maskrcnn_resnet50_fpn_v2(pretrained=True)
    in_features_box_pred = ret.roi_heads.box_predictor.cls_score.in_features
    ret.roi_heads.box_predictor = FastRCNNPredictor(in_features_box_pred, n_classes)

    in_features_mask = ret.roi_heads.mask_predictor.conv5_mask.in_channels
    n_hidden = 256 # TODO: CONFIGURE
    ret.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, n_hidden, n_classes)

    return ret

PENN_FUDAN_DATASET_DIR = "/Users/ericcarlson/Desktop/Datasets/PennFudanPed" # TODO
dataset = PennFudanDataset(PENN_FUDAN_DATASET_DIR)

indices = torch.randperm(len(dataset)).tolist()

dataloader_train = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=det_utils.collate_fn)
dataloader_val = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=det_utils.collate_fn)

n_classes = 2
model = mrcnn_to_finetune(n_classes)

if train:
    lr = 5e-3
    weight_decay = 5e-4
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=weight_decay)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # for inputs, targets in dataloader_train:
    #     print('inputs', len(inputs))
    #     for input in inputs:
    #         print input
    #     break

    n_epochs = 1 # TODO

    t0 = time.time()
    for epoch in range(n_epochs):
        print('epoch', epoch)
        # THIS INSTEAD OF det_engine.train_one_epoch
        for i, (inputs, targets) in enumerate(dataloader_train):
            # NOTE: Each iteration will take around one minute on a Macbook with 16 GB RAM!
            print('i', i)

            loss_dict = model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_val = losses.item()

            if not math.isfinite(loss_val):
                print(f"Loss is {loss_val}, stopping training")

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # if lr_scheduler is not None:
            #     lr_scheduler.step()

    total_time = time.time() - t0
    print('total_time', total_time)
    torch.save(model, 'mrcnn-penn-fudan-transfer.pt')
else:
    model = torch.load(os.path.join(PARK_ROOT_DIR, 'log/mrcnn-penn-fudan-transfer.pt'))

# %% CELL (EVALUATE)

det_engine.evaluate(model, dataloader_val, 'cpu')

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
