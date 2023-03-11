# %% CELL (CREATE MODEL TO FINE-TUNE)

import math
import sys
import torch
import time

from detection import engine as det_engine, utils as det_utils
from models.util.dataset import PennFudanDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torchvision.models import mobilenet_v2
from torch import optim
from torch.utils.data import DataLoader

if len(sys.argv) != 4:
    raise ValueError(f"There need to be three additional arguments,\npassed {sys.argv[1:]}).")

device = sys.argv[1]
PARK_ROOT_DIR = sys.argv[2]
# '/Users/ericcarlson/Desktop/Projects/park'
PENN_FUDAN_DATASET_DIR = sys.argv[3]
# "/Users/ericcarlson/Desktop/Datasets/PennFudanPed"

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
    n_hidden = 256
    ret.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, n_hidden, n_classes)

    return ret

dataset = PennFudanDataset(PENN_FUDAN_DATASET_DIR)

indices = torch.randperm(len(dataset)).tolist()

dataloader_train = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=det_utils.collate_fn)
dataloader_val = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=det_utils.collate_fn)

n_classes = 2
model = mrcnn_to_finetune(n_classes)
model.to(device)

# %% CELL (CHECK DEVICES)

print('cuda is available?', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print('device zero', torch.cuda.get_device_name(0))

# %% CELL (TRAIN)

lr = 5e-3
weight_decay = 5e-4
optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=weight_decay)

n_epochs = 16

logged_time = time.time()
model.train()
for epoch in range(n_epochs):
    print('epoch', epoch)
    epoch_loss = 0.0
    
    for i, (inputs, targets) in enumerate(dataloader_train):
        print('i', i)
        inputs = list(input.to(device) for input in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict = model(inputs, targets)
        # for k in loss_dict:
        #     print(k, '\t', loss_dict[k].item())
        losses = sum(loss for loss in loss_dict.values())
        loss_val = losses.item()
        epoch_loss += loss_val

        if not math.isfinite(loss_val):
            print(f"Loss is {loss_val}, stopping training")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f'epoch {epoch} loss \t {epoch_loss:.4f}')

logged_time = time.time() - logged_time
print('logged time', logged_time)

# %% CELL (SAVE MODEL)

torch.save(model, 'mobilenet-mrcnn-penn-fudan-finetune.pt')

# %% CELL (EVALUATE)

model.eval()
det_engine.evaluate(model, dataloader_val, device)

# %%
