# NOTE: This pipeline is only a place to look and copy, as of now.
# It is a copy, with small modifications, from an older project.

# %% CELL (START)
import os
import sys
import time
import torch
import torchvision
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models.shufflenet as shufflenet
# import models.mobilenet as mobilenet
from models.util.dataset import StandardDataset

from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

# %% CELL (SET UP DIRECTORIES)

BASE_DIR = os.sep + os.path.join("Users", "ericcarlson", "Desktop", "Projects")

LOCAL_DIR = os.path.join(BASE_DIR, "park")

TENSORS_DIR = os.path.join(BASE_DIR, "Datasets", "torch_data") # TODO: REMOVE

CSV_DIR = os.path.join(BASE_DIR, "Datasets", "csv", "ITD.csv") # TODO: REMOVE

TB_LOGDIR = os.path.join(LOCAL_DIR, "tensorboard")

SAVED_MODEL_DIR = os.path.join(LOCAL_DIR, "trained")

if not BASE_DIR:
    raise NotADirectoryError("Directories need to be set up.") # TODO

# %% CELL (RETRIEVE IMAGE IDS AND TENSORS)

image_ids = []
label_mappings = {}
csv_dataset = pd.read_csv(CSV_DIR)

# TODO

# %% CELL (DIVIDE IMAGE IDS INTO TRAIN/VAL/TEST)

train_ids, val_ids = train_test_split(image_ids, test_size=.10)
val_ids, test_ids = train_test_split(val_ids, test_size=0.50)
print('data', len(train_ids), len(val_ids), len(test_ids))

# %% CELL (SET UP TRAINING PARAMETERS)

lr = 1e-4
lr_decay = 0.95     # TODO: ONLY LATER
dropout_prob = 0.50 # TODO: ONLY LATER
prune_prob = 0.10   # TODO: ONLY LATER
prune_mod = sys.maxsize

batch_size = 128
num_workers = 2 # TODO: WHY?
num_classes = 2
num_epochs = 4 # TODO: INCREASE

state_dict_dir = None
pretrain = False
train_model = True
save_model = False

# TODO: MOVE
model_name = "shufflenet"

summary_writer = SummaryWriter(TB_LOGDIR)
hparams = {
    "MODEL_NAME": model_name, 
    "LEARNING_RATE": lr
} # "DROPOUT_PROB": dropout_prob}

# %% CELL (CREATE OR LOAD MODEL)

model = shufflenet.shufflenet_small()

if state_dict_dir:
    model.load_state_dict(torch.load(state_dict_dir))

# %% CELL (LOSS CRITERION, OPTIMIZER, LR SCHEDULER)

criterion = nn.CrossEntropyLoss()
# TODO: betas, AMS Grad, etc. LATER
optimizer = optim.Adam(params=model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

# %% CELL (PREPROCESSING, DATASET, DATALOADER)

# NOTE: NORMALIZATION FROM IMAGENET
normalize = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# NOTE: AUGMENT DATA BEFOREHAND
preprocess = None

# TODO: tsfm?
datasets = {
    'train': StandardDataset(train_ids, label_mappings, transform=preprocess, dataset_dir=TENSORS_DIR),
    'val': StandardDataset(val_ids, label_mappings, transform=preprocess, dataset_dir=TENSORS_DIR)
}
dataloaders = {
    x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for x in ['train', 'val']
}
dataset_sizes = {
    x : len(datasets[x])
    for x in ['train', 'val']
}

# %% CELL (CHECK IMAGES FROM DATALOADER)

inputs, labels = next(iter(dataloaders(['train'])))
# TODO
grid = torchvision.utils.make_grid(inputs, nrow=int(np.sqrt(batch_size)))
fig, ax = plt.subplots(1, figsize=(10, 10))

# TODO: "show_tensor"
grid = grid.numpy().transpose((1, 2, 0))
# grid = np.clip(image, 0, 1)
ax.imshow(grid)
ax.set_xticks([])
ax.set_yticks([])

plt.show()

# %% CELL (TRAIN MODEL)

# train(model=model,
#       criterion=criterion,
#       optimizer=optimizer,
#       scheduler=lr_scheduler,
#       loaders=dataloaders,
#       sizes=dataset_sizes,
#       writer=summary_writer,
#       num_epochs=num_epochs)

device = 'cpu'
log_freq = 64

start = time.time()

for epoch in range(num_epochs):
    print('epoch', epoch)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        losses = 0.0
        # corrects = 0.0
        batch_c = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                # TODO: ???
                loss = criterion(outputs, labels.long())
                # _, preds = torch.max(outputs, 1)
                # corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            batch_c += 1
            if batch_c % log_freq == 0:
                iteration = epoch * len(dataloaders[phase]) + batch_c
                # write_loss(summary_writer, losses, batch_c, iteration, phase)
                summary_writer.add_scalar(f'{phase} loss', losses / log_freq, iteration)
                losses = 0.0
            
            losses += loss.item() * inputs.size(0)

        # if phase == 'train' and lr_scheduler is not None:
        #    lr_scheduler.step()

        # losses = losses / dataset_sizes[phase]
        # accuracy = corrects / dataset_sizes[phase]
        print('phase', phase)
    
    print(f'{epoch} -> {epoch+1}')

# %% CELL (SAVE MODEL)

curr_time = float(time.time())
saved_model_fname = f'{model_name}_t{curr_time: .2f}.pt'
state_dict_path = os.path.join(SAVED_MODEL_DIR, saved_model_fname)

# TODO
# torch.save(model.state_dict(), state_dict_path)

# %% CELL (RECORD RESULTS)

val_acc = 0
test_acc = 0
total_acc = 0
metrics = {
    "hparam/val_accuracy": val_acc,
    "hparam/test_accuracy": test_acc,
    "hparam/total_accuracy": total_acc
}
summary_writer.add_hparams(hparam_dict=hparams, matric_dict=metrics)
summary_writer.close()

# CELL (TRACE MODEL)

# TODO: A TRACED MODEL ON A FIXED INPUT IS ALMOST CERTAINLY NOT ENOUGH
ex_input = torch.rand((2, 3, 244, 244))
# traced_model = torch.jit.trace(model, ex_input)
