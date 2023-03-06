# %% CELL (CREATE PROTONET, DATASET)

import os
import time
import sys
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib import ticker
import models.protonet as protonet
from models.util.dataset import CarLicensePlateDataset
from torch.utils.data import DataLoader

sidelen = 192
out_features = 4

model = protonet.ProtoNet(sidelen, out_features)
# weights_sample = model.sample_weights(16)

print('created ProtoNet')

# image_ids = []
# label_mappings = {}
# dataset = StandardDataset(ids=image_ids, labels=label_mappings, dataset_dir=TENSORS_DIR)
dataset = CarLicensePlateDataset()

batch_size=32 # TODO
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print('loaded dataset')

# %% CELL (TRAIN)

lr = 5e-4
# num_workers=2 # TODO
num_epochs=64 # TODO: 1024
# log_freq = 4 # TODO

optimizer = optim.Adam(params=model.parameters(), lr=lr)

start = time.time()
losses_arr = []

model.train()
for epoch in range(num_epochs):
    total_losses = 0.0
    batch_n = 0

    for inputs, bb_targets in dataloader:
        bb_targets = [t.view(-1,1) for t in bb_targets]
        bb_targets = torch.cat(bb_targets, 1)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            bb_outputs = model(inputs)

            loss = F.l1_loss(bb_outputs, bb_targets, reduction="none").sum(1)
            loss = loss.sum()
            # print('loss', loss.item())

            loss.backward()
            optimizer.step()

        total_losses += loss.item() * inputs.size(0)

        batch_n += 1
        # if batch_n % log_freq == 0:
        #     iteration = epoch * len(dataloader) + batch_n
        #     print(f'iter {iteration} \t loss {losses :.3f}')
        #     losses = 0.0

    print(f'epoch {epoch} \t loss {total_losses / batch_size:.3f} ')

    # figsize=(6.4, 4.8)
    # dpi=100
    # facecolor='white'
    # fig = plt.figure()
    # ax = fig.axes()
    losses_arr.append(total_losses / batch_size)
    plt.plot(losses_arr, 'ko')
    plt.xlim(-1, epoch+1)
    plt.ylim(0, int(max(losses_arr) * 1.1) )
    plt.show(block=True)

end = time.time()
print(f'it took {end - start:.4f} seconds')

print('done')

# %% CELL (VALIDATE)

from matplotlib import ticker

print(type(dataloader.sampler))

for inputs, bb_targets in dataloader:
    bb_targets = [t.view(-1, 1) for t in bb_targets]
    bb_targets = torch.cat(bb_targets, 1)
    print(bb_targets.shape)

    bb_outputs = model(inputs)
    bb_outputs = bb_outputs.detach()

    for i in range(8):
        tensor = inputs[i, :, :, :]
        plt.imshow(tensor.permute((1, 2, 0)).numpy())
        plt.gca().xaxis.set_major_locator(ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(ticker.NullLocator())
        print(i)

        bb = bb_targets[i, :]
        highlight_rect = plt.Rectangle((bb[2], bb[0]), bb[3]-bb[2], bb[1]-bb[0], color='red', fill=False, lw=2)
        plt.gca().add_patch(highlight_rect)

        bb = bb_outputs[i, :]
        highlight_rect = plt.Rectangle((bb[2], bb[0]), bb[3]-bb[2], bb[1]-bb[0], color='blue', fill=False, lw=2)
        plt.gca().add_patch(highlight_rect)

        print(bb_targets[i, :])
        print(bb_outputs[i, :])

        plt.show()

    # Validate one batch
    break

# %%
