from cv2 import resize
import numpy as np
import torch
import random

from typing import List
import matplotlib.pyplot as plt

SIZE = 192

# X:    tensor to mask
# bb:   [xmin, xmax, ymin, ymax]
def create_mask(X: torch.Tensor, bb: List):
    _, w, h = X.shape
    Y = np.zeros((w, h))

    # print(f'{bb[0]} : {bb[1]}', f'{bb[2]} : {bb[3]}')
    Y[bb[0]:bb[1], bb[2]:bb[3]] = 1

    cols, rows = np.nonzero(Y)

    return Y

# Turn a mask into a bounding-box.
def mask_bb(Y):
    cols, rows = np.nonzero(Y)
    if len(cols) == 0 or len(rows) == 0:
        raise Exception('There should be one or more nonzero pixels in the mask.')
    
    min_col, max_col = np.min(cols), np.max(cols)
    min_row, max_row = np.min(rows), np.max(rows)

    return [min_col, max_col, min_row, max_row]

def resize_with_bb(tensor: torch.Tensor, bb: List):
    ndarray = tensor.permute((1, 2, 0)).numpy()
    ndarray_resized = resize(ndarray, (SIZE, SIZE))
    Y = create_mask(tensor, bb)
    Y_resized = resize(Y, (SIZE, SIZE))
    bb_resized = mask_bb(Y_resized)

    return torch.Tensor(ndarray_resized).permute(2, 0, 1), bb_resized

def random_crop_with_bb(tensor: torch.Tensor, bb: List, target_w=192, target_h=192):
    _, tensor_w, tensor_h = tensor.shape
    xmin = random.randint(0, tensor_w - 1 - target_w)
    ymin = random.randint(0, tensor_h - 1 - target_h)

    tensor = tensor[xmin:xmin+target_w, ymin:ymin+target_h]

    bb = bb[xmin:xmin+target_w, ymin:ymin+target_h]

    return tensor, bb

def crop_flip_with_bb(tensor: torch.Tensor, bb: List):
    Y = create_mask(tensor, bb)
    tensor, Y = random_crop_with_bb(tensor, Y)

    tensor, bb = random_crop_with_bb(tensor, bb)

    if np.random.random() > 0.5: 
            tensor = np.fliplr(tensor).copy()
            Y = np.fliplr(Y).copy()
    
    bb = mask_bb(Y)
    
    return tensor, bb

def show_bb(img, bb, color='yellow'):
    highlight_rect = plt.Rectangle((bb[2], bb[0]), bb[3]-bb[2], bb[1]-bb[0], color=color, fill=False, lw=2)

    plt.imshow(img)
    plt.gca().add_patch(highlight_rect)
