import csv
import glob
import numpy as np
import os
import torch
import uuid

import xml.etree.ElementTree as ET
import torchvision.transforms as tf
from .boxes import resize_with_bb
from torch.utils.data import Dataset
from skimage import io

# TODO: VALIDATE
preprocess = tf.Compose([
    # tf.ToPILImage(),
    # tf.Resize(size=192),
    # tf.RandomCrop(size=128),
    tf.ToTensor()
])

def to_three_channel_torch(tensor: torch.Tensor):
    if tensor.ndimension() == 2:
        w, h = tensor.shape
        return torch.zeros((3, w, h)) + tensor.unsqueeze(0)
    elif tensor.ndimension() == 3:
        w, h = tensor.shape[1:]
        # CASE: B/W
        if tensor.shape[0] == 1:
            return torch.zeros(3, w, h) + tensor
        # CASE: RGBA
        elif tensor.shape[0] == 4:
            return tensor[:3, :, :]
        elif tensor.shape[0] == 3:
            return tensor
        else:
            raise Exception(f'Tensor with 3 dimensions has shape {tensor.shape}')
    else:
        raise Exception(f'Tensor has {tensor.ndimension()} dimensions')

def tensor_imread(fname: str):
    img = io.imread(fname)
    tensor = preprocess(img)
    tensor = to_three_channel_torch(tensor)
    
    return tensor

# ids: List[int] (index to image_id)
# labels: Map[int, int] (image_id to label)
#   (technically, this could be a parallel list associated with ids, too)
# dataset_dir: str (directory to stored tensors)
class StandardDataset(Dataset):
    def __init__(self, ids, labels, dataset_dir: str, transform=None):
        self.ids = ids
        self.labels = labels
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, dex):
        if torch.is_tensor(dex):
            dex = dex.tolist()
        image_id = self.ids[dex]

        filename = os.path.join(self.dataset_dir, image_id + ".pt")
        X = torch.load(filename)

        if self.transform:
            X = self.transform(X)

        y = self.labels[image_id]

        return X, y

clpd_xml_dir = "/Users/ericcarlson/Desktop/Datasets/CLPD/annotations"
clpd_images_dir = "/Users/ericcarlson/Desktop/Datasets/CLPD/images"
clpd_count = 433 # COUNT FROM ZERO
# clpd_csv_dir = "/Users/ericcarlson/Desktop/Datasets/CLPD/labels.csv"
# clpd_dataset_dir = "/Users/ericcarlson/Desktop/Datasets/CLPD/dataset.csv"
clpd_tensor_dir = "/Users/ericcarlson/Desktop/Datasets/CLPD/tensors"

# dataset_dir: str (directory to produce 'ids', 'labels', and image tensors with filenames '<id>.pt')
# Here, the possible 'id's lie between 0 and 'clpd_count'.
class CarLicensePlateDataset(Dataset):
    def __init__(self, transform=None):
        # self.ids = None 
        # self.labels = None
        self.bb = {} # Map[int, Tuple[int]]

        for fname in glob.iglob(os.path.join(clpd_xml_dir, "**.xml")):
            print(fname)

            xml_tree = ET.parse(fname)
            root_xml = xml_tree.getroot()

            png_fname = root_xml.find('filename').text
            id = int(png_fname[4:-4]) # Cars<id>.xml

            # print('id', id)

            bb_xml = root_xml.find('object').find('bndbox')
            ymin = int(bb_xml.find('ymin').text)
            ymax = int(bb_xml.find('ymax').text)
            xmin = int(bb_xml.find('xmin').text)
            xmax = int(bb_xml.find('xmax').text)

            # print('bb', ymin, ymax, xmin, xmax)

            bb = (ymin, ymax, xmin, xmax)

            save_to = os.path.join(clpd_tensor_dir, str(id) + ".pt")
            tensor = tensor_imread(os.path.join(clpd_images_dir, png_fname))

            # Process 'tensor' and 'bb' in parallel.
            tensor, bb = resize_with_bb(tensor, bb)

            self.bb[id] = bb
            torch.save(tensor, save_to)
            
            print('tensor saved to', save_to)
    
    def __len__(self):
        return clpd_count

    def __getitem__(self, dex):
        if torch.is_tensor(dex):
            # dex = dex.tolist()
            raise Exception('multiple indexes not implemented')

        X = torch.load(os.path.join(clpd_tensor_dir, str(dex) + ".pt"))
        
        y = self.bb[dex]

        return X, y
