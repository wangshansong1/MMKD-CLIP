# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image, ImageFile
import torch
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_downstream_dataset(args, catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']

    if name == 'LC25000_lung':
        if is_train:
            dataset_train = LC25000(args, root, transform=transform, split="train")
            dataset_val = LC25000(args, root, transform=transform, split="val")
            dataset_test = LC25000(args, root, transform=transform, split="test")
        else:
            dataset_test = LC25000(args, root, transform=transform, split="test")

    elif name == 'LC25000_colon':
        if is_train:
            dataset_train = LC25000(args, root, transform=transform, split="train")
            dataset_val = LC25000(args, root, transform=transform, split="val")
            dataset_test = LC25000(args, root, transform=transform, split="test")
        else:
            dataset_test = LC25000(args, root, transform=transform, split="test")

    if is_train:
        return dataset_train, dataset_val, dataset_test
    else:
        return dataset_test


class LC25000(torch.utils.data.Dataset):
    def __init__(self, args, parent_path, transform=None, split='test'):
        self.transform = transform
        self.split = split
        self.args = args
        self.data = []
        self.datasetname = parent_path.split('zeroshotclassification/')[-1].replace('/','')

        txt_map = {
            'train': os.path.join(parent_path,'train.txt'),
            'val': os.path.join(parent_path,'val.txt'),
            'test': os.path.join(parent_path,'test.txt')
        }

        txt_file = txt_map[self.split]
        split_dir = os.path.join(parent_path, self.split)
        
        with open(txt_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split()
                image_path = os.path.join(split_dir, image_name)
                self.data.append((image_path, label))


        if self.split == 'train' and hasattr(self.args, 'TS_ratio') and 0 < self.args.TS_ratio < 1.0:
            np.random.seed(42)

            label_to_samples = defaultdict(list)
            for image_path, label in self.data:
                label_to_samples[label].append((image_path, label))

            balanced_subset = []
            for label, samples in label_to_samples.items():
                sample_size = int(len(samples) * self.args.TS_ratio)
                sampled = random.sample(samples, sample_size)
                balanced_subset.extend(sampled)

            self.data = balanced_subset
        
        if self.datasetname == 'LC25000_lung':
            self.label2idx = {'lung_aca': 0, 'lung_n': 1, 'lung_scc': 2}
        else:
            self.label2idx = {'colon_aca': 0, 'colon_n': 1}
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        img = Image.open(image_path).convert('RGB')
        label = self.label2idx[label]

        if self.transform:
            image = self.transform(img)
        else:
            image = img

        return image, label
