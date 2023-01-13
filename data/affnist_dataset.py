import json
import os

import albumentations
import numpy as np
import torch

from collections import defaultdict
from pathlib import Path

from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset

import scipy.io as spio

train_domains = list(range(20))

def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def process_group(input_list):
    temp_group = []
    for item in input_list.transpose():
        rot_id = (item[0]+20) // 5 # rotation id
        if item[4]>0 and item[-1]<0:
            shift_id = 0
        elif item[4]>0:
            shift_id = 1
        elif item[-1]>0:
            shift_id = 2
        else:
            shift_id = 3
        gp_id = rot_id*4 + shift_id
    #     print('ground truth: ', rot_id, shift_id)
    #     print('try output y: ', gp_id//4, gp_id%4)
        temp_group.append(gp_id)
    return np.asarray(temp_group)

def read_dir(data_dir):
   
    dataset = loadmat(data_dir)

    ans_set = dataset['affNISTdata']['label_int']
    train_set = dataset['affNISTdata']['image'].transpose()/255.0
    grp_set = process_group(dataset['affNISTdata']['human_readable_transform'])

    train_set = train_set.reshape((len(ans_set), 40, 40, 1))
    train_set = train_set.astype(np.float32)

    ans_set = ans_set.astype(np.int32)
    grp_set = grp_set.astype(np.int32)
    return list(set(grp_set)), (grp_set, train_set, ans_set)

class affNISTDataset(Dataset):

    def __init__(self, split, root_dir):
        self.root_dir = Path(root_dir) / 'affnist' / split / '1.mat'
        clients, data = read_dir(self.root_dir)
        if split == 'train':
            self.n_groups = len(train_domains)
        else:
            self.n_groups = len(clients)
        self.groups = list(range(self.n_groups))

        self.image_shape = (1, 40, 40)

        agg_X, agg_y, agg_groups = [], [], []
        print("loading affNIST")
            
        for client in clients:
            if split == 'train' and client not in train_domains:
                continue
            ids = np.nonzero(data[0] == client)[0]
            if split == 'train':
                client_X = data[1][ids]
                client_y = data[2][ids]
                client_X = client_X[:int(len(client_X)*0.2)]
                client_y = client_y[:int(len(client_y)*0.2)]
                client_N = len(client_X)
            else:
                client_X = data[1][ids]
                client_y = data[2][ids]
                client_N = len(client_X)
            
            X_processed = np.array(client_X).reshape((client_N, 40, 40, 1))

            agg_X.append(X_processed)
            agg_y.extend(client_y)
            agg_groups.extend([client] * client_N)

        print("loaded")
        self._len = len(agg_groups)
        self._X, self._y = np.concatenate(agg_X), np.array(agg_y)
        self.group_ids = np.array(agg_groups)
        self.group_counts, _ = np.histogram(self.group_ids,
                                            bins=range(self.n_groups + 1),
                                            density=False)
        self.group_dist, _ = np.histogram(self.group_ids, 
                                          bins=range(self.n_groups+1), 
                                          density=True)
        
        # Classes
        self.n_classes = 10
        self.classes = np.array(range(self.n_classes))
        self.class_with_ids = self._get_class_ids(self._y)
        
        self.transform = get_transform()

        print("split: ", split)
        print("n groups: ", len(clients))
        print("Dataset size: ", len(self._y))

        print("Smallest group: ", np.min(self.group_counts))
        print("Largest group: ", np.max(self.group_counts))

    def __len__(self):
        return self._len
    
    def _get_class_ids(self, labels):
        """Returns the class ids for each example

            TODO: Clean up this function"""

        class_with_ids = {}
        for cls in range(self.n_classes):
            ids = np.nonzero(np.asarray(labels == cls))[0]
            class_with_ids[cls] = ids

        return class_with_ids
        

    def __getitem__(self, index):
        x = self.transform(**{'image': self._X[index]})['image']
        y = torch.tensor(self._y[index], dtype=torch.long)
        g = torch.tensor(self.group_ids[index], dtype=torch.long)
        return x, y, g

def get_transform():
    transform = albumentations.Compose([
        ToTensor(),
    ])
    return transform
