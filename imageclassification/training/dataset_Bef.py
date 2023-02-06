import os
import pickle
import copy

from termcolor import colored

import torch
import torch.utils.data as data

from sklearn.model_selection import StratifiedKFold

import numpy as np

import pandas as pd

import solt.data as sld

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from imageclassification.kvs import GlobalKVS
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from abc import abstractmethod

train_dir = '/scratch/project_2006161/cifar-10/data/MednistRGB_split2/train/'
val_dir = '/scratch/project_2006161/cifar-10/data/MednistRGB_split2/val/'

class ImageClassificationDataset(data.Dataset):
    def __init__(self, dataset, split, color_space='rgb', transformations=None):
        self.dataset = dataset
        self.split = split
        self.color_space = color_space
        self.transformations = transformations

    def __getitem__(self, ind):
        if isinstance(ind, torch.Tensor):
            ind = ind.item()

        entry = self.split.iloc[ind]
        indx = entry.ID-1 # ID is a row number starting from 1
        
        
        
        dimg = self.dataset[indx, :, :, :]   #### read images one by one, read from current filename
        

        if 'yuv' in self.color_space:
            dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2YUV)
            dimg[:, :, 0] = cv2.equalizeHist(dimg[:, :, 0])  #####

        img, label = self.transformations((entry.Label, dimg))

        res = {'img': img,'label': label}

        return res

    def __len__(self):
        return self.split.shape[0]


def unpickle(file):
    """
    Source: https://www.cs.toronto.edu/~kriz/cifar.html

    :param file: Python object produced with cPickle
    :return: dictionary
    """
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')

    return cifar_dict
    




from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

#from imageclassification.training.transformations import init_train_augs
#train_augs = init_train_augs(crop_mode='r', pad_mode='r')  # random crop, reflective padding
PAD_TO = 66    #### from 34 to 66
CROP_SIZE = 64   #convert grayscale image to rgb////////////// from 32 to 64
                
def init_dataset(path, dataset, batch='train'):  # path = kvs['args'].dataset_root
    
    if 'CIFAR10' == dataset:
        if 'train' == batch:
            batch_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
            print("batch_list: ",batch_list)
        elif 'test' == batch:
            batch_list = [
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]
        else:
            raise NotImplementedError
    elif 'CIFAR100' == dataset:
        if 'train' == batch:
            batch_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]
            
        elif 'test' == batch:
            batch_list = [
                ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
            ]
        else:
            raise NotImplementedError
    
    elif 'MedMNIST' == dataset:
        if 'train' == batch:
#            batch_list = []
            train_ds = datasets.ImageFolder(train_dir, transform=transform)
            #train_ds = ImageClassificationDataset(train_dir, transform=transform)
            train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)
            print("length train_loader: ", train_loader)
            for i, (img, label) in enumerate(train_loader):
                print("img:",img.shape,"label:", label.shape)
                #batch_list = [img, label]
                
        elif 'test' == batch:
#            batch_list = []
            test_ds = datasets.ImageFolder(val_dir, transform=transform)
            test_loader = DataLoader(test_ds, batch_size=300, shuffle=True, num_workers=10)
            print("length train_loader: ", test_loader)
            for i, (img, label) in enumerate(test_loader):
                print("img:",img.shape,"label:", label.shape)
                #batch_list = [img, label]
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError
        
    path = os.path.join(path, dataset)
    print("path: ",path)

    ds = []
    for entry in batch_list:
        print(colored('====> ', 'blue') + 'Processing file: ', os.path.join(path, entry[0]))
        batch = unpickle(os.path.join(path, entry[0]))
        tmp = batch[b'data']
        ds.append(tmp)

    ds = np.concatenate(ds)
    l_dataset = len(ds)
    #ds = ds.reshape(l_dataset, 3, 32, 32).swapaxes(1, 3).swapaxes(1, 2) #CIFAR
    ds = ds.reshape(l_dataset, 3, 64, 64).swapaxes(1, 3).swapaxes(1, 2) #MedNIST

    return ds, l_dataset

##################### init training metadata here; implement from scratch for MedNIST

#   training_img_lst = list(args.training_data_loc.glob('*.png'))
#   class_names = list(map(lambda x: x.stem.split('_')[0], training_img_lst))
#   df = pd.DataFrame(
#         {'Class': class_names, 'FilePath': training_img_lst})

def make_csv(save_path, file_name, path_dir):
    kvs = GlobalKVS()
    class_names = sorted([x for x in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, x))])
    num_class = len(class_names)
    X = [] 
    Y = []
    img_lst = [[os.path.join(path_dir, class_name, x) 
                    for x in os.listdir(os.path.join(path_dir, class_name))] 
                   for class_name in class_names]
    for i, class_name in enumerate(class_names):
        X.extend(img_lst[i])
        Y.extend([i] * len(img_lst[i]))
    
    meta = pd.DataFrame({'Label': Y, 'Filename': X})  # TODO: Add column ID
    idx = 0
    meta.insert(idx, 'ID', value=range(len(meta)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, file_name)):
        meta.to_csv(os.path.join(save_path, file_name), index=False)
    meta_read = pd.read_csv(os.path.join(kvs['args'].metadata_root, kvs['args'].dataset_name, kvs['args'].train_meta))
    return meta_read

def init_metadata():

       
    kvs = GlobalKVS()
    
    if 'CIFAR' in kvs['args'].dataset_name:
        meta = pd.read_csv(os.path.join(kvs['args'].metadata_root, kvs['args'].dataset_name, kvs['args'].train_meta))
    
    elif 'MedMNIST' in kvs['args'].dataset_name:
        save_path = "/scratch/project_2006161/cifar-10/metadata/MedMNIST"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        train_file_name = "train_meta.csv"
        test_file_name = "test_meta.csv"
        meta = make_csv(save_path, train_file_name, kvs['args'].training_data_loc)
        test_meta = make_csv(save_path, test_file_name, kvs['args'].testing_data_loc)    
        
##        train_ds = init_dataset(trainX, trainY, transforms= None)
##        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
##        for batch_file in train_loader:
     
    else:
        raise NotImplementedError


    print(f'Dataset (form CSV file) has {meta.shape[0]} entries')

    kvs.update('metadata', meta)
    

    skf = StratifiedKFold(n_splits=kvs['args'].n_folds)  # this cross-validation object is a variation of KFold that returns stratified folds, preserving the percentage of samples for each class
    
    

    if 'CIFAR10' in kvs['args'].dataset_name:
        cv_split = [x for x in skf.split(kvs['metadata']['Filename'].astype(str),
                                         kvs['metadata']['Label'],
                                         kvs['metadata']['ID'])]
    
    elif 'MedMNIST' in kvs['args'].dataset_name:
        cv_split = [x for x in skf.split(kvs['metadata']['Filename'].astype(str),
                                         kvs['metadata']['Label'],
                                         kvs['metadata']['ID'])]                                   
                                         
    elif 'CIFAR100' in kvs['args'].dataset_name:
        cv_split = [x for x in skf.split(kvs['metadata']['Filename'].astype(str),
                                         kvs['metadata']['Label'],
                                         kvs['metadata']['Group'],
                                         kvs['metadata']['ID'])]
                                         
   
    else:
        raise NotImplementedError

    kvs.update('cv_split_all_folds', cv_split)
    kvs.save_pkl(os.path.join(kvs['args'].snapshots, kvs['args'].dataset_name, kvs['snapshot_name'], 'session.pkl'))


def img_labels2solt(inp):
    label, img = inp
    return sld.DataContainer((img, label), fmt='IL')


def unpack_solt_data(dc: sld.DataContainer):
    return dc.data


def apply_by_index(items, transform, idx=0):
    """
    Applies callable to certain objects in iterable using given indices.
    Parameters

    :param items: tuple or list
    :param transform: callable
    :param idx: int or tuple or or list None
    :return: tuple
    """
    if idx is None:
        return items
    if not isinstance(items, (tuple, list)):
        raise TypeError
    if not isinstance(idx, (int, tuple, list)):
        raise TypeError

    if isinstance(idx, int):
        idx = [idx, ]

    idx = set(idx)
    res = []
    for i, item in enumerate(items):
        if i in idx:
            res.append(transform(item))
        else:
            res.append(copy.deepcopy(item))

    return res
