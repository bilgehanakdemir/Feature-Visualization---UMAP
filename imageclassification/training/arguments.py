import argparse

import ast

from pathlib import Path


def aslist(lst):
    return ast.literal_eval(lst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='data/')
    parser.add_argument('--metadata_root', default='metadata/')
    parser.add_argument('--dataset_name', choices=['CIFAR10',
                                                   'CIFAR100',
                                                   'MedMNIST',
                                                   ], default='MedMNIST')
    parser.add_argument('--snapshots', default='snapshots/')
    parser.add_argument('--logs', default='logs/')
    parser.add_argument('--train_meta', default='train_meta.csv')
    
    parser.add_argument('--experiment', type=str, choices=['vggbndrop',
                                                           'vgg'
                                                           ], default='vggbndrop')
    
    parser.add_argument('--num_classes', type=int, choices=[10, # CIFAR10
                                                            100, # CIFAR100
                                                            6    #MedNIST
                                                            ], default=6)
                                                            
    parser.add_argument('--training_data_loc', type=Path, default='/scratch/project_2006161/cifar-10/data/MednistRGB_split2/train/')   ###add argument defaults value 
    
    parser.add_argument('--testing_data_loc', type=Path, default='/scratch/project_2006161/cifar-10/data/MednistRGB_split2/val/')
    
    parser.add_argument('--color_space', type=str, choices=['yuv',
                                                            'rgb'
                                                            ], default='rgb')
    
    parser.add_argument('--optimizer', type=str, choices=['adam',
                                                          'QHM',
                                                          'QHAdam'
                                                          ], default='QHAdam')   # added by. (removed:'sgd')
    parser.add_argument('--set_nesterov', default=True) # used with SGD
    parser.add_argument('--learning_rate_decay', type=float, choices=[0.1,
                                                         0.2,  # https://github.com/szagoruyko/wide-residual-networks/blob/master/logs/vgg_24208029/
                                                         ], default=0.2)  #0.1 for MedNIST
    parser.add_argument('--lr', type=float, choices=[5e-4,
                                                     1e-3,
                                                     1e-2,
                                                     1e-1  # https://github.com/szagoruyko/wide-residual-networks/blob/master/logs/vgg_24208029/
                                                     ], default=5e-4) #1e-2
    parser.add_argument('--lr_drop', type=aslist, choices=[[160, 260], [100, 120]
                                              ], default=[40, 50]) #QHM
    parser.add_argument('--wd', type=float, choices=[5e-4,  # https://github.com/szagoruyko/wide-residual-networks/blob/master/logs/vgg_24208029/
                                                     1e-4,
                                                     1e-3,
                                                     1e-2
                                                     ], default=5e-4)

    parser.add_argument('--bs', type=int, default=128) #training' change to [4,32,256]  #warm up (Pytorch) method if huge more than 3000 samples >changed bs:128 to bs:256 to 32
    parser.add_argument('--val_bs', type=int, default=128) 

    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=-1)

    parser.add_argument('--n_epochs', type=int, choices=[50,300], default=50)
    parser.add_argument('--n_threads', type=int, choices=[24,
                                                     12,
                                                     6
                                                     ], default=24)

    parser.add_argument('--seed', type=int, default=444)

    args = parser.parse_args()

    return args
