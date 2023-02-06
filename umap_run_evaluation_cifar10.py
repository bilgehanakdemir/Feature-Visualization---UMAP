import sys
print(sys.version, sys.platform, sys.executable)

import argparse
import random
import os
import pickle
import gc

#from cifar10_dataset import colors_per_class

from functools import partial

import numpy as np
import cv2
import pandas as pd

import torch
print('Torch version:', torch.__version__)

from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms as tv_transforms

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

from tqdm import tqdm

import solt.transforms as slt
import solt.core as slc

from imageclassification.training.dataset import apply_by_index, img_labels2solt, unpack_solt_data
import imageclassification.training.model as mdl
from imageclassification.training.dataset import ImageClassificationDataset, init_dataset

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import matplotlib.pyplot as plt

import glob

from os.path import join

#from sklearn.manifold import TSNE

import umap

from time import perf_counter, sleep

from imageclassification import kvs

from imageclassification.training.dataset import init_dataset
#from imageclassification.training.arguments import parse_args
start = perf_counter()


PAD_TO = 34
CROP_SIZE = 32

DEBUG = sys.gettrace() is not None
print('Debug: ', DEBUG)

classes = {
    'airplane' : [100, 100, 255],
    'automobile' : [0, 0, 0],
    'bird' : [10, 189, 227],
    'cat' : [128, 80, 128],
    'deer' : [255, 159, 243],
    'dog' : [254, 202, 87],
    'frog' : [16, 172, 132],
    'horse' : [255, 107, 107],
    'ship' : [87, 101, 116],
    'truck' : [52, 31, 151],
}   

colors_per_class = {
    0 : [100, 100, 255],
    1 : [0, 0, 0],
    2 : [10, 189, 227],
    3 : [128, 80, 128],
    4 : [255, 159, 243],
    5 : [254, 202, 87],
    6 : [16, 172, 132],
    7 : [255, 107, 107],
    8 : [87, 101, 116],
    9 : [52, 31, 151],
}

def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_features(model, features, loader):

     # move the input and model to GPU for speed if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    model.eval()
    model.to(device)
    
    # read the dataset and initialize the data loader
    #dataset = AnimalsDataset(dataset, num_images)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)
    
    features = None
    
    #store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    
    running_loss = 0.0
    n_batches = len(loader)
    print("loader length:",n_batches)

    probs_lst = []
    gt_lst = []

    pbar = tqdm(total=n_batches)
    
    labels = []
    image_paths = []
    
    correct = 0
    all_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            
            labels += batch['label'].long().to(device)
            #print("labels",labels)
            inputs = batch['img'].to(device)
            #image_paths += batch['image_path']
            #print(i)
            #print(batch)
            #exit()

            #outputs, features = model(inputs)
            outputs = model(inputs)           # return x for visualization, output shape: (10000,10)
            
            labels = torch.Tensor(labels).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            loss = F.cross_entropy(outputs,labels)  
            probs_batch = F.softmax(outputs, 1).data.to('cpu').numpy()                ### what is probs_batch? why softmax?  probs_batch shape: (10000,10)
            
            gt_batch = batch['label'].numpy()     #gt_batch: [3 8 8 ... 5 1 7]

            probs_lst.extend(probs_batch.tolist())
            gt_lst.extend(gt_batch.tolist())

            running_loss += loss.item()

            pred = np.array(probs_lst).argmax(1)
            
            correct += np.equal(pred, np.array(gt_lst)).sum()
            
            all_samples += len(np.array(gt_lst))

            gc.collect()
            pbar.set_description(
                f"Evaluation accuracy: {100. * correct / all_samples:.0f}%")
            pbar.update()          
            ############### added by B.
            current_features = outputs.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features
                     
            #tsne = TSNE(n_components=2).fit_transform(features)   #features shape: (10000,10)
            
            print("features:", features.shape)
            embedding  = umap.UMAP(n_neighbors=50,min_dist=0.3,metric='correlation').fit_transform(features)   #features shape: (10000,10)
            print("UMAP Embedding:", embedding.shape)
            #reducer2 = UMAP(n_neighbors=100, n_components=3, n_epochs=1000, min_dist=0.5, local_connectivity=2, random_state=42)
            
            
            tx = embedding[:, 0]
            ty = embedding[:, 1]
            
            #print("labels type:",type(labels))
            
            
            #print("labels type:",type(labels))
            
            def visualize_umap_points(tx, ty, labels):
                labels = labels.int().tolist()
                print("labels type:",type(labels[0]))
                #print(labels)
                # initialize matplotlib plot
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
  
            # for every class, we'll add a scatter plot separately
                #for i, category in colors_per_class:
                list_cl = []
                for i in classes:
                  list_cl.append(i)
                #print(list_cl)
                #assert 1 == 2
                for category in colors_per_class:
                    #sys.exit()
                    # find the samples of the current class in the data
                    #indices = []
                    
                    #for i, l in enumerate(labels):
                    #  print("category: ", category)
                    #  print("l: ", l)
                    #  if l == category:
                    #     indices.append(i)
                    #indices = [i for i, l in enumerate(labels) if l == i]
                    indices = [i for i, l in enumerate(labels) if l == category]
                    
                    #print("indices:",indices)

                    # extract the coordinates of the points of this class only
                    current_tx = np.take(tx, indices)
                    current_ty = np.take(ty, indices)
                    #print("current_ty:",current_ty)

                    # convert the class color to matplotlib format:
                    # BGR -> RGB, divide by 255, convert to np.array
                    color = np.array([colors_per_class[category][::-1]], dtype=np.float64) / 255
                  

                    # add a scatter plot with the correponding color and label
                    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    ax.scatter(current_tx, current_ty, c=color, label=list_cl[category])

                  # build a legend using the labels we set previously
                ax.legend(loc='best')

                # finally, show the plot
                #plt.show()
                fig.savefig('UMAP600_bs_128.png', dpi=fig.dpi)   
            
            #print("labels:",labels)
            #print("tx:",tx)
            #print("ty:",ty)
            visualize_umap_points(tx, ty, labels)        
            
            
           
        gc.collect()
        pbar.close()
        
    # val_loss, preds, gt, val_acc
    return running_loss / n_batches, np.array(probs_lst), np.array(gt_lst), correct / all_samples



def eval_model(model, device, loader):
    model.eval()

    running_loss = 0.0
    n_batches = len(loader)

    probs_lst = []
    gt_lst = []

    #pbar = tqdm(total=n_batches)

    correct = 0
    all_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            labels = batch['label'].long().to(device)
            inputs = batch['img'].to(device)

            outputs = model(inputs)

            loss = F.cross_entropy(outputs, labels)

            probs_batch = F.softmax(outputs, 1).data.to('cpu').numpy()
            gt_batch = batch['label'].numpy()

            probs_lst.extend(probs_batch.tolist())
            gt_lst.extend(gt_batch.tolist())


            pred = np.array(probs_lst).argmax(1)

    # val_loss, preds, gt, val_acc
    return pred, gt_lst
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='data/')
    parser.add_argument('--metadata_root', default='metadata/')
    parser.add_argument('--dataset_name', choices=['CIFAR10',
                                                   'CIFAR100',
                                                   ], default='CIFAR10')
    parser.add_argument('--snapshots', default='snapshots/CIFAR10/2022_09_01_17_37')
    parser.add_argument('--experiment', type=str, choices=['vggbndrop',
                                                           'vgg'
                                                           ], default='vggbndrop')
    
    parser.add_argument('--num_classes', type=int, choices=[10, # CIFAR10
                                                            100 # CIFAR100
                                                            ], default=10)
    
    args = parser.parse_args()
    


    mean_vector, std_vector = np.load(os.path.join('snapshots/CIFAR10', 'mean_std.npy'))
    norm_trf = tv_transforms.Normalize(torch.from_numpy(mean_vector).float(), torch.from_numpy(std_vector).float())

    eval_trf = tv_transforms.Compose([
        img_labels2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(PAD_TO, PAD_TO)),
            slt.CropTransform(crop_size=(CROP_SIZE, CROP_SIZE), crop_mode='c'),  # center crop
        ]),
        unpack_solt_data,
        partial(apply_by_index, transform=tv_transforms.ToTensor(), idx=0),
        partial(apply_by_index, transform=norm_trf, idx=0)
        ])
    
    path = 'data/'
    dataset = 'CIFAR10'



    model_files = glob.glob(join(args.snapshots,'*.pth'))

    dataset, dataset_length = init_dataset(path, dataset, batch='test')

    metadata = pd.read_csv(os.path.join(args.metadata_root, 'CIFAR10', 'test_meta.csv'))

    eval_dataset = ImageClassificationDataset(dataset, metadata, 'rgb', eval_trf)

    eval_loader = DataLoader(eval_dataset, batch_size=10000, num_workers=1)


���
