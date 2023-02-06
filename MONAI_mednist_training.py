import os
import pickle
import copy
from termcolor import colored
import torch
print("cuda available: ", torch.cuda.is_available())
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
#from imageclassification.training import qhm
import numpy as np
import pandas as pd
import solt.data as sld
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from imageclassification.kvs import GlobalKVS
from PIL import Image
import matplotlib.pyplot as plt
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
#from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from monai.data import Dataset, DataLoader    #any differences here and below
from monai.data import decollate_batch, DataLoader
#from torch.utils.data import DataLoader
from monai.utils import set_determinism
#print_config()

import imageclassification.training.model as mdl

data_dir = ''
path1 = ''

class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
print("class_names: ",class_names)
num_class = len(class_names)

image_files = [[os.path.join(data_dir, class_name, x) 
                for x in os.listdir(os.path.join(data_dir, class_name))] 
               for class_name in class_names]

classes = {
    'AbdomenCT' : [100, 100, 255],
    'BreastMRI' : [0, 0, 0],
    'CXR' : [10, 189, 227],
    'ChestCT' : [128, 80, 128],
    'Hand' : [255, 159, 243],
    'HeadCT' : [254, 202, 87],
} 

colors_per_class = {
    0 : [100, 100, 255],
    1 : [0, 0, 0],
    2 : [10, 189, 227],
    3 : [128, 80, 128],
    4 : [255, 159, 243],
    5 : [254, 202, 87],
}
image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
num_total = len(image_label_list)
image_width, image_height = Image.open(image_file_list[0]).size

valid_frac, test_frac = 0.1, 0.1
trainX, trainY = [], []
valX, valY = [], []
testX, testY = [], []

for i in range(num_total):
    rann = np.random.random()
    if rann < valid_frac:
        valX.append(image_file_list[i])
        valY.append(image_label_list[i])
    elif rann < test_frac + valid_frac:
        testX.append(image_file_list[i])
        testY.append(image_label_list[i])
    else:
        trainX.append(image_file_list[i])
        trainY.append(image_label_list[i])
    
print("Training sample =",len(trainX),"Validation sample =", len(valX), "Test sample =",len(testX))

train_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    RandRotate(range_x=15, prob=0.5, keep_size=True),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
    ToTensor()
])


val_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    ToTensor()
])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])


class MedNISTDataset(data.Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

train_ds = MedNISTDataset(trainX, trainY, train_transforms)
train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=2)

val_ds = MedNISTDataset(valX, valY, val_transforms)  #val_transforms
val_loader = DataLoader(val_ds, batch_size=300, num_workers=2)

test_ds = MedNISTDataset(testX, testY, val_transforms)
test_loader = DataLoader(test_ds, batch_size=300, num_workers=2)

import torch_optimizer as optim2
import imageclassification.training.models as mdls
from torch.optim.lr_scheduler import MultiStepLR  
import torch.nn.functional as F
from tqdm import tqdm
import gc
import umap
import torchvision.datasets as dset
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)  
model = mdls.VGGBNDrop(num_classes=6, init_weights=True).to(device) # True to false

optimizer =  optim2.QHM(model.parameters(), weight_decay=1e-5, **optim2.QHM.from_synthesized_nesterov(alpha=0.1, beta1=0.9, beta2=0.6))     #**QHM.from_synthesized_nesterov(alpha=0.1, beta1=0.9, beta2=0.6))
#optimizer =  optim2.QHM(model.parameters(), lr=1e-5, weight_decay=5e-4, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), 1e-5)
print("optimizer:",optimizer)

scheduler = MultiStepLR(optimizer, milestones=[40, 50], gamma=0.2)
num_epoch = 300
max_ep = num_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_interval = 1
auc_metric = ROCAUCMetric()

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(num_epoch):
    print("-" * 10)
    print(f"epoch: {epoch + 1}/{num_epoch}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        print("batch_data: ",batch_data)
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), './snapshots/MedNIST/%d.pth' % epoch)
                print("New best metric model is saved.")
            print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.1f}"
                    f" at epoch: {best_metric_epoch}"
                )

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")
             
fig = plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val AUC")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
#plt.show()
plt.savefig('MONAI_mednist_loss.png', dpi=fig.dpi)




