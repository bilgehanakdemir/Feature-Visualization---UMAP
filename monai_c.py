import sys
import gc

from termcolor import colored

import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import MultiStepLR

from imageclassification.kvs import GlobalKVS
from imageclassification.training import utilities
from imageclassification.training import session
from imageclassification.training import dataset as ds
import imageclassification.training.metrics as metrics

from monai.config import print_config
from monai.transforms import \
    Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, Resize
from monai.networks.nets import densenet201
from monai.metrics import compute_roc_auc

np.random.seed(0)
print_config()

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

print(sys.version, sys.platform, sys.executable)

DEBUG = sys.gettrace() is not None
print('Debug: ', DEBUG)


if __name__ == "__main__":
    kvs = GlobalKVS()

    session.init_session()
    ds.init_metadata()
    dataset, dataset_length = ds.init_dataset(kvs['args'].dataset_root, kvs['args'].dataset_name)

    session.init_data_processing(dataset)
    writers = session.init_folds()

    for fold_id in kvs['cv_split_train']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)
        
        print(colored('====> ', 'blue') + f'Training fold {fold_id}...')

        train_index, val_index = kvs['cv_split_train'][fold_id]
        train_loader, val_loader = session.init_loaders(dataset, kvs['metadata'].iloc[train_index],
                                                        kvs['metadata'].iloc[val_index])

        net = utilities.init_model()
        print(net)
        optimizer = utilities.init_optimizer(net.parameters())
        print("optimizer:",optimizer)

        print('Learning rate drop schedule: ', kvs['args'].lr_drop)

        scheduler = MultiStepLR(optimizer, milestones=kvs['args'].lr_drop, gamma=kvs['args'].learning_rate_decay)

        train_losses = []
        val_losses = []

        for epoch in range(kvs['args'].n_epochs):
            kvs.update('cur_epoch', epoch)

            print(colored('====> ', 'red') + 'Learning rate: ', str(scheduler.get_lr())[1:-1])
            train_loss = utilities.train_epoch(net, optimizer, train_loader)
            val_out = utilities.validate_epoch(net, val_loader)
            val_loss, preds, gt, val_acc = val_out

            if (epoch+1)%25==0:
                print('Epoch: ', epoch)
                print('Acc: ', val_acc)

            metrics.log_metrics(writers[fold_id], train_loss, val_loss, gt, preds)

            session.save_checkpoint(net, 'val_loss', 'lt')  # lt, less than
            
            scheduler.step()

            gc.collect()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
        fig = plt.figure(figsize=(15, 15))

        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        fig.savefig('QHM_loss16_300.png', dpi=fig.dpi)
        #fig.close()
        #plt.show()

        del net

        torch.cuda.empty_cache()
        gc.collect()
