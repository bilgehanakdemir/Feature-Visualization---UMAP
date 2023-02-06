import os
import cv2
import glob
import shutil
import matplotlib.pyplot as plt
import splitfolders # or import splitfolders


#path = r'/scratch/project_2006161/mednist-copy/MedNIST2/'



input_folder = r"/scratch/project_2006161/mednist-copy/MedNIST2/"
output = "/scratch/project_2006161/cifar-10/data/MedNIST/" #where you want the split datasets saved. one will be created if it does not exist or none is set
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .1, .1)) # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.
























      
      