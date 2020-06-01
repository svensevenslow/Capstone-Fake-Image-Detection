from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
from flaskr import cyclegan_data 
import torch.nn as nn

from torchvision import transforms, models

BATCH_SIZE=16

class gan_data(cyclegan_data.cyclegan_data):
    
    def __init__(self,train = True):
        super(gan_data,self).__init__()
        self.batch_size = BATCH_SIZE
        self.train = train
    
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        if self.train:
            im = deepcopy(img.numpy()[16:240,16:240,:])
        else:
            im = deepcopy(img.numpy()[16:240,16:240,:])
        
        #fft

        im = im.astype(np.float32)
        im = im/255.0

        for i in range(3):
            img = im[:,:,i]
            fft_img = np.fft.fft2(img)
            fft_img = np.log(np.abs(fft_img)+1e-3)
            fft_min = np.percentile(fft_img,5)
            fft_max = np.percentile(fft_img,95)
            fft_img = (fft_img - fft_min)/(fft_max - fft_min)
            fft_img = (fft_img-0.5)*2
            fft_img[fft_img<-1] = -1
            fft_img[fft_img>1] = 1

            #take the whole band
            im[:,:,i] = fft_img

        im = np.transpose(im, (2,0,1))
        return (im, label)
    
    def __len__(self):
        return self.labels.size(0)
