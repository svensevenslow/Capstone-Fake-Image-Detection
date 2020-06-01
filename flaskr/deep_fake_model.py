from flask import (Blueprint, render_template, make_response)
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
# from gan_data import gan_data
from flaskr import gan_data
from collections import OrderedDict
import csv
from torchvision import transforms, models

bp = Blueprint('deep_fake_model', __name__, url_prefix='/model')

MODEL_PATH = './model/deep_fake_model.pth'
IMAGE_PATH = './uploads/tmp.jpg'

dataset_names = ['tmp']
TEST_BATCH_SIZE = 16

@bp.route('/deep_fake_prediction', methods=['GET'])
def get_model_prediction():
    model = load_model()
    test_loaders = create_loaders()
    res = 0
    show = ""
    for test_loader in test_loaders:
        predicts = test(test_loader['dataloader'], model, 0)*100
        res = predicts[0]
        print(res)
    if res < 50:
        show = "This is a Deep Fake"
    else:
        show = "This is not a Deep Fake"
    return render_template('homepage.html',show = show)


def load_model():
    print("loading model")
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    load_model = torch.load(MODEL_PATH)
    model.load_state_dict(load_model['state_dict'])
    print('Model is loaded')
    return model
    
def load_image():
    img = cv2.imread(IMAGE_PATH)
    if img.shape[0]!=256:
        img = cv2.resize(img, (256,256))
    
    im = deepcopy(img[16:240,16:240,:])
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

    return im
    
def create_loaders():
    test_dataset_names = copy.copy(dataset_names)
    kwargs = {}
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             gan_data.gan_data(train=False),
                        batch_size=TEST_BATCH_SIZE,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders

def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, predicts = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (image_pair, label) in pbar:

        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)

        out = model(image_pair)
        _, pred = torch.max(out,1)
        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)
        predicts.append(pred)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    
    # print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    # acc = np.sum(labels == predicts)/float(num_tests)
    # print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))
    
    # pos_label = labels[labels==1]
    # pos_pred = predicts[labels==1]
    # TPR = np.sum(pos_label == pos_pred)/float(pos_label.shape[0])
    # print('\33[91mTest set: TPR: {:.8f}\n\33[0m'.format(TPR))

    # neg_label = labels[labels==0]
    # neg_pred = predicts[labels==0]
    # TNR = np.sum(neg_label == neg_pred)/float(neg_label.shape[0])
    # print('\33[91mTest set: TNR: {:.8f}\n\33[0m'.format(TNR))

    return predicts

