# coding: utf-8
#本方法修改Attention GCN网络，CNN提取特征
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import shutil
from utils9 import  accuracy
from models9 import GAT, SpGAT,CNN_fea,GCN
import os
import utils9
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import sklearn
from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage import data,color,morphology,measure
from skimage.feature import local_binary_pattern



def generate_gt_shanxipucheng():
    gt_original=image.imread('./GT.jpg')
     #gt_original=image.imread('../../datasets/fangchenggang/ground_truth2.jpg')
    out_GT=np.zeros((gt_original.shape[0],gt_original.shape[1]),dtype="uint8")+1
    out_GT[(gt_original[:,:,0]>=200) *( gt_original[:,:,1]>=200)*(gt_original[:,:,2]<=50)]=1
    out_GT[(gt_original[:,:,0]>=200) *( gt_original[:,:,1]<=50)*(gt_original[:,:,2]<=50)]=0

    np.save('gt_numerical_shanxipucheng.npy',out_GT)
    #np.save('gt_numerical_fangchenggang.npy',out_GT)
generate_gt_shanxipucheng()
