# coding: utf-8
#本程序将防城港大图分割成12*12个小图，使用斜对角的7张图作为目标识别,其余29张图像测试，使用CNN提取超像素特征，程序最后输出像素级的分类精度，保存分类结果图。
from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils9
import models9
from models9 import CNN
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
import torch.utils.data as data
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as image
image.MAX_IMAGE_PIXELS = None
import sklearn
from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage import data,color,morphology,measure
from skimage.feature import local_binary_pattern
import random
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=150,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
patch_size=32
num_label=2
n_segments=800
rownum=5
colnum=17
max_n_superpxiel=800
batchsize=max_n_superpxiel
train_patch_ind=range(35,51)

test_patch_ind=range(rownum*colnum)
for j_tr in train_patch_ind:
    test_patch_ind.remove(j_tr)

# Load data
iput_img_original=image.imread('../../datasets/陕西蒲城/陕西蒲城_裁剪.jpg')
gt_original=image.imread('../../datasets/陕西蒲城/陕西蒲城GT_裁剪.jpg')

print(gt_original.shape)
h_ipt=iput_img_original.shape[0]
w_ipt=iput_img_original.shape[1]   
rowheight = h_ipt // rownum
colwidth = w_ipt // colnum 
print(rowheight,colwidth)
model_cnn = models9.CNN( )
optimizer_cnn = optim.Adam(model_cnn.parameters(),lr=args.lr, weight_decay=args.weight_decay)
if args.cuda:
    model_cnn.cuda()

##############训练集 for CNN ############################
'''
for i_tr in range(len(train_patch_ind)):
    i_train=train_patch_ind[i_tr]
    r=i_train//colnum
    c=i_train%colnum
    print( r*colnum+c )
    iput_img= iput_img_original[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth]
    
    gt      = gt_original      [r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth,:]                           
    adj,patch,labels,adj_pg= utils9.read_img(rownum,colnum,iput_img_original,gt_original,iput_img,gt,n_segments,max_n_superpxiel,patch_size)
    
    if i_tr==0:
        train_patch=patch
        train_labels=labels
    else:
        train_patch=torch.cat((train_patch,patch))
        train_labels=torch.cat((train_labels,labels))
    
    #train_patch[i_tr*max_n_superpxiel:(i_tr+1)*max_n_superpxiel,:,:,:]=patch
    #train_labels[i_tr*max_n_superpxiel:(i_tr+1)*max_n_superpxiel]=labels

# 打乱顺序
index_train = [i for i in range(len(train_patch))]
random.shuffle(index_train)
train_patch  = train_patch[index_train]/255.
train_labels = train_labels[index_train]

####################训练CNN#################################
def train(epoch,train_patch,train_labels):
    t = time.time()
    model_cnn.train()
    optimizer_cnn.zero_grad()
    for i_t in range (len(train_patch)//batchsize):
        output = model_cnn(train_patch[i_t*batchsize:(i_t+1)*batchsize].cuda())
        loss_train = F.nll_loss(output, train_labels[i_t*batchsize:(i_t+1)*batchsize].cuda())
        acc_train = utils9.accuracy(output, train_labels[i_t*batchsize:(i_t+1)*batchsize].cuda())
        loss_train.backward()
        optimizer_cnn.step()
        
    if (epoch == 149):
        state = {'net':model_cnn.state_dict(), 'optimizer':optimizer_cnn.state_dict(), 'epoch':epoch}
        
        torch.save(state, './model_cnn_shanxipucheng.pth')
        print("saved the model's parameter")         
    else :
        print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),'time:{:.4f}s'.format(time.time() - t))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch,train_patch,train_labels)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

###############测试集########################

num_superpixel=[]
for i_tr in range(len(test_patch_ind)):
    i_test=test_patch_ind[i_tr]
    r=i_test//colnum
    c=i_test%colnum
    print( r*colnum+c )
    iput_img= iput_img_original[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth]
    gt      = gt_original      [r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth,:]                           
    adj,patch,labels,adj_pg= utils9.read_img(rownum,colnum,iput_img_original,gt_original,iput_img,gt,n_segments,max_n_superpxiel,patch_size)
    
    if i_tr==0:
        test_patch=patch
        test_labels=labels
        num_superpixel.append(len(labels))
    else:
        test_patch=torch.cat((test_patch,patch))
        test_labels=torch.cat((test_labels,labels))
        num_superpixel.append(len(labels))
print(num_superpixel)
test_patch  = test_patch/255.
test_labels = test_labels

##########################测试CNN######################################

checkpoint = torch.load('./model_cnn_shanxipucheng.pth')
model_cnn.load_state_dict(checkpoint['net'])
def test(test_patch,test_labels,test_patch_ind):
    t = time.time()
    model_cnn.eval()
        
    import os
    if (os.path.exists('./test_results_CNN_shanxipucheng')==False):
        os.makedirs('./test_results_CNN_shanxipucheng')
    else:
        shutil.rmtree('./test_results_CNN_shanxipucheng')
        print("delete the previous results")
        os.makedirs('./test_results_CNN_shanxipucheng')
        
    
    for i_t in range (len(test_patch_ind)):
        print("test the %d th image, Finished!" %i_t)
        if i_t==0:
            start_ind=0
            end_ind=num_superpixel[i_t]
        else:
            start_ind=end_ind
            end_ind=end_ind+num_superpixel[i_t]
            
        output = model_cnn(test_patch[start_ind:end_ind].cuda())
        all_test_out=output.max(1)[1].detach().cpu().numpy()    
        np.save('./test_results_CNN_shanxipucheng/CNN_'+str(test_patch_ind[i_t])+'.npy',all_test_out)
        try:
            acc_test = acc_test+utils9.accuracy(output, test_labels[start_ind:end_ind].cuda())
        except:
            acc_test = utils9.accuracy(output, test_labels[start_ind:end_ind].cuda())
        
    print('acc_test: {:.4f}'.format(acc_test.item()/(len(test_patch_ind)),'time:{:.4f}s'.format(time.time() - t)))
test(test_patch,test_labels,test_patch_ind)
'''
# pixel-level evaluation
print("pixel-level evaluation")
all_test_out=[]
for j_test in test_patch_ind:
    all_test_out.append(np.load('./test_results_CNN_shanxipucheng/CNN_'+str(j_test)+'.npy'))

all_test_out=np.array(all_test_out)    
gt_numerical=np.load('gt_numerical_shanxipucheng.npy')
utils9.pixel_report(y_pred= all_test_out, gt_numerical = gt_numerical,n_segments=n_segments,max_n_superpxiel=max_n_superpxiel,rownum=rownum,colnum=colnum,train_patch_ind=train_patch_ind,test_patch_ind=test_patch_ind)

print("Test Finished!")