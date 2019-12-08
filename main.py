# coding: utf-8
#本方法尝试加入Random walks
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
from utils9 import accuracy
from models9 import GAT, SpGAT,AGCN,GCN,CNN_fea
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
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=150,help='Number of epochs to train.')
parser.add_argument('--semi_epochs', type=int, default=100,help='Number of epochs to semi_train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)







###########################set parameters#################################
#model_nm='GAT'
model_nm='GCN'
#model_nm='AGCN' # the proposed method


save_mode_dir='./model_shanxipucheng_'+model_nm+'.pth'

checkpoint_cnn = torch.load('./model_cnn_shanxipucheng.pth')

patch_size=32
num_label=2
n_segments=800
rownum=5
colnum=17
max_sp=800

train_patch_ind=range(35,51)
fea_cnn_dim=100
test_patch_ind=range(rownum*colnum)
for j_tr in train_patch_ind:
    test_patch_ind.remove(j_tr) 

batchsize=max_sp

########################select the name of Model########################
# Model and optimizer
#model = AGCN(nfeat=fea_cnn_dim, nhid=args.hidden, nclass=2, dropout=args.dropout,nheads=args.nb_heads, alpha=args.alpha)
#model = GAT(nfeat=fea_cnn_dim, nhid=args.hidden, nclass=2, dropout=args.dropout,nheads=args.nb_heads, alpha=args.alpha)
model= GCN(nfeat=fea_cnn_dim,   nhid=args.hidden, nclass=2, dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),  lr=args.lr,  weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()

model_cnnfea = CNN_fea( )

if args.cuda:
    model_cnnfea.cuda()
    


model_cnnfea.load_state_dict(checkpoint_cnn['net'])
model_cnnfea.eval()

# #########################Load data######################
train_features=np.load('train_features_Pucheng.npy')
train_labels=np.load('train_labels_Pucheng.npy')
train_adj=np.load('train_adj_Pucheng.npy')
train_adj_pg=np.load('train_adj_pg_Pucheng.npy')

test_features=np.load('test_features_Pucheng.npy')
test_labels=np.load('test_labels_Pucheng.npy')
test_adj=np.load('test_adj_Pucheng.npy')
test_adj_pg=np.load('test_adj_pg_Pucheng.npy')

##################### define Net ########################
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    for i_train in range(len(train_patch_ind)):

        output = model(torch.from_numpy(train_features[i_train]).cuda(), torch.from_numpy(train_adj[i_train]).cuda())#, 
        
        
        loss_train = F.nll_loss(output, torch.from_numpy(train_labels[i_train]).cuda())
        if (i_train==0):
            acc_train = utils9.accuracy(output, torch.from_numpy(train_labels[i_train]).cuda())
        else:
            acc_train =acc_train+ utils9.accuracy(output, torch.from_numpy(train_labels[i_train]).cuda())
        loss_train.backward()
        optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()/len(train_patch_ind)),'time:{:.4f}s'.format(time.time() - t))

####################train and save the Net###########################
# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

for epoch in range(args.epochs):

    train(epoch)
    if (epoch ==(args.epochs-1) ):
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        try:
            os.remove(save_mode_dir)
            torch.save(state, save_mode_dir)
        except:
            torch.save(state, save_mode_dir)  
        print("saved the model's parameter")
        
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


################################load the trained model###################################
t_total = time.time()
checkpoint = torch.load(save_mode_dir)
model.load_state_dict(checkpoint['net'])

###################################test#########################################
def test():
    model.eval()
    all_test_out=[]   
    
    import os
    if (os.path.exists('./test_results_AGCN_shanxipucheng_'+model_nm)==False):  #_Original,_my
        os.makedirs('./test_results_AGCN_shanxipucheng_'+model_nm)
    else:
        shutil.rmtree('./test_results_AGCN_shanxipucheng_'+model_nm)
        print("delete the previous results")
        os.makedirs('./test_results_AGCN_shanxipucheng_'+model_nm)
        
    for i_test in range(len(test_patch_ind)):

        i_train=test_patch_ind[i_test]
        r=i_train//colnum
        c=i_train%colnum
        print("row:",r,"col:",c,"img:", r*colnum+c )
        #iput_img= iput_orig[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth]
        #gt      = gt_original[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth,:]
        
        output = model(torch.from_numpy(test_features[i_test]).cuda(), torch.from_numpy(test_adj[i_test]).cuda())
        
        
        
        np.save('./test_results_AGCN_shanxipucheng_'+model_nm+'/AGCN_'+str(test_patch_ind[i_test])+'.npy',output.max(1)[1].detach().cpu().numpy())



# Testing
test()
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

################################ pixel-level evaluation#########################
print("pixel-level evaluation")
all_test_out=[]
for j_test in test_patch_ind:
    all_test_out.append(np.load('./test_results_AGCN_shanxipucheng_'+model_nm+'/AGCN_'+str(j_test)+'.npy'))
    
all_test_out=np.array(all_test_out)    
gt_numerical=np.load('gt_numerical_shanxipucheng.npy')

utils9.pixel_report(y_pred= all_test_out, gt_numerical = gt_numerical,n_segments=n_segments,max_n_superpxiel=max_sp,rownum=rownum,colnum=colnum,train_patch_ind=train_patch_ind,test_patch_ind=test_patch_ind)
