# Attention-Graph-Convolution-Network-for-Image-Segmentation-in-Big-SAR-Imagery-Data
this code implements the method proposed in paper "Attention Graph Convolution Network for Image Segmentation in Big SAR Imagery Data". if it helps you, please kindly cite this paper. https://doi.org/10.3390/rs11212586

## main.py
train, test our AGCN, and calculate the pixel-level Evaluation Metrics（Kappa, precison, recall, and confusion matrix）. Note: as comparison, this file also provide the code of GCN and GAT. If you want to see the results of GAT or GCN, set two parameters "model_nm" and "model" as GAT or GCN
## generate_gt.py	
generate the ground truth for training the Network
## layers.py
define the layers
## model_cnn_shanxipucheng.pth
the trained feature_extraction_net
## models9.py
define graph convolution Network 
## train_feature_extraction_net_Pucheng.py
train the feature_extraction_net
## utils9.py
define the function for calculating pixel-level Evaluation Metrics

## 
the code runs in python 2.7
networkx                      2.2//
numpy                         1.16.2
Pillow                        5.4.1
scikit-image                  0.14.2
scikit-learn                  0.20.3
scipy                         1.2.1
segraph                       0.5
setuptools                    40.8.0
singledispatch                3.4.0.3
six                           1.12.0
sklearn                       0.0
subprocess32                  3.5.3
toolz                         0.9.0
torch                         0.4.0
torchfile                     0.1.0

## 
any problem please email me : mafeimf@buaa.edu.cn
