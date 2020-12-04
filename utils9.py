# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch

from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import sklearn
from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage import data,color,morphology,measure
from skimage.feature import local_binary_pattern
import os
from sklearn.metrics import classification_report,confusion_matrix
import scipy.sparse as sp

def read_img(rownum,colnum,iput_img_original,gt_original,iput_img,gt,n_segments,max_n_superpxiel,patch_size):
    
    def crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size):
        patch_out=new_input_padding[wi_ipt:wi_ipt+patch_size,hi_ipt:hi_ipt+patch_size]
        return patch_out

    h_ipt=gt_original.shape[0]
    w_ipt=gt_original.shape[1]
    
    rowheight = h_ipt // rownum
    colwidth = w_ipt // colnum
    train_adj=[]
    train_features=[]
    train_labels=[]
    train_patch=[]
    test_adj=[]
    test_features=[]
    test_labels=[]
    test_patch=[]
    for r in range(1):#
        for c in range(1):#
            ALL_DATA_X_L=[]
            ALL_DATA_Y_L=[]

            segments = slic(iput_img, n_segments=n_segments, compactness=0.5)
            out=mark_boundaries(gt,segments)
            segments[segments>(max_n_superpxiel-1)]=max_n_superpxiel-1
            #得到每个super pixel的质心
            segments_label=segments+1  #这里+1，是因为regionprops函数的输入要求是label之后的图像，而label的图像的区域编号是从1开始的
            region_fea=measure.regionprops(segments_label)

            #定义一个边界扩大的patch_size的空矩阵，主要是为了当super pixel位于图像边缘时，    
            new_input_padding=np.zeros((rowheight+patch_size,colwidth+patch_size))
            #把这个iput_img放到new_input_padding中
                        
            new_input_padding[int(patch_size/2):-int(patch_size/2),int(patch_size/2):-int(patch_size/2)]=iput_img
            

            # Create graph of superpixels 
            from segraph import create_graph
            vertices, edges = create_graph(segments)
            #print( r*rownum+c ,len(vertices))
            
            # settings for LBP
            radius = 3
            n_points = 8 * radius
            METHOD = 'uniform'
            lbp = local_binary_pattern(iput_img, n_points, radius, METHOD)
            img_feature=[]
            
            #对所有的super pixel开始循环
            for ind_pixel in range (segments_label.max()):

                #计算当前superpixel的质心，为了生成切片，切片以这个质心为中心
                centriod=np.array(region_fea[ind_pixel].centroid).astype("int32")
                wi_ipt=centriod[0]
                hi_ipt=centriod[1]

                #得到这个超像素的所有像素的坐标，根据坐标能够知道这个超像素在GT图中的所有像素值all_pixels_gt
                #根据所有的像素，得到哪一个像素值最多，例如【0,0,0】最多，那这个超像素的标签就是“河流”

                all_pixels_gt=gt[region_fea[ind_pixel].coords[:,0],region_fea[ind_pixel].coords[:,1]]
                n0 = np.bincount(all_pixels_gt[:,0])
                n1 = np.bincount(all_pixels_gt[:,1])  
                n2 = np.bincount(all_pixels_gt[:,2])  
                gt_of_superp=[n0.argmax(),n1.argmax(),n2.argmax()] #gt_of_superp这个超像素中出现最多次的像素值

                 # red ---urban 
                if gt_of_superp[0]>=200 and gt_of_superp[1]<=50 and gt_of_superp[2]<=50:  
                    ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                    ALL_DATA_Y_L.append(0)

                  # yellow ---farmland   
                elif gt_of_superp[0]>=200 and gt_of_superp[1]>=200 and gt_of_superp[2]<=50: 
                    ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                    ALL_DATA_Y_L.append(1)

  
                else:
                    ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                    ALL_DATA_Y_L.append(1)

                #计算每个超像素的color difference（cd）， color histogram difference (hd） 和 texture disparity （lbpd）
                pixels_img=iput_img[region_fea[ind_pixel].coords[:,0],region_fea[ind_pixel].coords[:,1]] #超像素内所有的像素
                cd=np.mean(pixels_img)
                hd,a=np.histogram(pixels_img, bins=64,range=[0,255])
                lbp_a_supixel=lbp[region_fea[ind_pixel].coords[:,0],region_fea[ind_pixel].coords[:,1]]
                lbp_d,a=np.histogram(lbp_a_supixel.astype(np.int64), bins=n_points,range=[0,n_points])
                #所有超像素的featu10                
                img_feature.append(np.concatenate(([cd],hd,lbp_d)))      
            
            img_feature=np.array(img_feature)
            edges=np.array(edges)
            labels=np.array(ALL_DATA_Y_L)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
            
            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            features = normalize_features(img_feature)
            
            adj_propg=np.linalg.inv(sp.eye(adj.shape[0]).todense()-0.5*normalize_adj(adj).todense())
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            labels = torch.LongTensor(labels)
            
            adj = torch.FloatTensor(np.array(adj.todense()))
            adj_propg= torch.FloatTensor(np.array(adj_propg))
            #features = torch.FloatTensor(np.array(features.todense()))
            
    
            ALL_DATA_X_L=torch.FloatTensor(ALL_DATA_X_L)
            ALL_DATA_X_L=ALL_DATA_X_L.reshape(ALL_DATA_X_L.shape[0],1,ALL_DATA_X_L.shape[1],ALL_DATA_X_L.shape[2])
            

    return adj,ALL_DATA_X_L,labels,adj_propg

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def draw_fun(out_img,label,ind_pixel,region_fea):
    out_img[region_fea[ind_pixel].coords[:,0],region_fea[ind_pixel].coords[:,1]]=label

def pixel_report(y_pred,gt_numerical ,n_segments,max_n_superpxiel,rownum,colnum,train_patch_ind,test_patch_ind,target_name = ['urban','farmland']):
    #得到预测结果共29个out_img,即(rowheight,colwidth,29)
    iput_img_original=image.imread('SAR.jpg')
    h_ipt=iput_img_original.shape[0]
    w_ipt=iput_img_original.shape[1]   
    rowheight = h_ipt // rownum
    colwidth = w_ipt // colnum 
    #得到每个test块的预测值矩阵
    out_img_all=np.zeros(rowheight*colwidth*len(y_pred),dtype="uint8")#存储29个testpatch的预测输出
    
        #然后得到对应的gt的块的预测值矩阵
    gt_all= np.zeros(rowheight*colwidth*len(y_pred),dtype="uint8")
    index_1 = 0
    
    for img_i in range(len(y_pred)):
        print(test_patch_ind[img_i])
        ind_img= test_patch_ind[img_i]
        r=ind_img//colnum
        c=ind_img%colnum
        iput_img= iput_img_original[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth];
        pred_out=y_pred[img_i]
        
        segments = slic(iput_img, n_segments=n_segments, compactness=0.5)     
        segments[segments>(max_n_superpxiel-1)]=max_n_superpxiel-1
        np_segments=np.array(segments)
        out_img=np.zeros([rowheight,colwidth],dtype="uint8")        
        region_fea=measure.regionprops(segments+1)
        #函数draw_fun，在输出out_img中相应的像素点赋予其标签值
        index_2=0  #位于图像区域的super pixel的序号
        #对所有的super pixel开始循环，包括那些没有在成像区域的超像素
        for ind_pixel in range (segments.max()+1):
           #根据我们之前给出的标签，赋予相应的像素值对应的标签值
            
        # black ---river 
            if pred_out[index_2]==1: 
                draw_fun(out_img,1,ind_pixel,region_fea)
                index_2=index_2+1
                # red ---urban area 
            elif pred_out[index_2]==0: 
                draw_fun(out_img,0,ind_pixel,region_fea)
                index_2=index_2+1
        #得到该小块的预测值
        out_img_all[img_i*rowheight*colwidth:(img_i+1)*rowheight*colwidth] = out_img.flatten()
        
        #得到测试块对应的r/c
        ind_img= test_patch_ind[img_i]
        r=ind_img//colnum
        c=ind_img%colnum
        gt_all[img_i*rowheight*colwidth:(img_i+1)*rowheight*colwidth]  = gt_numerical[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth].flatten()
        
    print(sklearn.metrics.cohen_kappa_score(gt_all,out_img_all))
    print(classification_report(gt_all,out_img_all,target_names = target_name,digits=4))
    print(confusion_matrix(gt_all,out_img_all))
    
 
