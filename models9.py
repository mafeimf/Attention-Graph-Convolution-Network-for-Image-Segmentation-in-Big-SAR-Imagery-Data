#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SpGraphAttentionLayer,GraphAttentionLayer,GraphConvolution
from layers import GraphAttentionLayer_AGCN, GraphLayer2_noAttention
import numpy as np

    
class AGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(AGCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(nfeat,nhid)
        #self.gc2 = GraphConvolution(1000, 200)
        #self.gc3 = GraphConvolution(200, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.attentions = GraphAttentionLayer_AGCN(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        #for i, attention in enumerate(self.attentions):
         #   self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphLayer2_noAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        
        x,attention_adj=self.attentions(x, adj)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, attention_adj))#attention_adj
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.out_att(x, adj)
        
        return F.log_softmax(x, dim=1)



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.netd = nn.Sequential(
                    nn.Conv2d(in_channels=1,out_channels=20,kernel_size=3,stride=1,bias=True),
                    nn.BatchNorm2d(20),
                    nn.LeakyReLU(0.2,inplace=False),
                    
                    nn.Conv2d(20,40,3,2,bias=True),
                    nn.BatchNorm2d(40),
                    nn.LeakyReLU(0.2,inplace=False),
                
                    nn.Conv2d(40,80,3,2,bias=True),
                    nn.BatchNorm2d(80),
                    nn.LeakyReLU(0.2,inplace=False),
                    nn.MaxPool2d(2)#14*14-12*12-6*6
                    
                    
        )#31*31-28*28-14*14
                    
                    
        self.line = nn.Sequential(
                    nn.Linear(720,100,bias=True),
                    nn.BatchNorm1d(100),
                    nn.LeakyReLU(0.2,inplace=False)
        )
        self.classfier = nn.Sequential( 
                    nn.Linear(100,2,bias=True),
                    #nn.BatchNorm1d(10),
                    #nn.LogSoftmax()
        )
    def forward(self, x):
        x = self.netd(x)
        x = x.view(x.size(0), -1)
        fea = self.line(x)
        x = self.classfier(fea)
        return F.log_softmax(x, dim=1)#,fea 


class CNN_fea(nn.Module):
    def __init__(self):
        super(CNN_fea, self).__init__()
        self.netd = nn.Sequential(
                    nn.Conv2d(in_channels=1,out_channels=20,kernel_size=3,stride=1,bias=True),
                    nn.BatchNorm2d(20),
                    nn.LeakyReLU(0.2,inplace=False),
                    
                    nn.Conv2d(20,40,3,2,bias=True),
                    nn.BatchNorm2d(40),
                    nn.LeakyReLU(0.2,inplace=False),
                
                    nn.Conv2d(40,80,3,2,bias=True),
                    nn.BatchNorm2d(80),
                    nn.LeakyReLU(0.2,inplace=False),
                    nn.MaxPool2d(2)#14*14-12*12-6*6
                    
                    
        )#31*31-28*28-14*14
                    
                    
        self.line = nn.Sequential(
                    nn.Linear(720,100,bias=True),
                    nn.BatchNorm1d(100),
                    nn.LeakyReLU(0.2,inplace=False)
        )
        self.classfier = nn.Sequential( 
                    nn.Linear(100,2,bias=True),
                    #nn.BatchNorm1d(10),
                    #nn.LogSoftmax()
        )
    def forward(self, x):
        x = self.netd(x)
        x = x.view(x.size(0), -1)
        fea = self.line(x)
        x = self.classfier(fea)
        return F.log_softmax(x, dim=1),fea 

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat,nhid)
        #self.gc2 = GraphConvolution(1000, 200)
        #self.gc3 = GraphConvolution(200, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        '''x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)'''
        x = F.relu(self.gc4(x, adj))
        return F.log_softmax(x, dim=1)

