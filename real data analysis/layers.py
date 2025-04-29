import torch
import torch.nn.functional as F
from torch import nn
import math

class GraphConvolution(nn.Module):
    #图卷积层的作用是接收旧特征并产生新特征
    #因此初始化的时候需要确定两个参数：输入特征的维度与输出特征的维度
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #parameter 作用是将tensor设置为梯度求解，并将其绑定到模型的参数中。
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    #参数的初始化
    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    #定义前向计算（邻居聚合与特征变换）
    #输入是旧特征+邻接矩阵
    #输出是新特征
    def forward(self, input, adj):
        #特征变换
        support = torch.matmul(input, self.weight)
        #邻居聚合
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output    

class GAE1(nn.Module):
    def __init__(self,nfeatures,nhide):
        super(GAE1,self).__init__()
        self.gcn1=GraphConvolution(nfeatures,nhide)
        self.gcn2=GraphConvolution(nfeatures,nhide)
        self.gcn3=GraphConvolution(2*nhide,nhide)
        self.gcn4=GraphConvolution(2*nhide,nhide)
    def forward(self,x,adj):
        h1=F.tanh(self.gcn1(x[0],adj[0]))
        h2=F.tanh(self.gcn2(x[1],adj[1]))
        h=torch.cat([h1,h2],1)
        h3=F.tanh(self.gcn3(h,adj[0]))
        h4=F.tanh(self.gcn4(h,adj[1]))
        x=torch.cat([h3,h4],1)
        return x


class GAE2(nn.Module):
    def __init__(self,nfeatures,nhide):
        super(GAE2,self).__init__()
        self.gcn1=GraphConvolution(nfeatures,nhide)
        self.gcn2=GraphConvolution(nfeatures,nhide)
        self.gcn3=GraphConvolution(2*nhide,nhide)
        self.gcn4=GraphConvolution(2*nhide,nhide)
       
    def forward(self,x,adj):
        h1=F.tanh(self.gcn1(x[0],adj[0]))
        h2=F.tanh(self.gcn2(x[1],adj[1]))
        h=torch.cat([h1,h2],1)
        h3=F.tanh(self.gcn3(h,adj[0]))
        h4=F.tanh(self.gcn4(h,adj[1]))
        return h3,h4

class GAE3_1(nn.Module):
    def __init__(self,nfeatures,nhide):
        super(GAE3_1,self).__init__()
        self.gcn1=GraphConvolution(nfeatures,nhide)
        self.gcn2=GraphConvolution(nfeatures,nhide)
        self.gcn3=GraphConvolution(nfeatures,nhide)
        self.gcn4=GraphConvolution(3*nhide,nhide)
        self.gcn5=GraphConvolution(3*nhide,nhide)
        self.gcn6=GraphConvolution(3*nhide,nhide)
    def forward(self,x,adj):
        h1=F.tanh(self.gcn1(x[0],adj[0]))
        h2=F.tanh(self.gcn2(x[1],adj[1]))
        h3=F.tanh(self.gcn3(x[2],adj[2]))
        h=torch.cat([h1,h2,h3],1)
        h4=F.tanh(self.gcn4(h,adj[0]))
        h5=F.tanh(self.gcn5(h,adj[1]))
        h6=F.tanh(self.gcn6(h,adj[2]))
        x=torch.cat([h4,h5,h6],1)
        return x

class GAE3_2(nn.Module):
    def __init__(self,nfeatures,nhide):
        super(GAE3_2,self).__init__()
        self.gcn1=GraphConvolution(nfeatures,nhide)
        self.gcn2=GraphConvolution(nfeatures,nhide)
        self.gcn3=GraphConvolution(nfeatures,nhide)
        self.gcn4=GraphConvolution(3*nhide,nhide)
        self.gcn5=GraphConvolution(3*nhide,nhide)
        self.gcn6=GraphConvolution(3*nhide,nhide)

    def forward(self,x,adj):
        h1=F.tanh(self.gcn1(x[0],adj[0]))
        h2=F.tanh(self.gcn2(x[1],adj[1]))
        h3=F.tanh(self.gcn3(x[2],adj[2]))
        h=torch.cat([h1,h2,h3],1)
        h4=F.tanh(self.gcn4(h,adj[0]))
        h5=F.tanh(self.gcn5(h,adj[1]))
        h6=F.tanh(self.gcn6(h,adj[2]))
        return h3,h4,h6

class Floss1(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,B_hat,B):
        loss=torch.tensor(0.)
        for l in range(B.shape[0]):
            x=torch.norm(B_hat-B[l],2)
            loss+=x
        return loss


class Floss2(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,B_hat,B):
        loss=torch.tensor(0.)
        for l in range(B.shape[0]):
            x=torch.norm(B_hat[l]-B[l],2)
            loss+=x
        return loss
