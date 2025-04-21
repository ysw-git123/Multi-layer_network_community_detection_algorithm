import numpy as np
import pandas as pd
# import networkx as nx
import matplotlib.pyplot as plt
# #import multinetx as mx
import torch
import torch.nn.functional as F
from torch import nn
# #from early_stopping import EarlyStopping
# #import tensorflow as tf
import math
# from numpy import linalg as la
# import scipy.linalg as scln
# from collections import Counter
from data_loading import load_data,normalize
from evaluate import Q_matrix,caculate_modularity_matrix,Q_value
from layers import GAE1,GAE2,Floss1,Floss2,GAE3_1,GAE3_2
from get_label import get_label_U
from TWIST import train as TWIST_train
from TWIST import  write_UW
from evaluate import NMI

np.random.seed(2023)

import warnings
warnings.filterwarnings('ignore')

def train(A,Al,B,dataset='cora1',epoch=500,mode=1):
    #import matplotlib.pyplot as plt
    adj=torch.from_numpy(A).type(torch.float32)
    #adj=normalize(adj)
    if (B.shape[0]==2)&(mode==1):
        model1=GAE1(nfeatures=B.shape[1],nhide=200) 
        opt=torch.optim.Adam(model1.parameters(),lr=0.005)
        loss_fn=Floss1()
        l=[]
        for i in range(epoch):
            model1.train()
            opt.zero_grad()
            yp=model1(B,adj)
            #yp=torch.stack((yp1,yp2))
            #ypT=torch.transpose(yp,1,2)
            B_HAT=torch.matmul(yp,yp.T)
            B_hat=F.tanh(B_HAT)    #loss=loss_fn(yp_train,yy_train)
            loss=loss_fn(B_hat,B)
            loss.backward()
            opt.step()
            with torch.no_grad():
                if (i+1)%10==0:
                    print('epoch:',i+1,'loss:',loss_fn(B_hat,B).data.item())
                l.append(loss_fn(B_hat,B).data.item())
        plt.show()
        plt.plot(l)
        plt.title('the loss of dataset-{} with model{}'.format(data,mode))
        Z=model1(B,adj)
        return Z
    elif (B.shape[0]==2)&(mode==2):
        model2=GAE2(nfeatures=B.shape[1],nhide=200) 
        opt=torch.optim.Adam(model2.parameters(),lr=0.005)
        loss_fn=Floss2()
        l=[]
        for i in range(epoch):
            model2.train()
            opt.zero_grad()
            yp1,yp2=model2(B,adj)
            yp=torch.stack((yp1,yp2))
            ypT=torch.transpose(yp,1,2)
            B_HAT=torch.matmul(yp,ypT)
            B_hat=F.tanh(B_HAT)    #loss=loss_fn(yp_train,yy_train)
            loss=loss_fn(B_hat,B)
            loss.backward()
            opt.step()
            with torch.no_grad():
                if (i+1)%10==0:
                    #print('epoch:',i,'train loss:',loss_fn(model(xx,uu,adj)[train_item],yy_train).data.item())
                    print('epoch:',i+1,'loss:',loss_fn(B_hat,B).data.item())
                #l.append(loss_fn(model(xx,uu,adj)[train_item],yy_train).data.item())
                l.append(loss_fn(B_hat,B).data.item())
        plt.show()
        plt.plot(l)
        plt.title('the loss of dataset-{} with model{}'.format(data,mode))
        zp=model2(B,adj)
        Z=torch.concat([zp[0],zp[1]],1)
        return Z
    elif (B.shape[0]==3)&(mode==1):
        model3_1=GAE3_1(nfeatures=B.shape[1],nhide=200) 
        opt=torch.optim.Adam(model3_1.parameters(),lr=0.005)
        loss_fn=Floss1()
        l=[]
        for i in range(epoch):
            model3_1.train()
            opt.zero_grad()
            yp=model3_1(B,adj)
            #yp=torch.stack((yp1,yp2))
            #ypT=torch.transpose(yp,1,2)
            B_HAT=torch.matmul(yp,yp.T)
            B_hat=F.tanh(B_HAT)    #loss=loss_fn(yp_train,yy_train)
            loss=loss_fn(B_hat,B)
            loss.backward()
            opt.step()
            with torch.no_grad():
                if (i+1)%10==0:
                    print('epoch:',i+1,'loss:',loss_fn(B_hat,B).data.item())
                l.append(loss_fn(B_hat,B).data.item())
        plt.show()
        plt.plot(l)
        plt.title('the loss of dataset-{} with model{}'.format(data,mode))
        Z=model3_1(B,adj)
        return Z
    elif (B.shape[0]==3)&(mode==2):
        model3_2=GAE3_2(nfeatures=B.shape[1],nhide=200) 
        opt=torch.optim.Adam(model3_2.parameters(),lr=0.005)
        loss_fn=Floss2()
        l=[]
        for i in range(epoch):
            model3_2.train()
            opt.zero_grad()
            yp1,yp2,yp3=model3_2(B,adj)
            yp=torch.stack((yp1,yp2,yp3))
            ypT=torch.transpose(yp,1,2)
            B_HAT=torch.matmul(yp,ypT)
            B_hat=F.tanh(B_HAT)    #loss=loss_fn(yp_train,yy_train)
            loss=loss_fn(B_hat,B)
            loss.backward()
            opt.step()
            with torch.no_grad():
                if (i+1)%10==0:
                    print('epoch:',i+1,'loss:',loss_fn(B_hat,B).data.item())
                l.append(loss_fn(B_hat,B).data.item())
        plt.show()
        plt.plot(l)
        plt.title('the loss of dataset-{} with model{}'.format(data,mode))
        zp=model3_2(B,adj)
        Z=torch.concat([zp[0],zp[1],zp[2]],1)
        return Z
    elif mode=='TWIST':
        U0,W0=TWIST_train(A,Al)
        if dataset=='cora1':
            nclass=3
            write_UW(U0,W0,dataset=dataset,nclass=nclass,path='all+code/TWIST_U/',filename="{}_100_"+dataset+"_class_"+str(nclass)+".xlsx")
        elif dataset=='cora2':
            nclass=4
            write_UW(U0,W0,dataset=dataset,nclass=nclass,path='all+code/TWIST_U/',filename="{}_100_"+dataset+"_class_"+str(nclass)+".xlsx")
        elif dataset=='citeseer':
            nclass=3
            write_UW(U0,W0,dataset=dataset,nclass=nclass,path='all+code/TWIST_U/',filename="{}_100_"+dataset+"_class_"+str(nclass)+".xlsx")
        elif dataset=='simulation 300':
            nclass=2
            write_UW(U0,W0,dataset=dataset,nclass=nclass,path='all+code/TWIST_U/',filename="{}_100_"+dataset+"_class_"+str(nclass)+".xlsx")
        return U0
        
        
# model1(B,adj)
# model2(B,adj)
# train(B,adj,epoch=500,mode=1)

#def main(dataset,model):
def main(A,Al,label,dataset,model):

    #  dataset='simulation 300'
    # model=1
    #A,Al,label,_=load_data(dataset=dataset)
    print('=========='+dataset+'+model'+str(model)+'==========')
    for j in range(A.shape[0]):
        print('第{}层的平均度为：'.format(j+1),(A[j].sum()-A[j].shape[1])/(2*A[j].shape[1]))
    Q=Q_matrix(A)
    #model2=GAE2(nfeatures=Q.shape[1],nhide=200) 

    B=torch.from_numpy(Q).type(torch.float32)

    Z=train(A,Al,B,epoch=500,mode=model)
    if ((dataset=='cora1')|(dataset=='citeseer'))&(model!='TWIST'):
        label_pre=get_label_U(Z.detach().numpy(),nclass=3)
    elif ((dataset=='cora1')|(dataset=='citeseer'))&(model=='TWIST'):
        label_pre=get_label_U(Z,nclass=3)
    elif (dataset=='cora2')&(model!='TWIST'):
        label_pre=get_label_U(Z.detach().numpy(),nclass=4)
    elif (dataset=='cora2')&(model=='TWIST'):
        label_pre=get_label_U(Z,nclass=4)
    elif (dataset=='simulation 300')&((model==1)|(model==2)):
        label_pre=get_label_U(Z.detach().numpy(),nclass=2)
    elif (dataset=='simulation 300')&(model=='TWIST'):
        label_pre=get_label_U(Z,nclass=2)
    z=pd.get_dummies(label_pre).values
    z=torch.from_numpy(z).type(torch.float32)

    Qnm,Qsd=caculate_modularity_matrix(A)

    qnm=Q_value(Qnm,z)
    qsd=Q_value(Qsd,z)
    nmi=NMI(label,label_pre)
    # print(NMI(label,label_pre))
    print('Dataset-{}-model{}:Qnm'.format(dataset,model),qnm.numpy(),'Qsd',qsd.numpy(),'NMI',nmi)   
    return qnm,qsd,nmi,label_pre
    
# In[]
dataset=['citeseer']#['cora1','cora2','citeseer','simulation 300']
model=[1,2,'TWIST']
name=[]
qnm_list=[]
qsd_list=[]
nmi_list=[]
for data in dataset:
    A,Al,label,_=load_data(dataset=data)
    for mode in model:
        name.append(data+'-model'+str(mode))
        qnm,qsd,nmi,_=main(A=A,Al=Al,label=label,dataset=data,model=mode)
        print('*************************************')
        qnm_list.append(qnm.numpy())
        qsd_list.append(qsd.numpy())
        nmi_list.append(nmi)
for i,n in enumerate(name):
    print(n+':','Qnm',qnm_list[i],'Qsd',qsd_list[i],'NMI',nmi_list[i])

# L=[name,qnm_list,qsd_list,nmi_list]
# L=np.array(L)
# np.save('E:/LW_code/all+code/L.npy',L)

# A,Al,label,_=load_data(dataset='cora2')
# # main(dataset='simulation 300',model=1)

# Q=Q_matrix(A)
# #model2=GAE2(nfeatures=Q.shape[1],nhide=200) 

# B=torch.from_numpy(Q).type(torch.float32)
# Z=train(A,Al,B,epoch=500,mode=1)
# label_pre=get_label_U(Z.detach().numpy(),nclass=3)
# # from evaluate import NMI

# NMI(label,label_pre)

# nmi