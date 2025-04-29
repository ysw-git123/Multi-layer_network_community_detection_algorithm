import numpy as np
import torch
import math

def Q_matrix(A):
    Q=np.zeros((A.shape[0],A.shape[1],A.shape[2]))
    for m in range(A.shape[0]):
        Lm=A[m].sum()/2
        km=sum(A[m]).reshape(A.shape[1],-1)
        for i in range(A.shape[1]):
            Q[m]=A[m]-(km*km.T)/(2*Lm)
    return Q

def caculate_modularity_matrix(A):
    Qnm=np.zeros((A.shape[0],A.shape[1],A.shape[2]))
    Qsd=np.zeros((A.shape[0],A.shape[1],A.shape[2]))
    L=A.sum()/2
    k=(sum(A[0])+sum(A[1])).reshape(A.shape[1],-1)
    for m in range(A.shape[0]):
        Lm=A[m].sum()/2
        km=sum(A[m]).reshape(A.shape[1],-1)
        Qnm[m]=A[m]/(2*Lm)-(km*km.T)/(4*Lm*Lm)
        Qsd[m]=A[m]/(2*Lm)-(k*k.T)/(4*L*L)
    return Qnm,Qsd


def Q_value(Q,Z): 
    #Q=Qnm
    #qnm=0
    Q=torch.from_numpy(Q).type(torch.float32)
    K=torch.zeros((Z.shape[1],Z.shape[1]))
    for m in range(Q.shape[0]):
        t1=torch.matmul(Q[m],Z)
        t2=torch.matmul(Z.T,t1)
        K+=t2
    # for i in range(len(t2)):
    #     qnm+=K[i,i]
    return torch.trace(K)/Q.shape[0]

def count_value(Q,Z): 
    #Q=Qnm
    #qnm=0
    Q=Q.type(torch.float32)
    K=torch.zeros(Z.shape[1])
    t1=torch.matmul(Q,Z)
    t2=torch.matmul(Z.T,t1)
    for i in range(len(t2)):
        K[i]=t2[i,i]/(Z[:,i].sum()*Z[:,i].sum())
    return t2,K


# In[]标准化互信息：
def NMI(A,B):
    A = np.array(A)
    B = np.array(B)
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    #print(MIhat)
    return MIhat

# In[]
#acc

def acc(A,B):
    A = np.array(A)
    B = np.array(B)
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    
    xy=[]
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            # px = 1.0*len(idAOccur[0])/total
            # py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            xy.append(pxy)
    Mxy=np.array(xy).reshape(len(A_ids),len(B_ids))        
    #xy=torch.tensor(xy).view(3,3)
    # acc=Mxy.max(axis=1).sum()
    return Mxy.max(axis=1).sum()


# In[]

def KL_sim(feature):
    kl_matrix=torch.zeros(feature.shape[0],feature.shape[0])
    for i in range(feature.shape[0]):
        for j in range(i+1,feature.shape[0]):
            kl_matrix[i,j]=torch.matmul(feature[i,:],torch.log((1+feature[i,:])/(1+feature[j,:])))
    kl_matrix=kl_matrix+kl_matrix.T
    return kl_matrix/torch.log(torch.tensor(2))


def JS_sim(feature):
    js_matrix=torch.zeros(feature.shape[0],feature.shape[0])
    for i in range(feature.shape[0]):
        for j in range(i+1,feature.shape[0]):
            js_matrix[i,j]=torch.matmul(feature[i,:],torch.log(2*(1+feature[i,:])/(2+feature[j,:]+feature[i,:])))+torch.matmul(feature[j,:],torch.log(2*(1+feature[j,:])/(2+feature[j,:]+feature[i,:])))
    js_matrix=js_matrix+js_matrix.T
    return js_matrix/torch.log(torch.tensor(2))


# def KL_sim(feature):
#     kl_matrix=torch.zeros(feature.shape[0],feature.shape[0])
#     for i in range(feature.shape[0]):
#         for j in range(i+1,feature.shape[0]):
#             kl_matrix[i,j]=torch.matmul(feature[i,:],(1+feature[i,:])/(1+feature[j,:]))
#     kl_matrix=kl_matrix+kl_matrix.T
#     return kl_matrix#/torch.log(torch.tensor(2))


# def JS_sim(feature):
#     js_matrix=torch.zeros(feature.shape[0],feature.shape[0])
#     for i in range(feature.shape[0]):
#         for j in range(i+1,feature.shape[0]):
#             js_matrix[i,j]=torch.matmul(feature[i,:],2*(1+feature[i,:])/(feature[j,:]+1))+torch.matmul(feature[j,:],(1+feature[j,:])/(1+feature[i,:]))
#     js_matrix=js_matrix+js_matrix.T
#     return js_matrix#/torch.log(torch.tensor(2))
