import numpy as np
import pandas as pd
from numpy import linalg as la
import scipy.linalg as scln
def P_delta(U,delta):
   # print('hello')
    for i in range(len(U)):
        U[i,:]=U[i,:]*(min(delta,np.sqrt(sum(U[i,:]*U[i,:]))))/np.sqrt(sum(U[i,:]*U[i,:]))
        U[np.isnan(U)]=0
    u,s,vt=scln.svd(U)
   # u_p=u[:,0:r]
    return u

def tensorunfold(a,mode):
    z, x, c = a.shape
    if mode==1:
        a1 = a[0, :, :]
        for i in range(z - 1):
            b1 = a[i + 1, :, :]
            e = np.hstack((a1, b1))
            a1 = e
        return e
 
    elif mode==2:
        a1 = a[0, :, :].T
        for i in range(z - 1):
            b1 = a[i + 1, :, :].T
            e = np.hstack((a1, b1))
            a1 = e
        return e
 
    elif mode==3:
        a1 = a[0, :, :].T.reshape(1, -1)
        for i in range(z - 1):
            b1 = a[i + 1, :, :].T.reshape(1, -1)
            e = np.vstack((a1, b1))
            a1 = e
        return e
    else:
        print("tensor unford mode error")

def mtp1(A,U):
    Q=np.zeros((A.shape[0],U.shape[1],A.shape[2]))
    #Q.shape
    for i2 in range(A.shape[2]):
        for i3 in range(A.shape[0]):
            for j in range(U.shape[1]):
                for i1 in range(A.shape[1]):
                    Q[i3,j,i2]+=A[i3,i1,i2]*U[i1,j]
    return Q

def mtp2(A,U):
    Q=np.zeros((A.shape[0],A.shape[1],U.shape[1]))
    #Q.shape
    for i1 in range(A.shape[1]):
        for i3 in range(A.shape[0]):
            for j in range(U.shape[1]):
                for i2 in range(A.shape[2]):
                    Q[i3,i1,j]+=A[i3,i1,i2]*U[i2,j]
    return Q

def mtp3(A,U):
    Q=np.zeros((U.shape[1],A.shape[1],A.shape[2]))
    #Q.shape
    for i2 in range(A.shape[2]):
        for i1 in range(A.shape[1]):
            for j in range(U.shape[1]):
                for i3 in range(A.shape[0]):
                    Q[j,i1,i2]+=A[i3,i1,i2]*U[i3,j]
    return Q

# In[] 定义train
def train(A,Al):
    U, Sigma, VT = la.svd(Al)
    np.isnan(U).sum()


    #m=num_l    #随机块的个数m确定为网络的层数
    m=A.shape[0]
    r=20  #r是Z的秩，这里估计其

    U0=U[:,0:r]


    deg1=np.zeros((A.shape[1],))
    for l in range(A.shape[0]):
        for j in range(A.shape[2]):
            deg1+=A[l,:,j]
          
    deg2=np.zeros((A.shape[0],))
    for i in range(A.shape[1]):
        for j in range(A.shape[2]):
            deg2+=A[:,i,j]

    delta1=(2*np.sqrt(r)*max(deg1))/np.sqrt(sum(deg1*deg1))
    delta2=(2*np.sqrt(m)*max(deg2))/np.sqrt(sum(deg2*deg2))


    U0_p=P_delta(U0,delta1)
    U0_p=U0_p[:,0:r]


    MA3_1=tensorunfold(A,3)
    MA3_2=np.kron(U0_p,U0_p)
    MA3=np.dot(MA3_1,MA3_2)
    Uw,Sw,VTw=scln.svd(MA3)
    W0=Uw[:,0:m]

    i=0
    i_max=100
    while i<=i_max:
        
        U0_p=P_delta(U0,delta1)
        U0_p=U0_p[:,0:r]
        W0_p=P_delta(W0,delta2)
        W0_p=W0_p[:,0:m]
        i+=1
        if i%5==0:
            print('======第{}次迭代开始======'.format(i))
        t1=mtp2(A,U0_p)
        t1_1=mtp3(t1,W0_p)
        m1=tensorunfold(t1_1,1)
        u1,s1,vt1=scln.svd(m1)
        U0=u1[:,0:r]
        t2=mtp1(A,U0_p)
        t2_2=mtp2(t2,U0_p)
        m2=tensorunfold(t2_2,3)
        u2,s2,vt2=scln.svd(m2)
        W0=u2[:,0:m]
        #i+=1
        if (np.isnan(t1).sum()!=0)|(np.isnan(t2).sum()!=0)|(np.isnan(t1_1).sum()!=0)|(np.isnan(t2_2).sum()!=0):
            print("NAN warning!")
    return U0,W0

# In[] 写入
def write_UW(U0,W0,dataset,nclass,path='Cora/',filename="{}_100_{}_class_{}.xlsx"):
    DF=pd.DataFrame(U0)
    DF.to_excel("E:/LW_code/"+path+filename.format("U"), 'iter_100', index=False,float_format='%.5f')
    DFW=pd.DataFrame(W0)
    DFW.to_excel("E:/LW_code/"+path+filename.format("W",dataset,nclass), 'iter_100', index=False,float_format='%.5f')

# In[]计算准确率，输入必须为numpy
def acc(label,label_pre):
    s=0
    for i in range(len(label)):
        if np.argmax(label[i,:])==np.argmax(label_pre[i,:]):
            s+=1
    return s/len(label),s,i+1
