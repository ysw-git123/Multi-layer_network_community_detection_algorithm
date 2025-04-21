import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.cluster import KMeans

# In[]

def load_data(dataset='cora1'):
    torch.manual_seed(2023)
    if dataset=='cora1':
        cite=pd.read_csv("E:/NEW学习/LW/图神经网络/GCN/数据/cora/cora.cites",sep='\t',header=None)
        content=pd.read_csv("E:/NEW学习/LW/图神经网络/GCN/数据/cora/cora.content",sep='\t',header=None)
        # In[] 压缩成三类了
        content=content[(content[1434]=='Reinforcement_Learning')|(content[1434]=='Rule_Learning')|(content[1434]=='Theory')]
        #Counter(content.iloc[:,-1])

        # In[] 构造
               
        G = nx.Graph()
        for i in range(len(cite)):
           # print(i)
            edge=tuple(cite.iloc[i,:].values)
            G.add_edge(*edge)       
        
        adj1=nx.to_numpy_array(G)  #有重复结点，citeseer有自环
        node=content.iloc[:,0].values
        nodes=G.nodes
        nodes=[i for i in nodes]
        
        adj_idx=[]
        for i in range(len(nodes)):
            for j in range(len(node)):
                if nodes[i]==node[j]:
                    adj_idx.append(i)
                continue
        nodes=np.array(nodes)
        nodes1=nodes[adj_idx]
        idx=[]
        #node_name=[]
        node_idx=[]
        for i,k in enumerate(nodes1):
            for j in range(len(content)):
                 if content.iloc[j,0]==k:
                     idx.append(j)
                     #print('i',i,'k',k)
                     node_idx.append(i)
                     
        feature=content.iloc[:,1:-1].values[idx,:]
        node=content.iloc[:,0].values[idx]#同node_name
        label=pd.get_dummies(content.iloc[idx,-1])
        
        sim2=np.zeros((len(node),len(node)))
        sum_feature=np.sum(feature,axis=1)
        for i in range(len(node)):
            for j in range(i+1,len(node)):
                sim2[i,j]=np.dot(feature[i,:],feature[j,:])/np.sqrt(sum_feature[i]*sum_feature[j])
                #print(adj2[i,j])
        sim2=sim2+sim2.T

        adj2=np.zeros((len(node),len(node)))
        for i1 in range(len(node)):
            for i2 in range(i1+1,len(node)):
                #print(i1,i2)
                adj2[i1,i2]=np.random.binomial(1, sim2[i1,i2], size=None) 
        adj2=adj2+adj2.T#
        adj_1=adj1[adj_idx,:][:,adj_idx]
        adj_1=adj_1+np.eye(adj_1.shape[0])
        adj2=adj2+np.eye(adj2.shape[0])
        A=np.array([adj_1,adj2])
        Al=A[0]+A[1]
        return A,Al,content.iloc[idx,-1],label
    elif dataset=='cora2':
        cite=pd.read_csv("E:/NEW学习/LW/图神经网络/GCN/数据/cora/cora.cites",sep='\t',header=None)
        content=pd.read_csv("E:/NEW学习/LW/图神经网络/GCN/数据/cora/cora.content",sep='\t',header=None)
        # In[] 压缩成四类了
        content=content[(content[1434]=='Probabilistic_Methods')|(content[1434]=='Theory')|(content[1434]=='Genetic_Algorithms')|(content[1434]=='Case_Based')]
        #Counter(content.iloc[:,-1])

        # In[] 构造
               
        G = nx.Graph()
        for i in range(len(cite)):
            edge=tuple(cite.iloc[i,:].values)
            G.add_edge(*edge)       
        
        adj1=nx.to_numpy_array(G)  #有重复结点，citeseer有自环
        node=content.iloc[:,0].values
        nodes=G.nodes
        nodes=[i for i in nodes]
        
        adj_idx=[]
        for i in range(len(nodes)):
            for j in range(len(node)):
                if nodes[i]==node[j]:
                    adj_idx.append(i)
                continue
        nodes=np.array(nodes)
        nodes1=nodes[adj_idx]
        idx=[]
        #node_name=[]
        node_idx=[]
        for i,k in enumerate(nodes1):
            for j in range(len(content)):
                 if content.iloc[j,0]==k:
                     idx.append(j)
                     #print('i',i,'k',k)
                     node_idx.append(i)
                     
        feature=content.iloc[:,1:-1].values[idx,:]
        node=content.iloc[:,0].values[idx]#同node_name
        label=pd.get_dummies(content.iloc[idx,-1])
        
        sim2=np.zeros((len(node),len(node)))
        sum_feature=np.sum(feature,axis=1)
        for i in range(len(node)):
            for j in range(i+1,len(node)):
                sim2[i,j]=np.dot(feature[i,:],feature[j,:])/np.sqrt(sum_feature[i]*sum_feature[j])
                #print(adj2[i,j])
        sim2=sim2+sim2.T

        adj2=np.zeros((len(node),len(node)))
        for i1 in range(len(node)):
            for i2 in range(i1+1,len(node)):
                #print(i1,i2)
                adj2[i1,i2]=np.random.binomial(1, sim2[i1,i2], size=None) 
        adj2=adj2+adj2.T#+np.eye(adj2.shape[0])
        adj_1=adj1[adj_idx,:][:,adj_idx]
        adj_1=adj_1+np.eye(adj_1.shape[0])
        adj2=adj2+np.eye(adj2.shape[0])

        A=np.array([adj_1,adj2])
        Al=A[0]+A[1]
        return A,Al,content.iloc[idx,-1],label
    elif dataset=='citeseer':
                             
        cite=pd.read_csv("E:/NEW学习/LW/图神经网络/GCN/数据/citeseer-doc-classification/citeseer.cites",sep='\t',header=None)
        content=pd.read_csv("E:/NEW学习/LW/图神经网络/GCN/数据/citeseer-doc-classification/citeseer.content",sep='\t',header=None)
        
        content=content[(content[3704]=='AI')|(content[3704]=='ML')|(content[3704]=='Agents')]
        
        
        G = nx.Graph()
        for i in range(len(cite)):
            edge=tuple(cite.iloc[i,:].values)
            G.add_edge(*edge)       
        
        adj1=nx.to_numpy_array(G)  #有重复结点，citeseer有自环
        node=content.iloc[:,0].values
        nodes=G.nodes
        nodes=[i for i in nodes]
        
        adj_idx=[]
        for i in range(len(nodes)):
            for j in range(len(node)):
                if nodes[i]==node[j]:
                    adj_idx.append(i)
                continue
        nodes=np.array(nodes)
        nodes1=nodes[adj_idx]
        idx=[]
        #node_name=[]
        node_idx=[]
        for i,k in enumerate(nodes1):
            for j in range(len(content)):
                 if content.iloc[j,0]==k:
                     idx.append(j)
                     #print('i',i,'k',k)
                     node_idx.append(i)
                     
        feature=content.iloc[:,1:-1].values[idx,:]
        node=content.iloc[:,0].values[idx]#同node_name
        label=pd.get_dummies(content.iloc[idx,-1])
        #label.columns
        
        sim2=np.zeros((len(node),len(node)))
        sum_feature=np.sum(feature,axis=1)
        for i in range(len(node)):
            for j in range(i+1,len(node)):
                sim2[i,j]=np.dot(feature[i,:],feature[j,:])/np.sqrt(sum_feature[i]*sum_feature[j])
                #print(adj2[i,j])
        sim2=sim2+sim2.T
        
        adj2=np.zeros((len(node),len(node)))
        for i1 in range(len(node)):
            for i2 in range(i1+1,len(node)):
                #print(i1,i2)
                adj2[i1,i2]=np.random.binomial(1, sim2[i1,i2], size=None) 
        adj2=adj2+adj2.T#+np.eye(adj2.shape[0])
        adj_1=adj1[adj_idx,:][:,adj_idx]
        adj_1=adj_1+np.eye(adj_1.shape[0])
        adj2=adj2+np.eye(adj2.shape[0])

        A=np.array([adj_1,adj2])
        Al=A[0]+A[1]
        return A,Al,content.iloc[idx,-1],label
    elif dataset=='simulation 300':
        A,Al,label,q,_=data_sim(n=500,L=3,m=3,K=2,alpha=0.3)
        return A,Al,label,q
            
# In[]
# L=3
# m=3
# n=300
# K=2

def data_sim(n=300,L=3,m=3,K=2,alpha=0.3):
    np.random.seed(123)
    z=[]
    label=[]
    for i in range(m):
        z0=np.random.randint(0,K,n)
        label.append(z0)
        #z.append(tf.one_hot(z0,depth=K).numpy())
        z.append(pd.get_dummies(z0).values)

        
    z=np.array(z,dtype=int)

    num_layer=np.random.randint(0,m,L)
    #num_layer=[1,0,2]
    label=np.array(label)
    label=label[num_layer]
    #pd.DataFrame(q[0,:,0],q[0,:,1])
    kmeans = KMeans(n_clusters=K, random_state=2018)
    kmeans.fit(label.T)
    label_orig= kmeans.predict(label.T)

    Ik=np.eye(K)
    ik=np.ones((K,1))
    B=Ik+alpha*(ik*ik.T-Ik)
    p=0.32/(n/100)#n=200的时候平均度在20左右
    B0=p*B
    A=[]
    Al=np.zeros((n,n))
    for i in range(L):
        a=np.dot(np.dot(z[num_layer[i]],B0),z[num_layer[i]].T)
        #a[a<0.5]=0
        #c=np.zeros((3,3))
        a1=np.zeros((a.shape[0],a.shape[1]))
        for i1 in range(a.shape[0]):
            for i2 in range(i1+1,a.shape[1]):
                #print(i1,i2)
                a1[i1,i2]=np.random.binomial(1, a[i1,i2], size=None) 
        #c=c+c.T+np.eye(3)
        a1=a1+a1.T+np.eye(a.shape[0])
        print('第',i,'层的平均度为：',(a1.sum()-n)/(2*n))
        A.append(a1)
        Al+=a1

    A=np.array(A)
    return A,Al,label_orig,label,num_layer                
      
# In[]标准化
def normalize(mx):
    nmx=torch.zeros((mx.shape[0],mx.shape[1],mx.shape[2]))
    for i in range(mx.shape[0]):
        r_inv = np.power(mx[i].sum(1), -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        nmx[i,:,:]=torch.matmul(r_mat_inv,mx[i])
    return nmx
        
          