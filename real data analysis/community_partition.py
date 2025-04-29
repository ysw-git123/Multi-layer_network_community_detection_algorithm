import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jieba
import wordcloud
import networkx as nx

import torch
import torch.nn.functional as F
from torch import nn

#from data_loading import normalize
from evaluate import Q_matrix,caculate_modularity_matrix,Q_value
from evaluate import count_value,KL_sim,JS_sim
from layers import GAE1,GAE2,Floss1,Floss2
from get_label import get_label_U
# from TWIST import train as TWIST_train
from load_real_data import real_graph
from TWIST import train as TWIST_train

import random
from numpy import random as nprand
seed=hash('real')%2**32
nprand.seed(seed)
random.seed(seed)
import matplotlib.patches as mpatches

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

from collections import Counter 
# import h5py
import pickle

import warnings
warnings.filterwarnings('ignore')


A,feature,G,att0=real_graph()  #数据导入+网络构建
att=att0
from multi_graph import LayeredNetworkGraph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
LayeredNetworkGraph([G[1],G[0]],#color=colorlist,# node_labels=node_labels, 
                    ax=ax)#,layout=nx.spring_layout)
ax.set_axis_off()


Al=A[0]+A[1]
Qnm,Qsd=caculate_modularity_matrix(A)

kl_matrix=KL_sim(feature)
js_matrix=JS_sim(feature)

def train(A,B,epoch=500,mode=1):
    adj=A.type(torch.float32)
    if mode==1:
        #import matplotlib.pyplot as plt
        #adj=normalize(adj)
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
        plt.figure()
        plt.plot(np.linspace(1,len(l),len(l)),l)
        plt.title('the loss of real dataset with model{}'.format(mode))
        plt.show()
        Z=model1(B,adj)
        return Z
    elif mode==2:
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
        plt.figure()
        plt.plot(np.linspace(1,len(l),len(l)),l)
        # plt.plot(l)
        plt.title('the loss of real dataset with model{}'.format(mode))
        plt.show()
        zp=model2(B,adj)
        Z=torch.concat([zp[0],zp[1]],1)
        return Z
    elif mode==3:
        U0,W0=TWIST_train(A.numpy(),Al.numpy())
        return torch.from_numpy(U0)


Q=Q_matrix(A)
#model2=GAE2(nfeatures=Q.shape[1],nhide=200) 

B=torch.from_numpy(Q).type(torch.float32)



qnm_list=[]
qsd_list=[]
KL_list=[]
JS_list=[]


for ite in range(100):
    
    Z=train(A,B,epoch=1000,mode=1)
    label_pre=get_label_U(Z.detach().numpy(),nclass=3)

    qnm_l=[]
    qsd_l=[]
    KL_l=[]
    JS_l=[]
    for nclass in range(2,17):
        label_pre=get_label_U(Z.detach().numpy(),nclass=nclass)
        z=pd.get_dummies(label_pre).values
        z=torch.from_numpy(z).type(torch.float32)
        qnm=Q_value(Qnm,z)
        qsd=Q_value(Qsd,z)
        qnm_l.append(qnm.detach().numpy())
        qsd_l.append(qsd.detach().numpy())
        
        
        KL_m,KL_s=count_value(kl_matrix,z)
        JS_m,JS_s=count_value(js_matrix,z)
        
        KL_similarity=torch.sum(KL_s)/nclass
        JS_similarity=torch.sum(JS_s)/nclass
        KL_l.append(KL_similarity.detach().numpy())
        JS_l.append(JS_similarity.detach().numpy())
        
        #print('社区数量为：',nclass,'时，Qnm:',qnm.detach().numpy(),'Qsd',qsd.detach().numpy())
        print('社区数量为：',nclass,'用户的KL相似度：',KL_similarity.detach().numpy(),'用户的JS相似度：',JS_similarity.detach().numpy())
    
    x_data = [f"{i}" for i in range(2, 17)]
    plt.plot(x_data,JS_l,marker='o',linestyle='-')
    plt.plot(x_data,KL_l,marker='s',linestyle='-')
    plt.xlabel("The number of communities")
    plt.ylabel('Similarity index value')
    plt.legend(['JS similarity','KL similarity'])
    
    plt.show()
    
    x_data = [f"{i}" for i in range(2, 17)]
    
    plt.plot(x_data,qnm_l,marker='o',linestyle='-')
    plt.plot(x_data,qsd_l,marker='s',linestyle='-')
    plt.xlabel("The number of communities")
    plt.ylabel('Modularity Q index value')
    plt.legend(['Qnm value','Qsd value'])
    plt.show()
    
    qnm_list.append(qnm_l)
    qsd_list.append(qsd_l)
    KL_list.append(KL_l)
    JS_list.append(JS_l)

Qnm=np.array(qnm_list).mean(0)
Qsd=np.array(qsd_list).mean(0)
KL=np.array(KL_list).mean(0)
JS=np.array(np.array(JS_list)).mean(0)

    

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小

x_data = [f"{i}" for i in range(2, 17)]
plt.plot(x_data,JS,marker='o',linestyle='-')
plt.plot(x_data,KL,marker='s',linestyle='-')
plt.xlabel("The number of communities")
plt.ylabel('Similarity index value')
plt.legend(['JS similarity','KL similarity'])

plt.show()

x_data = [f"{i}" for i in range(2,17)]

plt.plot(x_data,Qnm,marker='o',linestyle='-')
plt.plot(x_data,Qsd,marker='s',linestyle='-')
# plt.xlabel("The number of communities")
# plt.ylabel('Modularity Q index value')
plt.legend(['Qnm value','Qsd value'])
plt.show()

# In[]


from multi_graph import LayeredNetworkGraph

# node_labels = {nn : str(nn) for nn in range(len(node))}
label_pre=get_label_U(Z.detach().numpy(),nclass=3)
color=['lime','yellow','fuchsia','cyan','silver','darkorange','red','blue','y','pink','c','m']
colorlist=np.array(color)[label_pre.astype(int)]
# colorlist=np.array(color)[np.hstack([np.ones(322),np.zeros(400)]).astype(int)]
# node=G[0].nodes
# node_labels = {nn : colorlist[nn] for nn in range(len(node))}

# colorlist=np.hstack([colorlist,colorlist])
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
LayeredNetworkGraph([G[1],G[0]],color=colorlist,# node_labels=node_labels, 
                    ax=ax)#,layout=nx.spring_layout)
#这里G是一个list
ax.set_axis_off()
plt.title('class='+str(3),y=0.96,loc='center',fontsize=12)
plt.show()

# nx.draw(G[1],node_color=colorlist,node_size=6)
# plt.show()
# Counter(label_pre)
nx.draw(G[0],node_color=colorlist,node_size=2.6)
plt.show()

# In[]

# Z=train(A,B,epoch=500,mode=1)
label_pre=get_label_U(Z.detach().numpy(),nclass=3)

qnm_l=[]
qsd_l=[]
KL_l=[]
JS_l=[]
for nclass in range(2,17):
    label_pre=get_label_U(Z.detach().numpy(),nclass=nclass)
    z=pd.get_dummies(label_pre).values
    z=torch.from_numpy(z).type(torch.float32)
    qnm=Q_value(Qnm,z)
    qsd=Q_value(Qsd,z)
    qnm_l.append(qnm.detach().numpy())
    qsd_l.append(qsd.detach().numpy())
    
    
    KL_m,KL_s=count_value(kl_matrix,z)
    JS_m,JS_s=count_value(js_matrix,z)
    
    KL_similarity=torch.sum(KL_s)/nclass
    JS_similarity=torch.sum(JS_s)/nclass
    KL_l.append(KL_similarity.detach().numpy())
    JS_l.append(JS_similarity.detach().numpy())
    
    #print('社区数量为：',nclass,'时，Qnm:',qnm.detach().numpy(),'Qsd',qsd.detach().numpy())
    print('社区数量为：',nclass,'用户的KL相似度：',KL_similarity.detach().numpy(),'用户的JS相似度：',JS_similarity.detach().numpy())

x_data = [f"{i}" for i in range(2, 17)]
plt.plot(x_data,JS_l,marker='o',linestyle='-')
plt.plot(x_data,KL_l,marker='s',linestyle='-')
plt.xlabel("The number of communities")
plt.ylabel('Similarity index value')
plt.legend(['JS similarity','KL similarity'])

plt.show()

x_data = [f"{i}" for i in range(2, 17)]

plt.plot(x_data,qnm_l,marker='o',linestyle='-')
plt.plot(x_data,qsd_l,marker='s',linestyle='-')
plt.xlabel("The number of communities")
plt.ylabel('Modularity Q index value')
plt.legend(['Qnm value','Qsd value'])
plt.show()






# In[]

from sklearn.manifold import TSNE 

# Z=train(A,B,epoch=500,mode=1)

color=['yellow','lime','cyan','fuchsia','silver','darkorange','red','blue','y','pink','c','m']
tsne = TSNE(n_components=2) 
z_tsne = tsne.fit_transform(Z.detach().numpy()) 
# z_tsne_data = np.vstack((z_tsne.T, label_pre.astype(int))).T 
def draw_tsne(z_tsne,label_pre):
    c=np.array(color)[label_pre.astype(int)]
    # df_tsne = pd.DataFrame(z_tsne_data, columns=['Dim1','Dim2','Class']) 
    # df_tsne['Class']
    # sns.set(style="ticks", color_codes=True)
    # sns.scatterplot(data=df_tsne, hue='Class', x='Dim1', y='Dim2',c=c) 
    plt.scatter(z_tsne[:,0],z_tsne[:,1],c=c,s=2.66)
    labels = []
    plt.xticks(labels)
    plt.yticks(labels)
    # plt.xlabel('Dim 1')
    # plt.ylabel('Dim 2')
    #plt.show()

for nclass in range(2,11):
    n=330+nclass-1
    plt.subplot(n)
    label_pre=get_label_U(Z.detach().numpy(),nclass=nclass)
    draw_tsne(z_tsne,label_pre)
    plt.title('class:'+str(nclass),y=0.93,loc='center',fontsize=12)
plt.show()

# In[]


def centrality(A,layers,node):
    A=A.type(torch.int32)
    minc=sum(A[0]&A[1])
    a0=sum(A[0])
    a1=sum(A[1])
    maxc=torch.zeros(len(a0))
    for i in range(len(a0)):
        maxc[i]=max(a0[i],a1[i])
    return (minc+maxc)/(layers*(node-1)),maxc


def page_rank(Al,v,p=0.85):
    num = Al.sum(axis = 0)
    M=Al/num
    i = 0
    v=cen
    while 1:
        v1 = p*torch.matmul(M,v) + (1-p) * v
        if torch.abs((v-v1)).sum() < 0.001:
            break
        else:
            v = v1
        i += 1
        if i==200:break
    print('求pr值迭代{}次'.format(i))
    return v


cen,neighbor=centrality(A,A.shape[0],A.shape[1])
prk=page_rank(Al,cen)
c,q=torch.sort(prk,descending=True)

# plt.subplot(2,1,1)
plt.figure(figsize=(20,10)) 
plt.plot(neighbor[q[0:100]],marker='o',markersize=6.88)
plt.title('number of neighbors',y=1,loc='center',fontsize=38)
plt.xlabel('ordinal number of nodes sorted by influence descending',fontsize=28)
plt.show()
# plt.subplot(2,1,2)
plt.figure(figsize=(20,10)) 
plt.plot(cen[q[0:100]],marker='o',markersize=6.88)
plt.title('degree centrality of multi-layer networks',y=1,loc='center',fontsize=38)
plt.xlabel('ordinal number of nodes sorted by influence descending',fontsize=28)
plt.show()



# In[] 判断划分是否有社区发生了信息茧房现象，这里给出了一组划分方案

def high_influence(label_pre,classes):
    c0,q0=torch.sort(prk[np.where(label_pre==classes)[0]],descending=True)
    print('community {}'.format(classes),Counter(np.array(att)[np.where(label_pre==classes)[0][q0[0:50]]]))
    return c0,q0
# # # 训练一组数据
# # Z=train(A,B,epoch=1000,mode=1)
# # label_pre=get_label_U(Z.detach().numpy(),nclass=3)

# #计算各组的观点比例
# # att1_0=Counter(np.array(att)[np.where(label_pre==0)[0]])[0]/Counter(label_pre)[0]
# # att1_1=Counter(np.array(att)[np.where(label_pre==0)[0]])[1]/Counter(label_pre)[0]
# # att1_2=Counter(np.array(att)[np.where(label_pre==0)[0]])[2]/Counter(label_pre)[0]

# # att2_0=Counter(np.array(att)[np.where(label_pre==1)[0]])[0]/Counter(label_pre)[1]
# # att2_1=Counter(np.array(att)[np.where(label_pre==1)[0]])[1]/Counter(label_pre)[1]
# # att2_2=Counter(np.array(att)[np.where(label_pre==1)[0]])[2]/Counter(label_pre)[1]

# # att3_0=Counter(np.array(att)[np.where(label_pre==2)[0]])[0]/Counter(label_pre)[2]
# # att3_1=Counter(np.array(att)[np.where(label_pre==2)[0]])[1]/Counter(label_pre)[2]
# # att3_2=Counter(np.array(att)[np.where(label_pre==2)[0]])[2]/Counter(label_pre)[2]

# comm1=np.array([0])
# comm2=np.array([0])
# ite=1
# while ((comm1.shape[0]==1)&(comm2.shape[0]==1))|ite<100:
#     print('还没找到合适的社区划分，再训练一组划分')
#     # 训练一组数据
#     Z=train(A,B,epoch=1000,mode=1)
#     label_pre=get_label_U(Z.detach().numpy(),nclass=3)
    
#     qnm_l=[]
#     qsd_l=[]
#     Q_l=[]
#     for nclass in range(2,17):
#         label_pre=get_label_U(Z.detach().numpy(),nclass=nclass)
#         z=pd.get_dummies(label_pre).values
#         z=torch.from_numpy(z).type(torch.float32)
#         qnm=Q_value(Qnm,z)
#         qsd=Q_value(Qsd,z)
#         qnm_l.append(qnm.detach().numpy())
#         qsd_l.append(qsd.detach().numpy())
        
#     Q_l=np.array(qnm_l)+np.array(qsd_l)
#     if np.argmax(Q_l)==1: 
#         att_0=[]
#         att_1=[]
#         att_2=[]
        
#         for j in range(3):
#             att_0.append(Counter(np.array(att)[np.where(label_pre==j)[0]])[0]/Counter(label_pre)[j])
#             att_1.append(Counter(np.array(att)[np.where(label_pre==j)[0]])[1]/Counter(label_pre)[j])
#             att_2.append(Counter(np.array(att)[np.where(label_pre==j)[0]])[2]/Counter(label_pre)[j])
#             if att_0[-1]>2/3: 
#                 print('Find Information Cocoon in Community {}'.format(j+1))
#                 c0,q0=high_influence(label_pre,classes=j)
                
#                 sentiments=np.array(att)[np.where(label_pre==j)[0]][q0[0:30]]
                
#                 color_map = {0: 'skyblue', 1: 'mediumspringgreen', 2: 'khaki'}
#                 bar_colors = [color_map[sent] for sent in sentiments]
                
#                 plt.figure()
#                 plt.bar(range(len(c0[0:30])),c0[0:30],color=bar_colors)
                
#                 plt.xlabel("ordinal number of nodes sorted by influence descending")
#                 plt.ylabel("influence score")
#                 plt.title("top 15% in Community1")
                
#                 legend_elements = [
#                     mpatches.Patch(color=color_map[0], label='negative'),
#                     mpatches.Patch(color=color_map[1], label='positive'),
#                     mpatches.Patch(color=color_map[2], label='neutral')
#                 ]
#                 plt.legend(
#                     handles=legend_elements,
#                     loc='lower center',
#                     bbox_to_anchor=(0.5, -0.25),
#                     ncol=3,
#                     frameon=False
#                 )
#                 plt.show()
#                 comm1=np.where(label_pre==j)[0]
        
#             elif (2/3<att_1[-1]/att_0[-1]<3/2)|(att_2[-1]>2/3):
#                 print('Comparison community {}'.format(j))
#                 comm2=np.where(label_pre==j)[0]
#             if (comm1.shape[0]>1)&(comm2.shape[0]>1):
#                 break
#     ite+=1
# if ite==100:
#     print('没找到合适的划分，那就导入一组示例吧！')
#     Z=np.load('Z.npy')
#     label_pre=get_label_U(Z,nclass=3)
    
#     c0,q0=high_influence(label_pre,classes=0)
#     _=high_influence(label_pre,classes=1)
#     _=high_influence(label_pre,classes=2)
     
#     sentiments=np.array(att)[np.where(label_pre==0)[0]][q0[0:30]]
    
#     color_map = {0: 'skyblue', 1: 'mediumspringgreen', 2: 'khaki'}
#     bar_colors = [color_map[sent] for sent in sentiments]
    
#     plt.figure()
#     plt.bar(range(len(c0[0:30])),c0[0:30],color=bar_colors)
    
#     plt.xlabel("ordinal number of nodes sorted by influence descending")
#     plt.ylabel("influence score")
#     plt.title("top 15% in Community1")
    
#     legend_elements = [
#         mpatches.Patch(color=color_map[0], label='negative'),
#         mpatches.Patch(color=color_map[1], label='positive'),
#         mpatches.Patch(color=color_map[2], label='neutral')
#     ]
#     plt.legend(
#         handles=legend_elements,
#         loc='lower center',
#         bbox_to_anchor=(0.5, -0.25),
#         ncol=3,
#         frameon=False
#     )
    
#     comm1=np.where(label_pre==0)[0]
#     comm2=np.where(label_pre==1)[0]
#     comm3=np.where(label_pre==2)[0]

# In[] 导入示例
Z=np.load('Z.npy')
label_pre=get_label_U(Z,nclass=3)

group=list([Counter(label_pre)[0],Counter(label_pre)[1],Counter(label_pre)[2]])

att1=Counter(np.array(att)[np.where(label_pre==0)[0]])
att2=Counter(np.array(att)[np.where(label_pre==1)[0]])
att3=Counter(np.array(att)[np.where(label_pre==2)[0]])


c0,q0=high_influence(label_pre,classes=0)
_=high_influence(label_pre,classes=1)
_=high_influence(label_pre,classes=2)
 
sentiments=np.array(att)[np.where(label_pre==0)[0]][q0[0:30]]

color_map = {0: 'skyblue', 1: 'mediumspringgreen', 2: 'khaki'}
bar_colors = [color_map[sent] for sent in sentiments]

plt.bar(range(len(c0[0:30])),c0[0:30],color=bar_colors)

plt.xlabel("ordinal number of nodes sorted by influence descending")
plt.ylabel("influence score")
plt.title("top 15% in Community1")

legend_elements = [
    mpatches.Patch(color=color_map[0], label='negative'),
    mpatches.Patch(color=color_map[1], label='positive'),
    mpatches.Patch(color=color_map[2], label='neutral')
]
plt.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    frameon=False
)

comm1=np.where(label_pre==0)[0]
comm2=np.where(label_pre==1)[0]
comm3=np.where(label_pre==2)[0]


# In[]


def lambda1_cal(A1,l1_att,l2_state,alpha,gamma1):
    lambda1=np.ones(A1.shape[0])
    for i in range(A1.shape[0]):
        if (l1_att[i]==1)&(l2_state[i]==1):
            lambda1[i]=1-torch.prod(1-gamma1*alpha*A1[i,:]*abs((l1_att-1)*(l1_att-2))*0.5).numpy()
        elif (l1_att[i]==1)&(l2_state[i]==0):
            lambda1[i]=1-torch.prod(1-alpha*A1[i,:]*abs((l1_att-1)*(l1_att-2))*0.5).numpy()
        elif (l1_att[i]==0)&(l2_state[i]==1):
            lambda1[i]=1-torch.prod(1-gamma1*alpha*A1[i,:]*abs(l1_att*(l1_att-2))).numpy()
        elif (l1_att[i]==0)&(l2_state[i]==0):
            lambda1[i]=1-torch.prod(1-alpha*A1[i,:]*abs(l1_att*(l1_att-2))).numpy()
        # elif l1_att[i]==2:
        #     lamda1[i]=1
    return lambda1

            
            
def lambda2_cal(A2,l1_att,l2_state,diffRate,gamma2):
    lambda2=np.ones(A2.shape[0])
    for i in range(A2.shape[0]):
        if l1_att[i]==2:
            lambda2[i]=1-torch.prod(1-gamma2*diffRate*A2[i,:]*abs((l1_att-1)*l1_att)*0.5).numpy()
        else:
            lambda2[i]=1-torch.prod(1-diffRate*A2[i,:]*abs((l1_att-1)*l1_att)*0.5).numpy()
    return lambda2
            
           
def lambda_neutral(a1,l1_s,l2_s,alpha,gamma1,flat):
    lambda_neutral=1
    if l1_s==2:
        if (flat==0)&(l2_s==1):
            lambda_neutral=1-torch.prod(1-gamma1*alpha*a1*abs((l1_s-1)*(l1_s-2))*0.5).numpy()
        elif (flat==1)&(l2_s==1):
            lambda_neutral=1-torch.prod(1-gamma1*alpha*a1*abs(l1_s*(l1_s-2))).numpy()
        elif (flat==0)&(l2_s==0):
            lambda_neutral=1-torch.prod(1-alpha*a1*abs((l1_s-1)*(l1_s-2))*0.5).numpy()
        elif (flat==1)&(l2_s==0):
            lambda_neutral=1-torch.prod(1-alpha*a1*abs(l1_s*(l1_s-2))).numpy()
        
    return lambda_neutral            
            
            
            
            
def transfer_P_or_N(A1,l1_att,l2_state,alpha,beta,gamma1,gamma2):
    lambda1=lambda1_cal(A1,l1_att,l2_state,alpha,gamma1)
    # lambda2=lambda2_cal(A2,l1_att,l2_state,diffrate,gamma2)
    for i in range(A1.shape[0]):
        lambda_p=lambda_neutral(A1[i,:],l1_att[i],l2_state[i],alpha,gamma1,flat=1)
        lambda_n=lambda_neutral(A1[i,:],l1_att[i],l2_state[i],alpha,gamma1,flat=0)
        if l1_att[i]==1:
            prob=np.random.uniform(0,1,size=None)
            if prob<=lambda1[i]*beta:
                l1_att[i]=2
            elif lambda1[i]*beta<prob<=lambda1[i]:
                l1_att[i]=0    
        elif l1_att[i]==0:
            prob=np.random.uniform(0,1,size=None)
            if prob<=lambda1[i]*beta:
                l1_att[i]=2
            elif lambda1[i]*beta<prob<=lambda1[i]:
                l1_att[i]=1
        elif l1_att[i]==2:
            prob=np.random.uniform(0,1,size=None)
            if prob<=lambda_p*R1:
                l1_att[i]=1
            elif lambda_p*R1<prob<=lambda_p*R1+lambda_n*R1:
                l1_att[i]=0
    return l1_att

def transfer_S_or_I(A2,l1_att,l2_state,alpha,beta,gamma1,gamma2,diffrate,srate):
    lambda2=lambda2_cal(A2,l1_att,l2_state,diffrate,gamma2)
    for i in range(A2.shape[0]):
        prob=np.random.uniform(0,1,size=None)
        if l2_state[i]==0:
            if prob<lambda2[i]*srate:
                l2_state[i]=1
        else:
            if prob<R2:
                l2_state[i]=0
    return l2_state




main_att=list(dict(sorted(att1.items(),key = lambda x:x[1],reverse = True)).keys())[0]
def exchange_att(commu,main_att,high_influ,ratio,l1_att):
    for i in range(int(ratio*commu.shape[0])):
        print(l1_att[commu[high_influ[i]]])
        if l1_att[commu[high_influ[i]]]==main_att:
            l1_att[commu[high_influ[i]]]=np.abs(main_att-1)#*np.ones(int(ratio*commu.shape[0]))
    return l1_att    
    
# l1_att[comm1[q0[0:int(eta*comm1.shape[0])]]]=np.abs(main_att-1)*np.ones(int(eta*comm1.shape[0]))    


def show_iteration(theta,eta,alpha,beta,diffrate,srate,R1,R2,gamma1,gamma2,commu,main_att,high_influ,att,result_commu):
    # commu 施加干预措施的社区, result_commu 希望输出哪个社区的比例
    sensitive_ratio=[theta]#用于存Layer2中不敏感节点的比例
    positive_ratio=[]#用于存Layer1中积极节点的比例
    negative_ratio=[]#用于存Layer1中消极节点的比例
    # 初始化layer2
    l2_state=np.concatenate([np.zeros(int(theta*A.shape[1])),np.ones(A.shape[1]-int(theta*A.shape[1]))])[np.random.permutation(A.shape[1])]
    # 初始化Layer1
    l1_att=np.array(att)
    l1_att=exchange_att(commu=commu,main_att=main_att,high_influ=high_influ,ratio=eta,l1_att=l1_att)    
    l1_att=np.array(l1_att)
    
    Nc=0
    if result_commu==0:
        Nc=comm1.shape[0]
    elif result_commu==1:
        Nc=comm2.shape[0]
    elif result_commu==2:
        Nc=comm3.shape[0]
    
    print('********before start',Counter(att1))
    print(Counter(np.array(l1_att)[np.where(label_pre==0)[0]]))     
    t=0
    while t<200:
        
        l1=transfer_P_or_N(A[0,:,:],l1_att,l2_state,alpha,beta,gamma1,gamma2)
        l2_state=transfer_S_or_I(A[1,:,:],l1_att,l2_state,alpha,beta,gamma1,gamma2,diffrate,srate)
        l1_att=l1 
        
        l1_att_0=np.count_nonzero(l1_att[np.where(label_pre==result_commu)[0]] == 0) 
        l1_att_1=np.count_nonzero(l1_att[np.where(label_pre==result_commu)[0]] == 1) 
        l1_att_2=np.count_nonzero(l1_att[np.where(label_pre==result_commu)[0]] == 2) 
        l2_state_0=np.count_nonzero(l2_state[np.where(label_pre==0)[0]] == 0) 
        
        
        sensitive_ratio.append(l2_state_0/Nc)# 存不敏感节点的比例
        positive_ratio.append(l1_att_1/Nc)
        negative_ratio.append(l1_att_0/Nc)
        
        t=t+1          
        if t%10==0:
            print('=====the {}th iteration======'.format(t))
            print(Counter(np.array(l1_att)[np.where(label_pre==0)[0]]))  
            print(Counter(np.array(l2_state)[np.where(label_pre==0)[0]])) 
    # plt.plot(sensitive_ratio,marker='.',markersize=5)
    return sensitive_ratio,positive_ratio,negative_ratio

theta=0.1 #敏感性系数
eta=0.1 #干预比例

alpha=0.3
beta=0.2
diffrate=0.3
srate=0.2
R1=0.3
R2=0.2
gamma1=1.0
gamma2=1.0

# sensitive_ratio,_=show_iteration(theta=theta,eta=eta,alpha=alpha,beta=beta,diffrate=diffrate,srate=srate,R1=R1,R2=R2,gamma1=gamma1,gamma2=gamma2,commu=comm1,main_att=main_att,high_influ=q0,att=att)
# In[] 稳定性分析

Eta=[0.05,0.10,0.15,0.20,0.25]

for eta in Eta:
    print('eta={}'.format(eta))
    sensitive_ratio,_,_=show_iteration(theta=theta,eta=eta,alpha=alpha,beta=beta,diffrate=diffrate,srate=srate,R1=R1,R2=R2,gamma1=gamma1,gamma2=gamma2,commu=comm1,main_att=main_att,high_influ=q0,att=att0,result_commu=0)
    plt.plot(sensitive_ratio,marker='.',markersize=5,c= (1,eta*4,0.1),label = r'$\eta={:.2f}$'.format(eta))
    plt.xlabel('Iteration Steps')
    plt.ylabel('Ratio')
    plt.yticks(np.arange(0,1.2,0.2),['0%','20%','40%','60%','80%','100%'])
plt.legend(fontsize=10)
plt.show()
 

# In[] Intervention Effects Analysis
## community1

percent_negative=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 0)/comm1.shape[0]]
percent_positive=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 1)/comm1.shape[0]]
percent_neutral=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 2)/comm1.shape[0]]
# Theta=[0.05,0.10,0.15,0.20,0.25]

for eta in Eta:
    print('eta={}'.format(eta))
    sensitive_ratio,positive_ratio,negative_ratio=show_iteration(theta=theta,eta=eta,alpha=alpha,beta=beta,diffrate=diffrate,srate=srate,R1=R1,R2=R2,gamma1=gamma1,gamma2=gamma2,commu=comm1,main_att=main_att,high_influ=q0,att=att0,result_commu=0)
    percent_negative.append(negative_ratio[-1])
    percent_positive.append(positive_ratio[-1])
    percent_neutral.append(1-positive_ratio[-1]-negative_ratio[-1])



N=len(percent_negative)
ind = np.arange(N)    # the x locations for the groups
width = 0.7       # the width of the bars: can also be len(x) sequence

plt.figure(figsize=(8,5)) 

color=['skyblue']*N+['khaki']*N+['mediumspringgreen']*N
p1 = plt.bar(ind, percent_negative, width,color=color)#, yerr=menStd)
p2 = plt.bar(ind, percent_positive, width, bottom=percent_negative,color=color[N:2*N])#, yerr=womenStd)
p3 = plt.bar(ind, percent_neutral, width, bottom=np.array(percent_negative)+np.array(percent_positive),color=color[2*N:3*N])
 
plt.ylabel('percentage(%)')
plt.xlabel('Proportion of intervention')
plt.title('Effect of Intervention in Community1')
plt.xticks(ind, ('0%', '5%', '10%', '15%', '20%','25%'))
plt.yticks(np.arange(0, 1.2, 0.2),['0','20','40','60','80','100'])
plt.legend((p1[0], p2[0], p3[0]), ('negative', 'positive', 'neutral'),
           ncol=3,bbox_to_anchor=(0.5, -0.1), loc=8, borderaxespad=-3,frameon=False)
plt.plot(percent_negative,marker='o',markersize=6.6,color='fuchsia',linewidth=1.6)
plt.plot(np.array(percent_negative)+np.array(percent_positive),marker='o',markersize=6.6,color='coral',linewidth=1.6)

plt.show()


# In[]

## community2

px0=[np.count_nonzero(np.array(att)[np.where(label_pre==1)[0]] == 0)/comm2.shape[0]]
px1=[np.count_nonzero(np.array(att)[np.where(label_pre==1)[0]] == 1)/comm2.shape[0]]
px2=[np.count_nonzero(np.array(att)[np.where(label_pre==1)[0]] == 2)/comm2.shape[0]]
# Theta=[0.05,0.10,0.15,0.20,0.25]

for eta in Eta:
    print('eta={}'.format(eta))
    sensitive_ratio,positive_ratio_2,negative_ratio_2=show_iteration(theta=theta,eta=eta,alpha=alpha,beta=beta,diffrate=diffrate,srate=srate,R1=R1,R2=R2,gamma1=gamma1,gamma2=gamma2,commu=comm1,main_att=main_att,high_influ=q0,att=att0,result_commu=1)
    px0.append(negative_ratio_2[-1])
    px1.append(positive_ratio_2[-1])
    px2.append(1-positive_ratio_2[-1]-negative_ratio_2[-1])



N=len(percent_negative)
ind = np.arange(N)    # the x locations for the groups
width = 0.7       # the width of the bars: can also be len(x) sequence

plt.figure(figsize=(8,5)) 

color=['skyblue']*N+['khaki']*N+['mediumspringgreen']*N
p1 = plt.bar(ind, px0, width,color=color)#, yerr=menStd)
p2 = plt.bar(ind, px1, width, bottom=px0,color=color[N:2*N])#, yerr=womenStd)
p3 = plt.bar(ind, px2, width, bottom=np.array(px0)+np.array(px1),color=color[2*N:3*N])
 
plt.ylabel('percentage(%)')
plt.xlabel('Proportion of intervention')
plt.title('Effect of Intervention in Community2')
plt.xticks(ind, ('0%', '5%', '10%', '15%', '20%','25%'))
plt.yticks(np.arange(0, 1.2, 0.2),['0','20','40','60','80','100'])
plt.legend((p1[0], p2[0], p3[0]), ('negative', 'positive', 'neutral'),
           ncol=3,bbox_to_anchor=(0.5, -0.1), loc=8, borderaxespad=-3,frameon=False)
plt.plot(px0,marker='o',markersize=6.6,color='fuchsia',linewidth=1.6)
plt.plot(np.array(px0)+np.array(px1),marker='o',markersize=6.6,color='coral',linewidth=1.6)

plt.show()

# In[] Sensitive Analysis
## theta, eta in community 1
eta=0.1 #干预比例

alpha=0.3
beta=0.2
diffrate=0.3
srate=0.2
R1=0.3
R2=0.2
gamma1=1.5
gamma2=1.5

plt.figure(figsize=(9.8,6.8)) 

for theta in [0.1,0.2,0.3,0.4,0.5]:

    percent_negative=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 0)/comm1.shape[0]]
    percent_positive=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 1)/comm1.shape[0]]
    percent_neutral=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 2)/comm1.shape[0]]
    # Theta=[0.05,0.10,0.15,0.20,0.25]

    for eta in Eta:
        print('theta={}, eta={}'.format(theta,eta))
        sensitive_ratio,positive_ratio,negative_ratio=show_iteration(theta=theta,eta=eta,alpha=alpha,beta=beta,diffrate=diffrate,srate=srate,R1=R1,R2=R2,gamma1=gamma1,gamma2=gamma2,commu=comm1,main_att=main_att,high_influ=q0,att=att0,result_commu=0)
        percent_negative.append(negative_ratio[-1])
        percent_positive.append(positive_ratio[-1])
        percent_neutral.append(1-positive_ratio[-1]-negative_ratio[-1])
    
    plt.plot(percent_negative,marker='*',c= (1,1-(theta-0.08)*2,0.1),markersize=8.88,linewidth=1.88,label = r'$\theta={}$'.format(theta))
    plt.yticks(np.arange(0,1.2,0.2),[0,20,40,60,80,100])
    plt.xticks(np.arange(0, 6, 1),['0%','5%','10%','15%','20%','25%'])
    plt.ylabel('percentage(%)')
    plt.xlabel('Proportion of intervention')
    plt.title('Sensitivity Analysis in Community 1')
    plt.legend()
plt.show()

## theta, eta in community 2

eta=0.1 #干预比例

alpha=0.3
beta=0.2
diffrate=0.3
srate=0.2
R1=0.3
R2=0.2
gamma1=1.5
gamma2=1.5

plt.figure(figsize=(9.8,6.8)) 

for theta in [0.1,0.2,0.3,0.4,0.5]:

    percent_negative=[np.count_nonzero(np.array(att)[np.where(label_pre==1)[0]] == 0)/comm1.shape[0]]
    percent_positive=[np.count_nonzero(np.array(att)[np.where(label_pre==1)[0]] == 1)/comm1.shape[0]]
    percent_neutral=[np.count_nonzero(np.array(att)[np.where(label_pre==1)[0]] == 2)/comm1.shape[0]]
    # Theta=[0.05,0.10,0.15,0.20,0.25]
    
    for eta in Eta:
        print('theta={}, eta={}'.format(theta,eta))
        sensitive_ratio,positive_ratio,negative_ratio=show_iteration(theta=theta,eta=eta,alpha=alpha,beta=beta,diffrate=diffrate,srate=srate,R1=R1,R2=R2,gamma1=gamma1,gamma2=gamma2,commu=comm1,main_att=main_att,high_influ=q0,att=att0,result_commu=1)
        percent_negative.append(negative_ratio[-1])
        percent_positive.append(positive_ratio[-1])
        percent_neutral.append(1-positive_ratio[-1]-negative_ratio[-1])
    
    plt.plot(percent_negative,marker='*',c= (0.6,1-(theta-0.08)*2,1),markersize=8.88,linewidth=1.88,label = r'$\theta={}$'.format(theta))
    plt.yticks(np.arange(0,1.2,0.2),[0,20,40,60,80,100])
    plt.xticks(np.arange(0, 6, 1),['0%','5%','10%','15%','20%','25%'])
    plt.ylabel('percentage(%)')
    plt.xlabel('Proportion of intervention')
    plt.title('Sensitivity Analysis in Community 2')
    plt.legend()
plt.show()



theta=0.1 #敏感性系数
eta=0.1 #干预比例

alpha=0.3
beta=0.2
diffrate=0.3
srate=0.2
R1=0.3
R2=0.2
gamma1=1.5
gamma2=1.5



plt.figure(figsize=(9.8,6.8)) 

for i1,i2 in [(1.0,1.0),(1.0,1.5),(1.0,2.0),(1.5,1.5),(1.5,2.0),(2.0,2.0)]:
    print(i1,i2)
    
    # percent_negative=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 0)/comm1.shape[0]]
    # percent_positive=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 1)/comm1.shape[0]]
    # percent_neutral=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 2)/comm1.shape[0]]
    # Theta=[0.05,0.10,0.15,0.20,0.25]
    percent_insensitive=[]
    for j in range(0,55,5):
        s_ratio,negative_ratio,positive_ratio=show_iteration(theta=theta,eta=eta,alpha=alpha,
                                                       beta=beta,diffrate=diffrate,
                                                       srate=j/100,R1=R1,R2=R2,
                                                       gamma1=i1,gamma2=i2,commu=comm1,
                                                       main_att=main_att,high_influ=q0,
                                                       att=att0,result_commu=0)
        # percent_negative.append(negative_ratio[-1])
        # percent_positive.append(positive_ratio[-1])
        # percent_neutral.append(1-positive_ratio[-1]-negative_ratio[-1])
        percent_insensitive.append(1-s_ratio[-1])
    
    plt.plot(percent_insensitive,marker='o',c= (i1/2.1,i2/2.1,0.6),markersize=6.88,linewidth=2.66,label = r'$\gamma_1=$'+str(i1)+' '+r'$\gamma_2=$'+str(i2))
    # plt.yticks(np.arange(0,1.2,0.2),[0,20,40,60,80,100])
    plt.xticks(np.arange(0,10.1,2),[0,0.1,0.2,0.3,0.4,0.5])
    plt.ylabel('percentage(%)')
    plt.xlabel(r'Acceptance Rate $\beta$')
    plt.title(r'Sensitivity Analysis for $\gamma_1$, $\gamma_2$ and $\beta$')
    plt.legend()
plt.show()


plt.figure(figsize=(9.8,6.8)) 

for b in [0.1,0.2,0.3,0.4,0.5]:
    
    # percent_negative=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 0)/comm1.shape[0]]
    # percent_positive=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 1)/comm1.shape[0]]
    # percent_neutral=[np.count_nonzero(np.array(att)[np.where(label_pre==0)[0]] == 2)/comm1.shape[0]]
    # # Theta=[0.05,0.10,0.15,0.20,0.25]
    percent_insensitive=[]
    
    for j in range(0,55,5):
        print(r'$\alpha$={}, diffRate={}'.format(b,j))
        s_ratio,_,_=show_iteration(theta=theta,eta=eta,alpha=b,beta=beta,diffrate=j/100,
                                  srate=srate,R1=R1,R2=R2,gamma1=gamma1,gamma2=gamma2,
                                  commu=comm1,main_att=main_att,high_influ=q0,
                                  att=att0,result_commu=0)       
        # percent_negative.append(negative_ratio[-1])
        # percent_positive.append(positive_ratio[-1])
        # percent_neutral.append(1-positive_ratio[-1]-negative_ratio[-1])
        percent_insensitive.append(1-s_ratio[-1])
    plt.plot(percent_insensitive,marker='o',c= (0.2,(b-0.05)*2,0.5),markersize=6.88,linewidth=2.66,label = r'$\alpha=$'+str(b))
    # plt.yticks(np.arange(0,1.2,0.2),[0,20,40,60,80,100])
    plt.xticks(np.arange(0,10.1,2),[0,0.1,0.2,0.3,0.4,0.5])
    plt.ylabel('percentage(%)')
    plt.xlabel(r'Propagation Rate $diffRate$')
    plt.title(r'Sensitivity Analysis for $\alpha$ and $diffRate$')
plt.legend()
plt.show()





