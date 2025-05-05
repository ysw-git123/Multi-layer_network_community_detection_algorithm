import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jieba
import wordcloud

import torch
import networkx as nx

def real_graph():  
    data=pd.read_excel("评论.xlsx")
    data['Unnamed: 7'][data["情感"]==0]='负面'
    data['Unnamed: 7'][data["情感"]==1]='正面'
    data['Unnamed: 7'][data["情感"]==2]='中立'
    data.rename(columns={"Unnamed: 7": "attitude","Unnamed: 5":"二次回复"},inplace=True)
    
    
    data['地区'] = data['ip地址'].apply(lambda x : list(set(jieba.cut(x))))
    
    jieba.load_userdict('newwords.txt')
    
    data['word'] = data['微博内容文本'].apply(lambda x : list(jieba.cut(x)))
    # 读取停用词数据
    stopwords = pd.read_csv('cn_stopwords.txt', encoding='utf8', names=['stopword'], index_col=False)
    #转化词列表 
    stop_list = stopwords['stopword'].tolist() 
    stop_list.append(' ')
    stop_list.append('…')
    data['ccut']=data['微博内容文本'].apply(lambda x:[i for i in jieba.cut(x) if i not in stop_list])
    
    G = nx.Graph()
    
    content=data[['用户名','ccut']]
    content_user=content['用户名'].tolist()
    cite=data[['用户名','回复对象','二次回复']]
    
    for i in range(len(cite)):
        edge=tuple(cite[['用户名','回复对象']].iloc[i,:].values)
        G.add_edge(*edge)       
        if cite['二次回复'][i] in content_user:
            edge=tuple(cite[['用户名','二次回复']].iloc[i,:].values)
            G.add_edge(*edge)       
    nx.draw(G,node_size=1.7)
    plt.show()
    adj1=nx.to_numpy_array(G) 
    
    ww=[]
    for i1 in range(3):
        #print(i1)
        tempt = []
        for i2 in data[data['情感']==i1]['ccut']:
            tempt.extend(i2)
            wwl = " ".join(tempt) 
        ww.append(wwl)
            
    from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
    # 定义函数
    def TF_IDF(corpus):
        vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵
        transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
        x = vectorizer.fit_transform(corpus)
        tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        word=vectorizer.get_feature_names_out()#获取词袋模型中的所有词语
        word_location = vectorizer.vocabulary_  # 词的位置
        weight=tfidf.toarray()#tf-idf权重矩阵
        return weight,word_location,x.toarray()
    
    weight,word_location,tf = TF_IDF(ww)
    
    
    word=[]
    location=[]
    for i in word_location.keys():
        word.append(i)
        location.append(word_location.get(i))
    #qqq=np.array(location)
    item=np.argsort(np.array(location))
    #qqq[item]
    word_array=np.array(word)
    word_array=word_array[item]
    
    ccuts=[]
    node=G.nodes
    for kk in node:
        ctempt=[]
        for i in range(len(data)):
            if data['用户名'][i]==kk:
                ctempt=ctempt+data['ccut'][i]
        ccuts.append(ctempt)
    
    #content=pd.DataFrame(ccuts)
        
    features=[]
    tf_idf=[]
    for e,j in enumerate(ccuts):
        #遍历数据集中每一个分词向量
        feature=np.zeros(len(word))
        tfidf=np.zeros(len(word))
        for k in j:#遍历每个用户词向量中的每一个词
            if k in word:
                num=np.where(word_array==k)[0]
                feature[num]=1
                tfidf[num]=weight[data.iloc[e,:]['情感'],num]
        features.append(feature)
        tf_idf.append(tfidf)
        
    np.random.seed(2023)
    tf_idf=np.array(tf_idf)
    sim2=np.zeros((len(node),len(node)))
    sum_feature=np.sum(tf_idf,axis=1)
    for i in range(len(node)):
        for j in range(i+1,len(node)):
            if (sum_feature[i]!=0)&(sum_feature[j]!=0):
                sim2[i,j]=np.dot(tf_idf[i,:],tf_idf[j,:])/np.sqrt(sum_feature[i]*sum_feature[j])
            else:
                sim2[i,j]=0
            #print(adj2[i,j])
    sim2=sim2+sim2.T
    
    adj2=np.zeros((len(node),len(node)))
    for i1 in range(len(node)):
        for i2 in range(i1+1,len(node)):
            #print(i1,i2)
            adj2[i1,i2]=np.random.binomial(1, sim2[i1,i2], size=None) 
    adj2=adj2+adj2.T#
    
    
    node=np.array(G.nodes)
    edgelist=[]
    G2=nx.Graph()
    for i in range(len(node)):
        G2.add_node(node[i])
        for j in range(i+1,len(node)):
            if adj2[i,j]==1:
                edge=tuple((node[i],node[j]))
                edgelist.append(edge)
                # G2.add_edge(*edge)
    G2.add_edges_from(edgelist)

    #pos = nx.spring_layout(G2,iterations=1, seed=2023)
    nx.draw(G2,node_size=6)          
    plt.show()
    a=np.array([adj1,adj2])
    A=torch.from_numpy(a).type(torch.float32)
    feature=torch.from_numpy(tf_idf).type(torch.float32)
    
    attitude=[]
    for j in range(len(node)):
        att=data['情感'][np.where(data['用户名']==node[j])[0]]
        attitude.append(max(att))
    return A,feature,[G,G2],attitude


























# np.where(data['用户名']==node[0])[0]
# len(np.where(data['用户名']==node[0]))
# data['情感'][np.where(data['用户名']==node[0])]

# attitude=[]
# #dl=[]
# for j in range(len(node)):
#     att=data['情感'][np.where(data['用户名']==node[j])[0]]
#     attitude.append(max(att))


    
#     if len(np.where(data['用户名']==node[j])[0])==1:
#         #attitude.append(data['情感'][np.where(data['用户名']==node[j])])
#         att=data['情感'][np.where(data['用户名']==node[j])[0]]
        
#         print(att)
#         print(max(att))
#     elif len(np.where(data['用户名']==node[j])[0])>1:
#         print(node[j])
#         att=data['情感'][np.where(data['用户名']==node[j])[0]]
#         #print(data['情感'][np.where(data['用户名']==node[j])[0]])
#         #print(att)
#         print(max(att))
#         attitude.append(max(att))

# max(att)
# c=[]    
# for cc in data['用户名'].unique:
#     print(cc)

# type(data['情感'][np.where(data['用户名']==node[j])[0]])
# from collections import Counter
# Counter(data['用户名'])
# len(Counter(data['用户名']))


# g0=np.array(G[0].nodes)
# g1=np.array(G2.nodes)
# sss=0
# for i in range(len(g0)):
#     if g0[i]!=g1[i]:
#         print(g0[i],g1[i])
#         sss+=1

# airports = ['B','C','A']
# edgelst  = [['C','B'],['A','B'],['A','C']]

# G = nx.Graph()

# G.add_nodes_from(airports)
# G.add_edges_from(edgelst)
# G.nodes
