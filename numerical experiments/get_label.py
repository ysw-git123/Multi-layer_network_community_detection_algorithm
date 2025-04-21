from sklearn.cluster import KMeans

def get_label_U(U0,nclass=3):
    # 训练聚类模型
    kmeans = KMeans(n_clusters=nclass, random_state=2018)
    kmeans.fit(U0)
    pre_label = kmeans.predict(U0)
    #np.unique(pre_label)
    #Counter(pre_label)
    #Counter(label[0])
    return pre_label

