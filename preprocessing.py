import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def load_data(data_path):
    data_csv = pd.read_csv(data_path, index_col=0)
    cells = data_csv.columns.values
    genes = data_csv.index.values
    data = data_csv.values.T
    return data, cells, genes

def impute_dropout(X, seed=None, drop_rate=0.1):
    X_zero = np.copy(X)
    i, j = np.nonzero(X_zero)
    if seed is not None:
        np.random.seed(seed)
    
    ix = np.random.choice(
        range(len(i)), 
        int(np.floor(drop_rate * len(i))), 
        replace=False
    )
    X_zero[i[ix], j[ix]] = 0.0
    return X_zero

def get_cluster_label(data, n_cluster, n_cell, n_gene, npc=20):
    if n_cluster>1:
        n = np.min((n_cell, n_gene))
        pca = PCA(n_components=n)
        imputer = SimpleImputer(strategy='mean')   # 也可选 'median' 或 'constant'
        data_imputed = imputer.fit_transform(data)  # data是你的原始输入矩阵
        pcs = pca.fit_transform(data_imputed)
        var = (pca.explained_variance_ratio_).cumsum()
        npc_raw = (np.where(var > 0.7))[0].min()
        if npc_raw > npc:
            npc_raw = npc
        pcs = pcs[:,:npc_raw]
        kmean = KMeans(n_clusters=n_cluster, random_state=1).fit(
            StandardScaler().fit_transform(pcs)
        )
        clustering_label = kmean.labels_
        
    return pcs, clustering_label

def take_norm(data, cellwise_norm=True, log1p=True):
    data_norm = data.copy()
    if cellwise_norm:
        libs = data.sum(axis=1)
        norm_factor = np.diag(np.median(libs) / libs)
        data_norm = np.dot(norm_factor, data_norm)
    if log1p:
        data_norm = np.log2(data_norm + 1.)
    return data_norm