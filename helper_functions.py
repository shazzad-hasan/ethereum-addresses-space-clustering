import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def data_pipeline(df):
    data = df.iloc[:, 1:-1]
    log = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    scaler = StandardScaler()
    pca = PCA(n_components=data.shape[1])
    pipeline = Pipeline([
        ("log", log),
        ("scaler", scaler),
        ("pca", pca)
    ])

    processed_data = pipeline.fit_transform(data)
    return pipeline, processed_data


def make_cluster(data, n_clusters, n_init, max_iter):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    return kmeans.fit(data)

def calc_tsne(data, n_components, perplexity, n_iter, learning_rate):
    tsne_model = TSNE(n_components=n_components, 
                perplexity=perplexity,
                n_iter=n_iter,
                learning_rate=learning_rate
            )
    tsne_transformed_data = tsne_model.fit_transform(data)
    return tsne_transformed_data

def assign_cluster_to_data(data, clusters):
    df = data.copy() 
    df["cluster"] = -1
    for i, row in df.iterrows():
        df.iat[i, -1] = clusters[i]
    return df

def find_category_of_cluster(clusters, df_with_cluster, category):
    cluster_type = 0
    num_cluster_type = 0
    label_density = 0
    print(category)

    for cluster in np.unique(clusters.labels_):
        size_of_cluster = np.sum(clusters.labels_==cluster)
        d = df_with_cluster[df_with_cluster["cluster"]==cluster]
        num_addresses = np.sum(d['Entity']==category)
        density = (num_addresses / size_of_cluster) * 100
        if num_addresses > num_cluster_type:
            label_density = density
            num_cl_type = num_addresses
            cluster_type = cluster
        print(f"Cluster num: {cluster}, num of addresses: {num_addresses}, cluster size: {size_of_cluster}, lebel density: {density}")
    
    return cluster_type





