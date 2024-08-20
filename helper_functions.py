import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def data_pipeline(df):
    data = df.iloc[:, 1:-1]
    pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=data.shape[1]))
    ])
    processed_data = pipeline.fit_transform(data)
    return pipeline, processed_data


def make_cluster(data, n_clusters, n_init, max_iter):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=42)
    return kmeans.fit(data)

def calculate_tsne(data, n_components, perplexity, n_iter, learning_rate):
    tsne_model = TSNE(n_components=n_components,perplexity=perplexity,n_iter=n_iter,learning_rate=learning_rate, random_state=42)
    tsne_transformed_data = tsne_model.fit_transform(data)
    return tsne_transformed_data

def assign_cluster_to_data(data, clusters):
    df = data.copy() 
    df["cluster"] = clusters
    return df

def find_category_of_cluster(clusters, df_with_cluster, category):
    best_cluster = None
    max_category_count = 0
    highest_density = 0
    
    print(category)

    unique_clusters = np.unique(clusters.labels_)
    
    for cluster in unique_clusters:
        cluster_data = df_with_cluster[df_with_cluster["cluster"] == cluster]
        cluster_size = len(cluster_data)
        category_count = np.sum(cluster_data['Entity'] == category)
        density = (category_count / cluster_size) * 100
        
        if category_count > max_category_count:
            highest_density = density
            max_category_count = category_count
            best_cluster = cluster
        
        print(f"Cluster num: {cluster}, num of addresses: {category_count}, cluster size: {cluster_size}, label density: {density:.4f}")
    
    return best_cluster





