import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def sil_scores(X, max_n_clusters, init, n_init, max_iter, tol, algorithm):
    """
    Compute the average silhouette scores for a range of cluster numbers.
    """
    result = []
    range_n_clusters = range(1, max_n_clusters + 1)
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, 
                        init=init, 
                        n_init=n_init, 
                        max_iter=max_iter, 
                        tol=tol, 
                        algorithm=algorithm, 
                        random_state=42)

        cluster_labels = kmeans.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg:.3f}")
        
        result.append(silhouette_avg)
    
    return result

def plot_silhouette_scores(X, max_n_clusters, init, n_init, max_iter, tol, algorithm):
    """
    Calculate silhouette scores for each k number of clusters and plot it with markers.
    """
    silhouette_scores = []
    range_n_clusters = range(2, max_n_clusters + 1, 2)

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, 
                        init=init, 
                        n_init=n_init, 
                        max_iter=max_iter, 
                        tol=tol, 
                        algorithm=algorithm, 
                        random_state=42)

        preds = kmeans.fit_predict(X)
        score = silhouette_score(X, preds)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(range_n_clusters, silhouette_scores, marker='s', color='k', linestyle='-.')

    max_score_idx = silhouette_scores.index(max(silhouette_scores))
    optimal_k = range_n_clusters[max_score_idx]

    plt.axvline(x=optimal_k, color='g', linestyle=':', label=f'Optimal k = {optimal_k}')

    plt.title("The Silhouette method for optimal number of clusters")
    plt.xlabel("Number of clusters, k")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.xticks(range_n_clusters)
    # plt.grid(True)
    plt.show()

def silhouette_plotter(X, max_n_clusters, init, n_init, max_iter, tol, algorithm, tsne_X):
    """
    Plot silhouette analysis for different numbers of clusters using KMeans clustering.
    """
    all_scores = []
    range_n_clusters = range(2, max_n_clusters + 1, 2)
    
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        kmeans = KMeans(n_clusters=n_clusters, 
                        init=init, 
                        n_init=n_init, 
                        max_iter=max_iter, 
                        tol=tol, 
                        algorithm=algorithm, 
                        random_state=42)

        cluster_labels = kmeans.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        all_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg:.3f}")

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  

        ax1.set_title("Silhouette plot for various clusters")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  
        ax1.set_xticks(np.linspace(-0.1, 1, 6))

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(tsne_X[:, 0], tsne_X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ax2.set_title("Visualization of the clustered data")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}", fontweight='bold')

        plt.show()
    
    return all_scores

def plot_elbow_method(X, optimal_k, max_n_clusters, init, n_init, max_iter, tol, algorithm):
    """
    Plot the elbow method for determining the optimal number of clusters.
    """
    wcss = []
    range_n_clusters = range(1, max_n_clusters + 1)
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, 
                        init=init, 
                        n_init=n_init, 
                        max_iter=max_iter, 
                        tol=tol, 
                        algorithm=algorithm, 
                        random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range_n_clusters, wcss, marker='s', color='k', linestyle='-.')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')

    plt.axvline(x=optimal_k, color='g', linestyle=':', label=f'Optimal k = {optimal_k}')

    plt.xticks(range(1, max_n_clusters + 1))
    # plt.grid(True)
    plt.legend()
    plt.show()