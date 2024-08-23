import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(clusters, tsne_results):
    """
    Plot the clusters found by t-SNE dimensionality reduction.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        mask = clusters == cluster
        ax.scatter(tsne_results[mask, 0], tsne_results[mask, 1], s=20, alpha=0.8, label=f"Cluster {cluster}")

    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    for lh in legend.legendHandles:
        lh.set_alpha(1)

    ax.set_title("Clusters")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    plt.show()

def plot_tsne_with_labels(tsne_results, df, dflabel, categs, colors):
    """
    Plot t-SNE results, highlighting data points with specific labels.
    """
    labeled_addresses = dflabel["address"].values
    labelmask = df["address"].isin(labeled_addresses).values

    address_to_category = dict(zip(dflabel["address"], dflabel["Entity"]))

    labeled_tsne = tsne_results[labelmask]
    unlabeled_tsne = tsne_results[~labelmask]

    plt.figure(figsize=(10, 8))
    plt.scatter(unlabeled_tsne[:, 0], unlabeled_tsne[:, 1], s=20, c="gray", alpha=0.3)

    for category, color in zip(categs, colors):
        catmask = np.array([address_to_category.get(addr) == category for addr in df["address"]])
        plt.scatter(tsne_results[catmask & labelmask][:, 0], tsne_results[catmask & labelmask][:, 1],
                    s=20, c=color, alpha=1, label=category)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title("Labeled Data Points")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()


def plot_tsne_with_labeled_clusters(tsne_results, cl, clusters, categs, colors):
    """
    Plot t-SNE results with clusters highlighted. Labeled clusters are displayed with specific colors.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for cluster_label in np.unique(cl.labels_):
        mask = cl.labels_ == cluster_label
        
        if cluster_label in clusters:
            idx = clusters.index(cluster_label)
            label = categs[idx]
            color = colors[idx]
            ax.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                       s=100, c=color, alpha=0.4, label=f'Cluster {cluster_label} - "{label}"')
        else:
            ax.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                       s=20, c='gray', alpha=0.3)
    
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    for lh in legend.legendHandles:
        lh.set_alpha(1)

    plt.title('Labeled Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

def plot_all(tsne_results, clusters, dataset, dflabel, cl_types, categories, colors):
    plot_tsne(clusters.labels_, tsne_results)
    plot_tsne_with_labels(tsne_results, dataset, dflabel, categories, colors)
    plot_tsne_with_labeled_clusters(tsne_results, clusters, cl_types, categories, colors)



