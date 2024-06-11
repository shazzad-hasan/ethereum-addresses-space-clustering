import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(clusters, tsne_results):
    """Plot the clusters found by reducing dimensions with calc_tsne."""
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(tsne_results[mask][:, 0],tsne_results[mask][:, 1],s=20,alpha=0.5,label=cluster)

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for lh in legend.legendHandles:
        lh.set_alpha(1)

    plt.title("Clusters: T-SNE", fontsize=20)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()


def plot_tsne_with_labels(tsne_results, df, dflabel, categs, colors):
    """Plot the clusters but only highlighting the data points with labels."""

    labeled_addresses = dflabel["ethereum_address"].values
    labelmask = np.array([addr in labeled_addresses for addr in df["ethereum_address"]])

    # Helper function for category mask
    def cat(addr, labeled_addresses, dflabel):
        if addr not in labeled_addresses:
            return False
        else:
            idx = int(np.where(labeled_addresses == addr)[0][0])
            return dflabel.iloc[idx, 1]

    subset, not_subset = tsne_results[labelmask], tsne_results[~labelmask]
    fig = plt.figure(figsize=(10, 8))
    # not labelled points
    plt.scatter(not_subset[:, 0], not_subset[:, 1], s=20, c="gray", alpha=0.3)

    # categories
    cats = np.array(
        [cat(addr, labeled_addresses, dflabel) for addr in df["ethereum_address"]]
    )  
    for c in list(dflabel["Entity"].unique()):
        mask = dflabel["Entity"] == c
        # category mask
        catmask = cats == c
        if c in categs:
            idx = categs.index(c)
            color = colors[idx]
            plt.scatter(
                tsne_results[(labelmask & catmask)][:, 0],tsne_results[(labelmask & catmask)][:, 1],s=20,c=color,alpha=1,label=c)

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for lh in legend.legendHandles:
        lh.set_alpha(1)

    plt.title("Clusters - Labeled Data Points: T-SNE", fontsize=20)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

def plot_tsne_with_labeled_clusters(tsne_results, cl, clusters, categs, colors):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    for c in np.unique(cl.labels_):
        mask = cl.labels_==c

        if c in clusters:
            idx = clusters.index(c)
            lbl = categs[idx]
            color = colors[idx]
            plt.scatter(tsne_results[mask][:,0], tsne_results[mask][:,1], s=100,c=color,alpha=.4,label=('Cluster {} - "{}" '.format(c,lbl) ))
        else:
             plt.scatter(tsne_results[mask][:,0], tsne_results[mask][:,1], c='gray',s=20, alpha=.3)

    leg = plt.legend(bbox_to_anchor=(1, 1))
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.title('T-SNE', fontsize=20)
    plt.xlabel('first principal component')
    plt.ylabel('second principal component')
    plt.show()

def plot_all(tsne_results, clusters, dataset, dflabel, cl_types, categories, colors):
    plot_tsne(clusters.labels_, tsne_results)
    plot_tsne_with_labels(tsne_results, dataset, dflabel, categories, colors)
    plot_tsne_with_labeled_clusters(tsne_results, clusters, cl_types, categories, colors)



