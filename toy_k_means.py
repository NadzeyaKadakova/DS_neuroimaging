import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def calc_inertia(X, labels, cluster_centers):

    K               = len(np.unique(labels))

    inertias            = np.zeros(K)

    for k_i in range(K):

        X_k         = X[labels==k_i, :]

        # number of points in this cluster
        n_k = np.sum(labels == k_i)

        #loop over all points in the cluster
        for j in range(n_k):
            x_j     = X_k[j, :]

            diff_j  = x_j - cluster_centers[k_i, :]

            inertias[k_i] += np.sum(diff_j**2)

    inertia     = sum(inertias)

    return inertia

#**** CREATE A FUNCTION THAT CALCULATES W (eq. 14.28 in ESLII)
#def calc_W(...)


if __name__ == '__main__':

    #create unit vectors v1, v2 in  [1 1], [-1 1] directions
    v1              = np.array([[ 0],  [1]])
    v1              = v1/np.linalg.norm(v1)

    v2              = np.array([[-1],  [1]])
    v2              = v2/np.linalg.norm(v2)

    v3              = np.array([[1],  [0]])
    v3              = v3/np.linalg.norm(v3)

    n_sample_per_cluster       = 50

    CLUSTER_SEPARATION = 2

    np.random.seed(1)

    #generate 2 clusters
    MEAN1           = [1, CLUSTER_SEPARATION/2]
    COV1            = 0.4 * (v1 @ v1.T) + 0.6 * (v2 @ v2.T)
    cluster1        = np.random.multivariate_normal(MEAN1, COV1, n_sample_per_cluster)

    MEAN2           = [1, -(CLUSTER_SEPARATION/2)]
    COV2            = 0.1*(v1 @ v1.T) + 0.9 * (v3 @ v3.T)
    cluster2        = np.random.multivariate_normal(MEAN2, COV2, n_sample_per_cluster)


    plt.figure(1)
    plt.plot(cluster1[:,0], cluster1[:,1], '.b')
    plt.plot(cluster2[:,0], cluster2[:,1], '.b')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Unclustered')
    plt.show()

    #plot raw data with cluster labels
    plt.figure(2)
    plt.plot(cluster1[:,0], cluster1[:,1], '.r', label='cluster 1')
    plt.plot(cluster2[:,0], cluster2[:,1], '.g', label='cluster 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('True clusters')
    plt.legend()
    plt.show()


    #******

    X               = np.r_[cluster1, cluster2]

    #***** fit some K-means models
    K_vals          = [1, 2, 3, 4]
    colors          = ['r', 'b', 'g', 'm', 'c']

    plt.figure(3)

    #local version of inertia
    inertia_sklearn = np.zeros(len(K_vals))
    inertia_local   = np.zeros(len(K_vals))

    #within-cluster scatter W, defined by equation 14.28 in ESLII
    W               = np.zeros(len(K_vals))

    for i, K_i in enumerate(K_vals):

        kmeans_model        = KMeans(n_clusters=K_i)
        kmeans_model.fit(X)

        labels              = kmeans_model.labels_
        cluster_centers_    = kmeans_model.cluster_centers_


        #Scikit-learn's inertia metric: Sum of squared distances of samples to their closest cluster center.
        inertia_sklearn[i]  = kmeans_model.inertia_

        #local version of above
        inertia_local[i]    = calc_inertia(X, labels, cluster_centers_)

        #W metric: sum of squared distances between all points within each cluster, summed across all clusters
        #*** CHANGE THIS TO CALL calc_W(...)
        W[i]                = 0

        unique_labels       = np.unique(labels)

        plt.subplot(2, 2, i+1)

        #plot raw data with cluster labels
        for j, label_j in enumerate(unique_labels):
            plt.plot(X[labels==label_j, 0], X[labels==label_j, 1], '.'+ colors[j])#, label='cluster ' + str(j+1))

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K = ' + str(K_i))
        #plt.legend()

    plt.show()


    plt.figure(4)
    plt.subplot(1,2,1)
    plt.plot(K_vals, inertia_sklearn, '.-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia, sklearn version')

    plt.subplot(1,2,2)
    plt.plot(K_vals, inertia_local, '.-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia, local version')

    plt.show()

    plt.figure(5)
    plt.subplot(1,2,1)
    plt.plot(K_vals, W, '.-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Within-cluster dissimilarty (W)')

    plt.subplot(1,2,2)
    plt.plot(K_vals, inertia_sklearn, '.-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia, sklearn version')
    plt.show()

    print('done.')