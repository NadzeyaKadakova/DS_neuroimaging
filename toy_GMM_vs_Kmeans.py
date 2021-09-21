import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#FROM: https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
from matplotlib.patches import Ellipse
def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs    = np.linalg.eigh(cov)
    order               = eigvals.argsort()[::-1]
    eigvals, eigvecs    = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy              = eigvecs[:,0][0], eigvecs[:,0][1]
    theta               = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height       = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height, angle=np.degrees(theta), **kwargs)

if __name__ == '__main__':


    #samples per group
    n_group = 100

    #means and covariances for both clusters
    MEAN1   = [3, 5]
    MEAN2   = [5, 8]
    COV1    = [[1.7, 0.5], [0.5, 1.7]]
    COV2    = [[0.4, 0],   [0, 0.4]]

    np.random.seed(5)

    #generate the data
    cluster1 = np.random.multivariate_normal(MEAN1, COV1, n_group)
    cluster2 = np.random.multivariate_normal(MEAN2, COV2, n_group)

    # ***************

    X       = np.r_[cluster1, cluster2]
    K       = 2

    # *********** K-means **************
    kmeans_model    = KMeans(n_clusters=K)
    kmeans_model.fit(X)

    labels_kmeans   = kmeans_model.labels_
    cluster_kmeans_1 = X[labels_kmeans==0,:]
    cluster_kmeans_2 = X[labels_kmeans==1,:]

    # *********** GMM **************
    gmm_model       = GaussianMixture(n_components=K, random_state=1)
    gmm_model       = gmm_model.fit(X)

    labels_gmm      = gmm_model.predict(X)
    cluster_gmm_1     = X[labels_gmm==0,:]
    cluster_gmm_2     = X[labels_gmm==1,:]


    #********* plot *********
    plt.figure(1)

    f       = plt.figure(figsize=(10, 3))
    ax      = f.add_subplot(131)
    ax.plot(cluster1[:,0],     cluster1[:,1],  '.r')
    ax.plot(cluster2[:, 0],    cluster2[:, 1], '.g')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('True clusters')

    ax2     = f.add_subplot(132)
    ax2.plot(cluster_kmeans_1[:,0],     cluster_kmeans_1[:,1],  '.g')
    ax2.plot(cluster_kmeans_2[:, 0],    cluster_kmeans_2[:, 1], '.r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('K means clustering')

    ax3     = f.add_subplot(133)
    ax3.plot(cluster_gmm_1[:,0],     cluster_gmm_1[:,1],  '.r')
    ax3.plot(cluster_gmm_2[:, 0],    cluster_gmm_2[:, 1], '.g')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('GMM clustering')

    plt.show()


    # *************** compare cluster 95% confidence ellipses between true and GMM distributions
    plt.figure(2)

    f           = plt.figure(figsize=(7, 3))


    #number of std. dev.'s enclosed by ellipses
    NUM_STDs_ELLIPSE = 2

    ax          = f.add_subplot(121)

    ax.scatter(cluster1[:,0], cluster1[:,1], color='r', s=4)
    ellipse1    = get_cov_ellipse(COV1, MEAN1, NUM_STDs_ELLIPSE, fc='r', alpha=0.1)
    ax.add_patch(ellipse1)

    ax.scatter(cluster2[:,0], cluster2[:,1], color='g', s=4)
    ellipse2    = get_cov_ellipse(COV2, MEAN2, NUM_STDs_ELLIPSE, fc='g', alpha=0.1)
    ax.add_patch(ellipse2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('True clusters')

    ax2     = f.add_subplot(122)

    ax2.scatter(cluster_gmm_1[:, 0], cluster_gmm_1[:, 1], color='r', s=4)
    ellipse1 = get_cov_ellipse(COV1, MEAN1, NUM_STDs_ELLIPSE, fc='r', alpha=0.1)
    ax2.add_patch(ellipse1)

    ax2.scatter(cluster_gmm_2[:, 0], cluster_gmm_2[:, 1], color='g', s=4)
    ellipse2 = get_cov_ellipse(COV2, MEAN2, NUM_STDs_ELLIPSE, fc='g', alpha=0.1)
    ax2.add_patch(ellipse2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('GMM clustering')

    plt.show()



    print('done.')