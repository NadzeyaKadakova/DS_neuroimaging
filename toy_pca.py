import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(1)


    n_samples       = 200

    # class mean vectors
    mean_vec       = [0, 0]

    # create unit vectors v1 in [1 1] direction
    v1              = np.array([[ 1],  [1]])
    v1              = v1 /np.linalg.norm(v1)

    # create covariance matrix with information restricted to v1 direction (i.e. rank 1)
    covMat1         = v1 @ v1.T

    # generate samples from 2-D Gaussian with specified mean and covariance
    X1              = np.random.multivariate_normal(mean_vec, covMat1, n_samples)

    # create unit vectors v2 in [-1 1] direction
    v2              = np.array([[-1],  [1]])
    v2              = v2 /np.linalg.norm(v2)

    # create covariance matrix with information restricted to v2 direction (i.e. rank 1)
    covMat2         = v2 @ v2.T

    # generate samples from 2-D Gaussian with specified mean and covariance
    X2              = np.random.multivariate_normal(mean_vec, covMat2, n_samples)

    plt.figure(1)
    plt.plot(X1[:,0], X1[:,1],  '.r', label='cov1 samples')
    plt.plot(X2[:,0], X2[:,1], '.b',  label='cov2 samples')
    plt.xlabel('x1, feature dimension 1')
    plt.ylabel('x2, feature dimension 2')
    plt.legend()
    plt.show()


    # create covariance matrix with 95% of variance in v1 direction, 5% in v2 direction
    covMat_both     = 0.95 * (v1 @ v1.T) + 0.05 * (v2 @ v2.T)

    # generate samples from 2-D Gaussian with specified mean and covariance
    X_both          = np.random.multivariate_normal(mean_vec, covMat_both, n_samples)

    plt.figure(2)
    plt.plot(X_both[:,0], X_both[:,1],  '.m', label='cov_both samples')
    plt.xlabel('x1, feature dimension 1')
    plt.ylabel('x2, feature dimension 2')
    plt.legend()
    plt.show()


    #**** now let's get the 'natural axes' (v1 and v2) empirically via PCA

    #calculate empirical covaraince matrix
    mean_S          = np.mean(X_both, axis=0)
    X_centered      = X_both - np.tile(mean_S, (n_samples, 1))
    S               = 1/(n_samples-1) * ((X_centered.T)@X_centered)

    #same as above
    S_v2            = np.cov(X_both.T)
    print('sum(|S - S_v2|): %f' % ( np.sum(np.abs(S - S_v2))))

    #calculate principal components (eigenvectors of S) via numpy
    #the corresponding eigenvalues are the amount of variance explained by each eigenvector
    e_vals, e_vecs  = np.linalg.eig(S)

    #divide each eigenvalue by the sum to get fraction of explained variance for each
    e_vals          = e_vals/np.sum(e_vals)

    print('Eigenvector 1 (i.e. PC1): %s, explained variance: %.1f%%. \nActual axis v1: %s, actual variance: 95%%' % (str(e_vecs[:,0]), e_vals[0]*100, str(v1.T)))
    print('Eigenvector 2 (i.e. PC2): %s, explained variance: %.1f%%. \nActual axis v1: %s, actual variance:  5%%' % (str(e_vecs[:,1]), e_vals[1]*100, str(v2.T)))

    #let's say we wanted to create T that explains at least 90% of our data
    T_mat               = e_vecs[0]

    #create projected features and visualize them
    X_both_proj     = X_both @ T_mat

    plt.figure(3)
    plt.plot(X_both_proj, np.zeros(X_both_proj.shape), '.m', label='projected samples')
    plt.xlabel('u1, feature dimension 1')
    plt.legend()
    plt.show()

    #***** scikit-learn version
    #it's doing the same thing, but in practice it may be more computationally efficient
    from sklearn.decomposition import PCA
    pca             = PCA()
    pca.fit(X_both)
    print('*** Scikit-learn version *** ')
    print(pca.components_)
    print(pca.explained_variance_ratio_)

    print('done.')