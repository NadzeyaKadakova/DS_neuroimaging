import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import normalize

#Adapted from: https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py

if __name__ == '__main__':

    # #############################################################################
    # Generate sample data
    np.random.seed(0)
    n_samples       = 2000
    time            = np.linspace(0, 8, n_samples)

    # Signal 1 : sinusoidal signal
    s1              = np.sin(2 * time)

    # Signal 3: saw tooth signal
    s2              = signal.sawtooth(2 * np.pi * time)

    plt.figure(1)
    plt.plot(time, s1, '-b')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.show()

    plt.figure(2)
    plt.plot(time, s2, '-r')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.tight_layout()
    plt.show()

    #stack them together and add some noise
    S               = np.c_[s1, s2]
    S               += 0.2 * np.random.normal(size=S.shape)  # Add noise

    # Standardize data - compute std. dev. for each signal (row-wise) and rescale both
    S               /= S.std(axis=0) #np.tile(S.std(axis=1).reshape(2,1), (1, n_samples))

    # Mix the three independent signals together
    # Mixing matrix
    A               =  np.array([[2/3, 1/5], [1/3, 4/5]])

    # Generate observations (matrix multiplication of signals S by transposed mixing matrix A.T)
    X               = S @ A

    plt.figure(3)
    plt.subplot(1,2,1)
    plt.plot(time, X[:,1], '-c')
    plt.xlabel('time')
    plt.ylabel('signal')

    plt.subplot(1,2,2)
    plt.plot(time, X[:,1], '-m')
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.show()

    # Compute ICA
    ica             = FastICA(n_components=2)
    # Reconstruct signals
    S_est           = ica.fit_transform(X) #ica.fit_transform(X.T)#
    # Get estimated mixing matrix
    A_est               = ica.mixing_

    # We can `prove` that the ICA model applies by reverting the unmixing.
    assert np.allclose(X, np.dot(S_est, A_est.T) + ica.mean_)

    # For comparison, compute PCA
    pca             = PCA(n_components=2)
    # Reconstruct signals based on orthogonal components
    H               = pca.fit_transform(X)

    # #############################################################################
    # Plot results

    plt.figure()

    models          = [X, S, S_est, H]
    names           = ['Observations (mixed signal)',
                        'True Sources',
                        'ICA recovered signals',
                     '  PCA recovered signals']
    colors          = ['red', 'steelblue', 'orange']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.tight_layout()
    plt.show()


    print('True mixing matrix A: ')
    print(np.around(A, 2))
    print('Estimated mixing matrix A: ')
    print(np.around(A_est, 2))

    #normalize the columns of estimated A by dividing by column sum
    A_est_normed        = A_est / np.sum(A_est, axis=0)

    print('Estimated mixing matrix A, normed')
    print(np.around(A_est_normed, 2))
