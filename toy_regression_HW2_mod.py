import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


if __name__ == '__main__':

    X_START         = 0
    X_END           = 9

    n_per_group     = np.arange(100, 10000, 100)

    ideal_intercept = 5
    ideal_slope     = 2
    group_difference = -1

    NOISE_MEAN      = 0
    NOISE_STD_vec   = [1, 5, 10]

    #the ideal parameters
    beta_ideal      = [ideal_intercept, ideal_slope, group_difference]

    np.random.seed(1)

    #matrix of resulting group_difference estimate absolute errors
    AbsErr_mat    = np.zeros((len(n_per_group), len(NOISE_STD_vec)))

    #matrix of group_difference estiamte 95% CI widths
    CI_width_mat   = np.zeros((len(n_per_group), len(NOISE_STD_vec)))

    for i, n_group_i in enumerate(n_per_group):

        #observations of age
        x_observed      = np.linspace(X_START, X_END, n_group_i).reshape(n_group_i, 1)

        #ideal observations of brain size in full-term group
        y_group1_ideal  = ideal_slope*x_observed + ideal_intercept

        #create ideal observations for both groups
        y_group1        = y_group1_ideal
        y_group2        = y_group1_ideal + group_difference

        for j, noise_std_i in enumerate(NOISE_STD_vec):

            #generate zero-mean observation error (i.e. noise)
            #by sampling from a Gaussian with the distribution we want
            y_group1_noisy  = y_group1 + np.random.normal(NOISE_MEAN, NOISE_STD_vec[j], (n_group_i,1))
            y_group2_noisy  = y_group2 + np.random.normal(NOISE_MEAN, NOISE_STD_vec[j], (n_group_i,1))

            y_train_j       = np.r_[y_group1_noisy, y_group2_noisy]

            #np.r_ concatenate row-wise (vertically)
            y_groups        = np.r_[y_group1, y_group2]
            x_groups        = np.r_[x_observed, x_observed]

            #create group coding variable by stacking zeros and ones vertically
            x_coding        = np.r_[np.zeros(x_observed.size), np.ones(x_observed.size)]

            #form design matrix
            X_design_j      = np.c_[np.ones(x_groups.shape), x_groups, x_coding]

            #let's see if we can get the true group difference back
            #beta_est        = np.linalg.pinv(X_design_groups) @ y_groups

            model_j         = sm.OLS(y_train_j, X_design_j)
            results_j       = model_j.fit()

            beta_est_j      = results_j.params
            CI_mat          = results_j.conf_int()

            AbsErr_mat[i, j]    = np.abs(beta_est_j[2]-beta_ideal[2])
            CI_width_mat[i, j]  = CI_mat[2,1] - CI_mat[2,0]

    plt.subplot(1,2,1)
    color_codes             = ['.--k', '.--m', '.--c']
    for i,std_i in enumerate(NOISE_STD_vec):
        plt.plot(n_per_group, AbsErr_mat[:, i],  color_codes[i], label='error std='+str(std_i))
    plt.xlabel('Number of samples per group')
    plt.ylabel('Group diff |error|')
    plt.legend()

    plt.subplot(1,2,2)
    color_codes             = ['.--k', '.--m', '.--c']
    for i,std_i in enumerate(NOISE_STD_vec):
        plt.plot(n_per_group, CI_width_mat[:, i],  color_codes[i], label='error std='+str(std_i))
    plt.xlabel('Number of samples per group')
    plt.ylabel('Group diff CI width')
    plt.legend()

    plt.show()

    print('done')


