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

            #add noise to the samples for each group
            #by sampling from a Gaussian with the distribution we want

            #concatenate them

            #form design matrix

            #fit the OLS model

            #grab the group difference parameter and corresponding CI width
            #save them to AbsErr_mat and CI_width_mat


            pass


    #once you're out of the nested loop, create the figures

    #use    plt.subplot(1,2,1)
    #then plt.subtplot(1,2,2)
    #don't forget the xlabel, ylabel and legend for both

    #add plt.show() at the end to render the figure


