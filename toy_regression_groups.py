import numpy as np
import matplotlib.pyplot as plt

#create and plot a prediction line for a particular group
def plot_group_line(x_pred, group_code, beta, plot_format, plot_label):

    #create the design matrix and then create the predicted line
    X_pred      = np.c_[np.ones(x_pred.shape), x_pred, np.tile(group_code, x_pred.shape)]
    y_pred      = X_pred @ beta

    plt.plot(x_pred, y_pred,  plot_format, label=plot_label)

#create the scatter plots of actual observations for both groups and then create prediction lines
def plot_groups(X_train, y_train, list_group_indices, beta_ideal, beta_estimated):

    #make sure there are only 2 groups
    assert(len(list_group_indices)==2)

    #get the rows that go with each group
    i_group1    = list_group_indices[0]
    i_group2    = list_group_indices[1]

    x_group1    = X_train[i_group1, 1]
    y_group1    = y_train[i_group1]

    x_group2    = X_train[i_group2, 1]
    y_group2    = y_train[i_group2]

    #plot the points, with some transparency (alpha = 0.5)
    plt.plot(x_observed, y_group1, 'b.', alpha=0.5, markersize=12, label='full term')
    plt.plot(x_observed, y_group2, 'r.', alpha=0.5, markersize=12, label='pre term')

    num_samples_line = 30
    x_line      = np.linspace(np.min(x_group1), np.max(x_group1), num_samples_line)

    #we want to create a line for each group based on ideal and estimate params
    plot_group_line(x_line, 0, beta_ideal,      'b--', 'ideal, full')
    plot_group_line(x_line, 1, beta_ideal,      'r--', 'ideal, pre')
    plot_group_line(x_line, 0, beta_estimated,  'b-', 'est, full')
    plot_group_line(x_line, 1, beta_estimated,  'r-', 'est, pre')

    plt.xlabel('age',       fontsize=15)
    plt.ylabel('brain size',fontsize=15)
    plt.ylim([0, 30])
    plt.legend()


if __name__ == '__main__':

    X_START         = 0
    X_END           = 9

    n_per_group     = 10    #10, 500, 500000
    ideal_intercept = 5
    ideal_slope     = 2
    group_difference = -1

    NOISE_MEAN      = 0
    NOISE_STD_DEV   = 2

    #observations of age
    x_observed      = np.linspace(X_START, X_END, n_per_group).reshape(n_per_group, 1)

    #constant column
    x_const         = np.ones(n_per_group).reshape((n_per_group,1))

    #observations of brain size
    y_observed      = ideal_slope*x_observed + ideal_intercept

    #create ideal observations for both groups
    y_group1        = y_observed
    y_group2        = y_observed + group_difference

    #np.r_ concatenate row-wise (vertically)
    y_groups        = np.r_[y_group1, y_group2]
    x_groups        = np.r_[x_observed, x_observed]

    #create group coding variable by stacking zeros and ones vertically
    x_coding        = np.r_[np.zeros(x_observed.size), np.ones(x_observed.size)]

    X_design_groups  = np.c_[np.ones(x_groups.shape), x_groups, x_coding]

    #let's see if we can get the true group difference back
    beta_est        = np.linalg.pinv(X_design_groups) @ y_groups

    beta_ideal      = np.reshape([ideal_intercept, ideal_slope, group_difference], (3, 1))

    #get the row indices that go with each group
    list_group_indices = [np.where(x_coding==0)[0], np.where(x_coding==1)[0]]

    plt.figure(1)
    plot_groups(X_design_groups, y_groups, list_group_indices, beta_ideal, beta_est)
    plt.show()

    #************ add some noise and let's see if we can still detect the group diff
    n_samples       = x_groups.shape[0]

    #set the randomizer seed so we get the same random values each time we run
    np.random.seed(1)

    #generate zero-mean observation error (i.e. noise)
    #by sampling from a Gaussian with the distribution we want
    y_group1_noisy  = y_group1 + np.random.normal(NOISE_MEAN, NOISE_STD_DEV, (n_per_group,1))
    y_group2_noisy  = y_group2 + np.random.normal(NOISE_MEAN, NOISE_STD_DEV, (n_per_group,1))
    y_groups_noisy  = np.r_[y_group1_noisy, y_group2_noisy]

    #re-estimate params - notice: same design matrix
    beta_est_noisy  = np.linalg.pinv(X_design_groups) @ y_groups_noisy

    plt.figure(2)
    plot_groups(X_design_groups, y_groups_noisy, list_group_indices, beta_ideal, beta_est_noisy)
    plt.show()

    #print estimate params (transpose first)
    print('beta_noisy: ' + str(beta_est_noisy.T))

    print('done')


