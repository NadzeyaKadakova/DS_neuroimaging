import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

if __name__ == '__main__':

    #parameters for class 0 (e.g. pre-term)
    n_class0        = 30
    CLASS0_MEAN     = 14
    CLASS0_STD      = 2

    # parameters for class 1 (e.g. full-term)
    n_class1        = 30
    CLASS1_MEAN     = 15
    CLASS1_STD      = 2

    np.random.seed(1)

    #draw some samples for each class based on each's normal distribution parameters
    feature_class0  = np.random.normal(CLASS0_MEAN, CLASS0_STD, n_class0)
    feature_class1  = np.random.normal(CLASS1_MEAN, CLASS1_STD, n_class1)

    #visualize the samples via histograms
    nbins           = 30
    features_both   = [feature_class0, feature_class1]
    list_names      = ['pre-term (class 0)', 'full-term (class 1)']
    plt.figure(1)
    plt.hist(features_both, nbins, label=list_names,  histtype='bar')
    plt.legend(loc='upper left', fontsize=15)
    plt.xlabel('Feature value')
    plt.ylabel('')
    plt.show()

    #make sure it's a column vector
    feature_class0  = feature_class0.reshape(n_class0, 1)
    feature_class1  = feature_class1.reshape(n_class1, 1)

    #assign y = 0 to class 0, y = 1 to class 1
    y_labels_class0 = np.zeros(feature_class0.shape)
    y_labels_class1 = np.ones(feature_class1.shape)

    #visualize samples and class labels as a scatter plot
    plt.figure(2)
    plt.plot(feature_class0, y_labels_class0, '.b', label='pre  (y=0)')
    plt.plot(feature_class1, y_labels_class1, '.r', label='full (y=1)')
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

    #********** logistic regression part

    #create the design matrix for each class and stack them vertically
    X_class0    = np.c_[np.ones((n_class0,1)), feature_class0]
    X_class1    = np.c_[np.ones((n_class1,1)), feature_class1]
    X_design    = np.r_[X_class0, X_class1]

    y_both      = np.r_[y_labels_class0, y_labels_class1]

    #train the model
    model       = sm.Logit(y_both, X_design)
    result      = model.fit()

    print(result.summary())

    print('Exp(beta): ' + str(np.exp(result.params)))

    x_line      = np.linspace(7.5,20,100)
    X_line      = np.c_[np.ones(x_line.shape), x_line]

    #create y_line the logistic regression line
    #remember the form of the logistic function:
    #y = 1/(1 + np.exp(-1*x))

    #**** Once you've created y_line, uncomment this
    # plt.figure(3)
    # plt.plot(feature_class0, y_labels_class0, '.b', label='pre  (y=0)')
    # plt.plot(feature_class1, y_labels_class1, '.r', label='full (y=1)')
    # plt.legend(loc='upper left', fontsize=15)
    # plt.plot(x_line, y_line)
    # plt.show()

    print('done.')