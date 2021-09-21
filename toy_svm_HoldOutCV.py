import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#given training data (X_train, y_train) and C-cost param, train an SVC classifier and return it
def train_plot_SVC_classifier(X_train, y_train, SVC_C_cost):

    svc_model       = svm.LinearSVC(C = SVC_C_cost, max_iter=10000)
    svc_model.fit(X_train, y_train.ravel())

    X_class1        = X_train[np.where(y_train==1)[0],  :]
    X_class2        = X_train[np.where(y_train==-1)[0], :]

    #f(x) = beta.T%x + intercept
    beta            = svc_model.coef_[0]
    b0              = svc_model.intercept_

    min_x1          = np.floor(np.min(X_class1[:,1]))
    max_x1          = np.ceil(np.max(X_class1[:,1]))
    x1_line         = np.linspace(min_x1, max_x1, 100)

    #get the f(x) = 0, -1, +1 lines by solving for x2 in f(x) = b0 + b1*x1 + b2*x2
    f0_line         = (   -b0 - beta[0]*x1_line)/beta[1]
    fplus1_line     = ( 1 -b0 - beta[0]*x1_line)/beta[1]
    fminus1_line    = (-1 -b0 - beta[0]*x1_line)/beta[1]

    #plot the data points plus decision line
    plt.plot(X_class1[:,0], X_class1[:,1], '.r', label='class 2 (+1)')
    plt.plot(X_class2[:,0], X_class2[:,1], '.b', label='class 1 (-1)')
    plt.plot(x1_line, f0_line, '-k')
    plt.plot(x1_line, fplus1_line, '--r')
    plt.plot(x1_line, fminus1_line, '--b')
    plt.legend(fontsize=9)

    return svc_model


#generate features and class labels for given mean/covariance and n/class
def generate_X_y(mean_class1, covMat_class1, mean_class2, covMat_class2, n_class):

    #***** generate train set data
    X_class1        = np.random.multivariate_normal(mean_class1, covMat_class1, n_class)
    X_class2        = np.random.multivariate_normal(mean_class2, covMat_class1, n_class)
    X               = np.r_[X_class1, X_class2]

    y1              =    np.ones((n_class, 1))
    y2              = -1*np.ones((n_class, 1))
    y               = np.r_[y1, y2]

    return X,y

if __name__ == '__main__':

    #************************************
    # set simulation parameters

    n_train         = 500
    n_validation    = 200
    n_test          = 100


    CLASS_SEPARATION = 1

    #class mean vectors
    mean_class1     = [-(CLASS_SEPARATION/2), 0]
    mean_class2     = [ (CLASS_SEPARATION/2), 0]

    #create unit vectors v1, v2 in  [1 1], [-1 1] directions
    v1              = np.array([[ 1],  [1]])
    v1              = v1/np.linalg.norm(v1)
    v2              = np.array([[-1],  [1]])
    v2              = v2/np.linalg.norm(v2)

    # create diagonal covariance matrix, same for both classes
    # (v1 @ v1.T) creates a rank 1 matrix with information lying in v1
    # create covaraince matrix with 90% of variance along v1, 10% along v2
    covMat          = 0.9 * (v1 @ v1.T) + 0.1 * (v2 @ v2.T)

    #possible C-cost values
    SVC_C_vals      = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    # ************************************

    np.random.seed(1)

    #*** generate training data
    n_train_class   = np.round(n_train/2).astype(int)
    X_train, y_train = generate_X_y(mean_class1, covMat, mean_class2, covMat, n_train_class)

    #plot training data points
    X_class1        = X_train[np.where(y_train == 1)[0], :]
    X_class2        = X_train[np.where(y_train == -1)[0], :]
    plt.plot(X_class1[:,0], X_class1[:,1], '.r', label='class 2 (+1), train')
    plt.plot(X_class2[:,0], X_class2[:,1], '.b', label='class 1 (-1), train')
    plt.legend(fontsize=12)


    #*** generate validation  data
    #ASSUMPTION: validation and testing data comes from same distributions as training data
    #this is actually a core assumption in classification
    n_val_class     = np.round(n_validation/2).astype(int)
    X_val, y_val    = generate_X_y(mean_class1, covMat, mean_class2, covMat, n_val_class)

    #plot validation data points
    X_class1        = X_val[np.where(y_val == 1)[0], :]
    X_class2        = X_val[np.where(y_val == -1)[0], :]
    plt.figure(1)
    plt.plot(X_class1[:,0], X_class1[:,1], 'xm', label='class 2 (+1), validate')
    plt.plot(X_class2[:,0], X_class2[:,1], 'xc', label='class 1 (-1), validate')
    plt.legend(fontsize=12)
    plt.ylim([-5, 5])
    plt.title('Raw data')
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()

    #*** generate testing data
    #ASSUMPTION: validation and testing data comes from same distributions as training data
    #this is actually a core assumption in classification
    n_test_class    = np.round(n_test/2).astype(int)
    X_test, y_test  = generate_X_y(mean_class1, covMat, mean_class2, covMat, n_test_class)

    #**** Experiment 1: the wrong way to do it: tuning C on test set
    #   1. Combine training + validation set into one big training set input to SVC
    #   2. Train SVC on a range of different C values, each time using big training set
    #   3. See which one minimizes error on testing set (gets the most test set labels right)
    #   4. Use this optimized C and report this testing error
    X_train_big     = np.r_[X_train, X_val]
    y_train_big     = np.r_[y_train, y_val]

    error_rates     = np.zeros(len(SVC_C_vals))

    plt.figure(2)
    for i, C_i in enumerate(SVC_C_vals):
        plt.subplot(2, 3, i+1)

        svc_model_i     = train_plot_SVC_classifier(X_train_big, y_train_big, C_i)

        plt.title('C = ' + str(C_i))
        plt.ylim([-5, 5])
        plt.xlabel('x1')
        plt.ylabel('x2')

        # predict TEST SET labels using trained SVC model
        y_test_pred     = svc_model_i.predict(X_test).reshape(y_test.shape)

        error_rates[i]  = np.sum(y_test != y_test_pred)/len(y_test)
    plt.show()

    #get index where error rate is lowest
    iBest           = np.argmin(error_rates)
    print('Experiment #1: best C: %f, best test set error: %.1f%%' % (SVC_C_vals[iBest], error_rates[iBest]*100))


    #**** Experiment 2: the right way to do it: tuning C on validation set
    #   1. Train SVC on training data across a range of different C values
    #   2. See which one minimizes the validation set error (gets the most validation set labels right)
    #   3. Use this optimized C parameter to predict test set labels and report this testing error
    validation_error_rates      = np.zeros(len(SVC_C_vals))
    training_error_rates        = np.zeros(len(SVC_C_vals))

    plt.figure(3)
    for i, C_i in enumerate(SVC_C_vals):
        plt.subplot(2, 3, i+1)

        #using X_train, y_train and not X_train_big, y_train_big
        svc_model_i     = train_plot_SVC_classifier(X_train, y_train, C_i)

        plt.title('C = ' + str(C_i))
        plt.ylim([-5, 5])
        plt.xlabel('x1')
        plt.ylabel('x2')

        # predict TRAINING SET labels using trained SVC model
        y_train_pred    = svc_model_i.predict(X_train).reshape(y_train.shape)
        training_error_rates[i]  = np.sum(y_train != y_train_pred)/len(y_train)

        # predict VALIDATION SET labels using trained SVC model
        y_val_pred     = svc_model_i.predict(X_val).reshape(y_val.shape)
        validation_error_rates[i]  = np.sum(y_val != y_val_pred)/len(y_val)

    plt.show()

    #get index where validation error rate is lowest
    iBest           = np.argmin(validation_error_rates)
    C_best          = SVC_C_vals[iBest]
    svc_model_best  = train_plot_SVC_classifier(X_train, y_train, C_best)

    # predict TEST SET labels using C that minimized VALIDATION SET error
    y_test_pred     = svc_model_best.predict(X_test).reshape(y_test.shape)
    error_rate_test = np.sum(y_val != y_val_pred) / len(y_val) * 100
    print('Experiment #2: best C: %f, best test set error: %.1f%%' % (C_best, error_rate_test))

    #create the bias-variance trade-off figure
    plt.figure(4)
    plt.plot(np.log10(SVC_C_vals), training_error_rates,    '.-b', label='training error')
    plt.plot(np.log10(SVC_C_vals), validation_error_rates,  '.-r', label='validation error')
    plt.xlabel('Model complexity: Log10 C')
    plt.ylabel('Prediciton error')
    plt.legend()
    plt.show()
    print('done.')