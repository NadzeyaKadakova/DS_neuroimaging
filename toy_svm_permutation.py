import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


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

#run the whole train/validate/test hold-out framework on given data, return out-of-sample accuracy
def train_test_SVM_hold_out(X_train, X_val, X_test, y_train, y_val, y_test):

    #possible C-cost values
    SVC_C_vals              = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    validation_error_rates = np.zeros(len(SVC_C_vals))
    training_error_rates    = np.zeros(len(SVC_C_vals))

    for i, C_i in enumerate(SVC_C_vals):
        svc_model_i         = svm.LinearSVC(C=C_i, max_iter=10000)
        svc_model_i.fit(X_train, y_train.ravel())

        # predict TRAINING SET labels using trained SVC model
        y_train_pred        = svc_model_i.predict(X_train).reshape(y_train.shape)
        training_error_rates[i] = np.sum(y_train != y_train_pred) / len(y_train)

        # predict VALIDATION SET labels using trained SVC model
        y_val_pred          = svc_model_i.predict(X_val).reshape(y_val.shape)
        validation_error_rates[i] = np.sum(y_val != y_val_pred) / len(y_val)

    # get index where validation error rate is lowest
    iBest                   = np.argmin(validation_error_rates)
    C_best                  = SVC_C_vals[iBest]

    svc_model_best          = svm.LinearSVC(C=C_best, max_iter=10000)
    svc_model_best.fit(X_train, y_train.ravel())

    # predict TEST SET labels using C that minimized VALIDATION SET error
    y_test_pred             = svc_model_best.predict(X_test).reshape(y_test.shape)
    accuracy_test           = np.sum(y_val == y_val_pred) / len(y_val) * 100

    return accuracy_test

if __name__ == '__main__':

    #************************************
    # set simulation parameters

    n_train         = 500
    n_validation    = 200
    n_test          = 100


    CLASS_SEPARATION = 1 #0.1

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


    #compute quantity of interest: out-of-sample accuracy
    accuracy_OOS_true   = train_test_SVM_hold_out(X_train, X_val, X_test, y_train, y_val, y_test)

    NUM_PERMUTATION_ITERATIONS = 500

    np.random.seed(1)

    #initialize list of permuted quantities of interest
    accuracy_OOS_perm   = []

    for i in np.arange(NUM_PERMUTATION_ITERATIONS):

        #shuffle y's randomly
        y_train_perm_i  = np.random.permutation(y_train)
        y_val_perm_i    = np.random.permutation(y_val)
        y_test_perm_i   = np.random.permutation(y_test)

        # compute quantity of interest for this permutation
        accuracy_OOS_i = train_test_SVM_hold_out(X_train, X_val, X_test, y_train_perm_i, y_val_perm_i, y_test_perm_i)

        #NOTE: to compare two models via permutation test, you'd train/test/validate a 2nd model here and build
        #       up a null distribution of differences in accuracies, then calc p-val of true difference as below

        accuracy_OOS_perm.append(accuracy_OOS_i)

        if (i%20==0):
            print('Iteration: %d' % (i))

    plt.figure(1)
    plt.hist(accuracy_OOS_perm)
    plt.title('Histogram of permuted accuracies (null distribution)')
    plt.show()

    #done with permutation test, compute p-value for true quantity of interest under estimated null distribution
    p_val               = np.sum(accuracy_OOS_perm > accuracy_OOS_true)/NUM_PERMUTATION_ITERATIONS

    print('True estimate of out-of-sample accuracy: %.1f, p-val: %f' % (accuracy_OOS_true, p_val))