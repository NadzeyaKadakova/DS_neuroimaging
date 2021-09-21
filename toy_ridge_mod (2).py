import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

if __name__ == '__main__':

    n_total                 = 100

    #dimensionality of features, i.e. polynomial model order
    D                       = 30   

    n_train         = np.round(n_total * 0.70)
    n_validation    = np.round(n_total * 0.15)
    n_test          = n_total - n_train - n_validation

    print('n training: %d, n validation: %d, n testing: %d' % (n_train, n_validation, n_test))

    #coefficients of the ideal cubic function
    ideal_beta      = np.array([-1, -0.25, 0.75, 0.25]).reshape(4, 1)

    X_START         = -10
    X_END           = 10
    x_observed      = np.linspace(X_START, X_END, n_total).reshape(n_total, 1)

    #design matrix is: [1 x x^2 x^3]
    X_ideal         = np.c_[np.ones((n_total, 1)), x_observed, x_observed**2, x_observed**3]

    #y = b0 + b1*x + b2*x^2 + b3*x^3 , i.e. X_design @ ideal_beta, where @ is matrix multiplication
    y_ideal         = X_ideal @ ideal_beta

    np.random.seed(1)

    NOISE_MEAN      = 0
    NOISE_STD       = 20
    y_observed      = y_ideal + np.random.normal(NOISE_MEAN, NOISE_STD, (n_total,1))


    #**** plot all data
    plt.figure(1)
    plt.plot(x_observed, y_observed,    '.b', label='observations')
    plt.plot(x_observed, y_ideal,       '-c', label='ideal')
    plt.legend()
    plt.title('Full dataset')
    plt.show()

    #create full data set's feature matrix X with columns: intercept + x + x^2 + ... + x^d
    X               = np.ones((n_total, 1))
    for i in range(1, D + 1):
        X           = np.c_[X, x_observed**i]

    index_all       = np.arange(0, n_total)

    #**** now split the data up into train/validation/test sets
    fraction_remain = (n_validation + n_test)/n_total
    index_train, index_remain                   = train_test_split(index_all,       test_size = fraction_remain, random_state = 1)

    fraction_test   = n_test/(n_validation + n_test)
    index_validate, index_test                  = train_test_split(index_remain,    test_size = fraction_test, random_state = 1)

    #**** plot data after train/validate/test split
    plt.figure(2)
    plt.plot(x_observed[index_train],       y_observed[index_train],    '.b', label='train')
    plt.plot(x_observed[index_validate],    y_observed[index_validate], '.g', label='validate')
    plt.plot(x_observed[index_test],        y_observed[index_test],     '.r', label='test')
    plt.legend()
    plt.title('Train/validate/test split')
    plt.show()

    X_train         = X[index_train, :]
    X_validate      = X[index_validate, :]
    X_test          = X[index_test, :]

    y_train         = y_observed[index_train]
    y_validate      = y_observed[index_validate]
    y_test          = y_observed[index_test]

    #standardize (i.e. z-score) the columns of all train/validate/test sets based on train set
    scaler          = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train)
    X_train         = scaler.transform(X_train)
    X_validate      = scaler.transform(X_validate)
    X_test          = scaler.transform(X_test)

    #****** linear regression
    ols_model       = LinearRegression().fit(X_train, y_train)
    pred_ols        = ols_model.predict(X_test)
    error_test_ols  = np.sqrt(np.mean((pred_ols - y_test)**2))

    print('OLS out-of-sample RMSE: %.1f' % (error_test_ols))

    #****** ridge regression
    lambda_vals     = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

    #loop through ridge lambda values, train up ridge regression and predict validation set targets, save RMSE
    error_validation = np.zeros((len(lambda_vals), 1))
    for i, lambda_i in enumerate(lambda_vals):

        ridge_model_i = Ridge(alpha=lambda_i)
        ridge_model_i.fit(X_train, y_train)

        pred_i      = ridge_model_i.predict(X_validate)

        error_validation[i] = np.sqrt(np.mean((pred_i - y_validate)**2))


    #choose best ridge model based on validation set error
    iBest           = np.argmin(error_validation)
    lambda_best     = lambda_vals[iBest]
    ridge_model_best = Ridge(alpha=lambda_best)
    ridge_model_best.fit(X_train, y_train)

    #predict test set targets using best ridge model and calculate test set RMSE
    pred_best       = ridge_model_best.predict(X_test)
    error_test_ridge = np.sqrt(np.mean((pred_best - y_test) ** 2))

    print('Ridge out-of-sample RMSE: %.1f' % (error_test_ridge))

    #**************** plot OLS and ridge  fits
    n_line              = 1000
    X_START             = -10
    X_END               = 10
    x_line              = np.linspace(X_START, X_END, n_line).reshape(n_line, 1)

    #create X_line, the polynomial feature matrix for the line
    X_line              = np.ones((n_line, 1))
    for i in range(1, D + 1):
        X_line          = np.c_[X_line, x_line**i]

    #standardize X_line as before
    X_line              = scaler.transform(X_line)

    #predict targets on the line with different models
    pred_train_OLS      = ols_model.predict(X_line)
    pred_train_ridge    = ridge_model_best.predict(X_line)

    plt.figure(3)
    plt.plot(x_observed[index_train],   y_observed[index_train],    '.b')
    plt.plot(x_line,                    pred_train_OLS,             '-m', label='OLS')
    plt.plot(x_line,                    pred_train_ridge,           '-c', label='ridge')
    plt.legend()
    plt.ylim([-300, 300])
    plt.title('Training data fits')
    plt.show()

    # output the 2-norm of the OLS and ridge solutions
    print('OLS   ||beta||: ' + str(np.sqrt(np.sum(ols_model.coef_**2))))
    print('Ridge ||beta||: ' + str(np.sqrt(np.sum(ridge_model_best.coef_**2))))

    print('done')