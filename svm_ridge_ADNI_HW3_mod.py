import pandas as pd
import numpy as np
import glob
import nibabel as nib
import matplotlib.pyplot as plt
from compute_atlas_ROIs import load_ROI_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import Ridge

def cross_validated_SVM(X, y):

    n_total                 = X.shape[0]

    #using 10-fold cross-validation here
    NUM_FOLDS               = 10

    #possible C-cost values
    SVC_C_vals              = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1, 100])

    #********************** cross-validation section *************************

    # ******* the scikit learn version - the way you do it in the 'real-world'
    # #create the K-fold cross-validation object
    # from sklearn import model_selection
    # cv                      = model_selection.KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=1)
    #
    # i                       = 0
    # for train_indices_i, validation_indices_i in cv.split(X):
    #
    #     #create train/test split for this CV fold
    #     X_train_i, X_val_i = X[train_indices_i, :], X[validation_indices_i]
    #     y_train_i, y_val_i = y[train_indices_i],    y[validation_indices_i]
    #
    #     ... rest of code ...

    # ******* non-scikit learn verson, to better understand how it all works
    #COMMENT:
    all_indices             = np.arange(n_total)
    np.random.seed(1)
    shuffled_indices        = np.random.permutation(all_indices)

    #COMMENT:
    list_indices            = np.array_split(shuffled_indices, NUM_FOLDS)

    #COMMENT:
    error_training_mat      = np.zeros((NUM_FOLDS, len(SVC_C_vals)))
    error_validation_mat    = np.zeros((NUM_FOLDS, len(SVC_C_vals)))

    #don't forget to scale training/validation data properly
    scaler                  =  StandardScaler(copy=True, with_mean=True, with_std=True)

    #**** FILL THIS IN
    # outer loop through list_indices:
    #   1. setting current list as validation set, remainder as training set for current fold
    #   2. creating current training and validation split of X and y
    #   3. scaling data appropriately
    # inner loop through all possible C values
    #    1. training current model
    #    2. predicting training and validation labels
    #    3. saving current training and validation errors
    for i, list_i in enumerate(list_indices):

        #create train/validation split for this CV fold
        validation_indices_i    = list_i
        train_indices_i = all_indices[~np.isin(all_indices, validation_indices_i)]

        X_train_i, X_val_i      = X[train_indices_i, :], X[validation_indices_i]
        y_train_i, y_val_i      = y[train_indices_i],    y[validation_indices_i]

        scaler.fit(X_train_i)
        X_train_i               = scaler.transform(X_train_i)
        X_val_i                 = scaler.transform(X_val_i)

        for j, C_j in enumerate(SVC_C_vals):

            #print('training Fold: %d, SVM C value: %d' % (i, j))

            # train with this fold's training data
            svc_model_ij        = svm.LinearSVC(C=C_j, max_iter=5000)
            svc_model_ij.fit(X_train_i, y_train_i.ravel())

            # predict this fold's training labels
            y_pred_train_ij     = svc_model_ij.predict(X_train_i).reshape(y_train_i.shape)
            # calculate training error and store it
            error_training_mat[i, j]    = np.sum(y_train_i != y_pred_train_ij) / len(y_train_i)

            # predict this fold's validation labels
            y_pred_val_ij       = svc_model_ij.predict(X_val_i).reshape(y_val_i.shape)
            # calculate validation error and store it
            error_validation_mat[i, j]    = np.sum(y_val_i != y_pred_val_ij) / len(y_val_i)

    #calculate mean training and validation erros across all folds for each model
    mean_training_errors         = np.mean(error_training_mat, axis=0)
    mean_validation_errors       = np.mean(error_validation_mat, axis=0)

    # select best model based on lowest mean validation error
    # **** FILL THIS IN
    iBest                       = np.argmin(mean_validation_errors)
    best_mean_error             = mean_validation_errors[iBest] * 100
    C_best                      = SVC_C_vals[iBest]

    #print best model
    print('Best model is C = %f, with mean validation misclassification error of %.2f%%.' % (C_best, best_mean_error))

    #create the bias-variance trade-off figure
    # **** FILL THIS IN
    plt.figure(1)
    plt.plot(np.log10(SVC_C_vals), mean_training_errors,    '.-b', label='training error')
    plt.plot(np.log10(SVC_C_vals), mean_validation_errors,  '.-r', label='validation error')
    plt.xlabel('Model complexity: Log10 C')
    plt.ylabel('Prediction error')
    plt.legend()
    plt.show()

    #train up best SVC model on full dataset (original X, y)
    #don't forget to standardize
    # **** FILL THIS IN
    svc_model_full           = svm.LinearSVC(C=C_best, max_iter=5000)
    scaler.fit(X)
    X                        = scaler.transform(X)
    svc_model_full.fit(X, y.ravel())

    #get coefficients from full model and return them
    # **** FILL THIS IN
    coef_full               = svc_model_full.coef_.ravel()

    return coef_full

def cross_validated_ridge(X, y):

    # **** FILL THIS IN

    #very similar to above, but using root mean squared error (RMSE)
    #to assess train/validation error within each fold

    #tune the ridge regression using these lambda values: 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8

    #when you've written the function, uncomment this
    ##return coef_full


    n_total                 = X.shape[0]

    #using 10-fold cross-validation here
    NUM_FOLDS               = 10

    #possible lambda values
    ridge_lambda_vals       = np.array([1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8])

    all_indices             = np.arange(n_total)
    np.random.seed(1)
    shuffled_indices        = np.random.permutation(all_indices)

    list_indices            = np.array_split(shuffled_indices, NUM_FOLDS)

    error_training_mat      = np.zeros((NUM_FOLDS, len(ridge_lambda_vals)))
    error_validation_mat    = np.zeros((NUM_FOLDS, len(ridge_lambda_vals)))

    #don't forget to scale training/validation data properly
    scaler                  =  StandardScaler(copy=True, with_mean=True, with_std=True)

    for i, list_i in enumerate(list_indices):

        # create train/validation split for this CV fold
        validation_indices_i    = list_i
        train_indices_i         = all_indices[~np.isin(all_indices, validation_indices_i)]

        X_train_i, X_val_i      = X[train_indices_i, :], X[validation_indices_i]
        y_train_i, y_val_i      = y[train_indices_i],    y[validation_indices_i]

        scaler.fit(X_train_i)
        X_train_i               = scaler.transform(X_train_i)
        X_val_i                 = scaler.transform(X_val_i)

        for j, lamda_j in enumerate(ridge_lambda_vals):

            #print('training Fold: %d, lamda value: %d' % (i, j))

            # train with this fold's training data
            ridge_model_ij              = Ridge(alpha=lamda_j)
            ridge_model_ij.fit(X_train_i, y_train_i.ravel())

            # predict this fold's training targets
            y_pred_train_ij             = ridge_model_ij.predict(X_train_i).reshape(y_train_i.shape)

            # calculate training RMSE and store it
            error_training_mat[i, j]    = np.sqrt(np.mean((y_pred_train_ij - y_train_i)**2))

            # predict this fold's validation labels
            y_pred_val_ij               = ridge_model_ij.predict(X_val_i).reshape(y_val_i.shape)

            # calculate validation RMSE and store it
            error_validation_mat[i, j]  = np.sqrt(np.mean((y_pred_val_ij - y_val_i)**2))

    #calculate mean training and validation erros across all folds for each model
    mean_training_errors         = np.mean(error_training_mat, axis=0)
    mean_validation_errors       = np.mean(error_validation_mat, axis=0)

    # select best model based on lowest mean validation error
    iBest                       = np.argmin(mean_validation_errors)
    best_mean_error             = mean_validation_errors[iBest]
    lamda_best                  = ridge_lambda_vals[iBest]

    #create the bias-variance trade-off figure
    # **** FILL THIS IN
    plt.figure(1)
    plt.plot(-np.log10(ridge_lambda_vals), mean_training_errors,    '.-b', label='training error')
    plt.plot(-np.log10(ridge_lambda_vals), mean_validation_errors,  '.-r', label='validation error')
    plt.xlabel('Model complexity: -Log10 lamda')
    plt.ylabel('Prediction error')
    plt.legend()
    plt.show()

    #print best model
    print('Best model is lambda = %f, with mean validation RMSE of %.2f.' % (lamda_best, best_mean_error))

    #train up best ridge model on full dataset (original X, y)
    #don't forget to standardize
    ridge_model_full        = Ridge(alpha=lamda_best)
    scaler.fit(X)
    X                        = scaler.transform(X)
    ridge_model_full.fit(X, y.ravel())

    #get coefficients from full model and return them
    coef_full               = ridge_model_full.coef_.ravel()

    return coef_full



#visualize the coefficients of classifier/regression as a brain image
def plot_weight_map(beta_vec, atlas_vals_vec, atlas_mat):

    #initialize the brain image of beta (i.e. coefficient) values and set background values (i.e. not in atlas) to NaN
    beta_image              = np.zeros(atlas_mat.shape)
    beta_image[:]           = np.NaN

    #create beta image using atlas_vals_vec (the vector of atlas values that corresponds to beta_vec) and atlas_mat
    for i, val_i in enumerate(atlas_vals_vec):
      beta_image[atlas_mat == val_i] = beta_vec[i]

    #popular solution - problem is it doesn't use atlas_vals_vec:
    # beta_vec                = beta_vec.ravel()
    # for index in np.ndindex(atlas_mat.shape):
    #     x, y, z = index
    #     value = int(atlas_mat[x, y, z])
    #     if value > 0:
    #         beta_image[x, y, z] = beta_vec[value-1]

    mid_indices             = np.array(atlas_mat.shape) // 2

    #************ visualize coefficients as a brain

    #choose the 'jet' colormap and set the 'bad' values (i.e. NaNs) to black, creating a black background
    import copy
    cmap_local              = copy.copy(plt.cm.jet)
    cmap_local.set_bad(color='black')

    V_ABS_MAX               = np.max(np.abs(beta_vec))

    plt.subplot(2, 2, 1)
    img                     = plt.imshow(np.rot90(beta_image[mid_indices[0], :, :]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    plt.subplot(2, 2, 2)
    img                     = plt.imshow(np.rot90(beta_image[:, mid_indices[1], :]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    plt.subplot(2, 2, 3)
    img                     = plt.imshow(np.rot90(beta_image[:, :, mid_indices[2]]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    #done outside function
    #plt.show()

#visualize the coefficients of classifier/regression as a brain image
def plot_weight_map_v2(beta_vec, atlas_vals_vec, atlas_mat):

    #initialize the brain image of beta (i.e. coefficient) values and set background values (i.e. not in atlas) to NaN
    beta_image              = np.zeros(atlas_mat.shape)
    beta_image[:]           = np.NaN

    #create beta image using atlas_vals_vec (the vector of atlas values that corresponds to beta_vec) and atlas_mat
    #for i, val_i in enumerate(atlas_vals_vec):
    #  beta_image[atlas_mat == val_i] = beta_vec[i]

    #popular solution - problem is it doesn't use atlas_vals_vec:
    beta_vec                = beta_vec.ravel()
    for index in np.ndindex(atlas_mat.shape):
        x, y, z = index
        value = int(atlas_mat[x, y, z])
        if value > 0:
            beta_image[x, y, z] = beta_vec[value-1]

    mid_indices             = np.array(atlas_mat.shape) // 2

    #************ visualize coefficients as a brain

    #choose the 'jet' colormap and set the 'bad' values (i.e. NaNs) to black, creating a black background
    import copy
    cmap_local              = copy.copy(plt.cm.jet)
    cmap_local.set_bad(color='black')

    V_ABS_MAX               = np.max(np.abs(beta_vec))

    plt.subplot(2, 2, 1)
    img                     = plt.imshow(np.rot90(beta_image[mid_indices[0], :, :]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    plt.subplot(2, 2, 2)
    img                     = plt.imshow(np.rot90(beta_image[:, mid_indices[1], :]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    plt.subplot(2, 2, 3)
    img                     = plt.imshow(np.rot90(beta_image[:, :, mid_indices[2]]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    #done outside function
    #plt.show()


if __name__ == '__main__':

    #***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    df_raw              = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit']  = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns  = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)

    #we care about all the basics as before, plus Image UID
    measures_we_care_about = ['RID', 'DX', 'MMSE', 'IMAGEUID']
    df                  = df_raw[measures_we_care_about]
    print(df.shape)

    #****************** CREATE IMAGING FEATURES
    # *** load images
    #get the file names we want to load
    nii_files               = glob.glob('../data_lesson3/ADNI_60_mwrc1/mwrc1*.nii')

    #get the Image UIDs from the raw filenames
    imageUIDs               = np.array([int(str_i.split('_')[-1].split('.')[0][1:]) for str_i in nii_files])

    #get the atlas - in this case it's the Harvard-Oxford cortical atlas
    nii_atlas               = nib.load('../data_lesson3/HarvardOxford_Cortical_warped.nii')
    atlas_mat               = nii_atlas.get_fdata()
    #compute ROI means based on the atlas
    X, atlas_vals_vec       = load_ROI_matrix(nii_files, atlas_mat)


    #sort them from smallest to largest
    index_sorted            = np.argsort(imageUIDs)
    X                       = X[index_sorted, :]
    imageUIDs               = imageUIDs[index_sorted]

    #intersect ADNIMERGE with available image UIDs and sort them
    df                      = df.loc[np.isin(df.IMAGEUID, imageUIDs)]
    df                      = df.sort_values(by = ['IMAGEUID'])
    assert(np.all(df.IMAGEUID==imageUIDs))

    n                       = df.shape[0]

    #coding AD vs (CN + MCI)
    y_svc                   = np.array(df.DX == 'Dementia').astype(int).reshape(n, 1)
    #change from 0/1 to -1/1 labelling to be consistent with SVC setup
    y_svc[y_svc==0]         = -1

    # ****************** SVC PART
    print('***** Building AD vs (MCI + CN) classifier')
    beta_svm                = cross_validated_SVM(X, y_svc)

    plot_weight_map(beta_svm, atlas_vals_vec, atlas_mat)
    plt.suptitle('SVC coefficients from AD vs (MCI + CN) classifier')
    plt.show()

    # #permute atlas_vals_vec and beta_svm in the same way
    # shuffled_index          = np.random.permutation(np.arange(len(beta_svm)))
    # beta_svm                = beta_svm[shuffled_index]
    # atlas_vals_vec          = atlas_vals_vec[shuffled_index]

    #re-plot using original solution
    plot_weight_map(beta_svm, atlas_vals_vec, atlas_mat)
    plt.suptitle('SVC coefficients from AD vs (MCI + CN) classifier')
    plt.show()

    # re-plot using alternative solution
    plot_weight_map_v2(beta_svm, atlas_vals_vec, atlas_mat)
    plt.suptitle('SVC coefficients from AD vs (MCI + CN) classifier')
    plt.show()

    # ****************** RIDGE PART
    print('***** Building ridge-based predictor of MMSE ')
    y_ridge                 = np.array(df.MMSE).reshape(n,1)
    beta_ridge              = cross_validated_ridge(X, y_ridge)

    plot_weight_map(beta_ridge, atlas_vals_vec, atlas_mat)
    plt.suptitle('Ridge regression coefficients from MMSE predictor')
    plt.show()
