import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import os

from ROI_regression_ADNI import prep_data

if __name__ == '__main__':

    #the prepared DataFrame
    dataFilename = '../data/df_ADNI_ROIs.xlsx'

    #if the DataFrame we need hasn't been prepared yet, do it
    if not os.path.exists(dataFilename):
        df = prep_data()

        #needs: pip install openpyxl
        df.to_excel(dataFilename)
    else:
        df = pd.read_excel(dataFilename)

    #get the atlas - in this case it's the Harvard-Oxford cortical atlas
    nii_atlas               = nib.load('../data_lesson3/HarvardOxford_Cortical_warped.nii')
    atlas_mat               = nii_atlas.get_fdata()

    #read the atlas mapping
    df_atlas                = pd.read_csv('../data_lesson3/HarvardOxford_mapping.csv')

    list_label_ints         = list(df_atlas['Label'])
    list_label_strings      = list(df_atlas['ROI'])

    #setup the design matrix used everywhere
    sex_col                 = (df.Sex=='Female').astype(int)

    #because we have three groups (CN, MCI, AD) we can investigate when GM density actually changes
    #to do this we create two coding variables:
    #coding CN vs (MCI + AD)
    is_not_CN               = (df.DX != 'CN').astype(int)
    #coding (CN + MCI) vs AD
    is_AD                   = (df.DX == 'Dementia').astype(int)
    #so now we can see if GM density changes early (significant is_not_CN variable) or late (significant is_AD var.)
    #or possibly both or neither

    #the design matrix will be:
    # intercept + age + sex (coding variable) + education years + DX (coding variable)
    X_whole                 = np.c_[np.ones((df.shape[0], 1)), df.Age_visit, sex_col, df.Education_Years, is_not_CN, is_AD]

    np.random.seed(1)


    n_whole                 = X_whole.shape[0]
    n_test                  = 10
    n_train                 = n_whole - n_test

    #randomly select n_test (here 10) subjects for testing, the rest are for training
    index_test              = np.random.choice(np.arange(0, n_whole), n_test)
    index_train             = np.setdiff1d(np.arange(0, n_whole), index_test)

    X_train                 = X_whole[index_train,:]
    X_test                  = X_whole[index_test, :]

    MSE_percentages         = np.zeros(atlas_mat.shape)

    for i, label_int_i in enumerate(list_label_ints):
        label_string_i      = list_label_strings[i]

        y_i                 = df[label_string_i].to_numpy().reshape(df.shape[0],1)

        #split targets just as we split design matrix
        y_train_i           = y_i[index_train]
        y_test_i            = y_i[index_test]

        #needs: pip install statsmodels
        model_i             = sm.OLS(y_train_i, X_train)

        #this holds our estimate beta
        results_i           = model_i.fit()

        #does: X_test * beta_estimated
        y_predict_i         = results_i.predict(X_test)

        #calculate mean squared error as a percentage of the true mean of the ROI
        MSE_percentage_i    = np.mean((y_test_i - y_predict_i)**2)/np.mean(y_test_i) * 100

        MSE_percentages[atlas_mat == label_int_i]    = MSE_percentage_i

    print('done')

    #********* VISUALIZE PREDICTION MSE PERCENTAGES
    plt.figure(1)
    ax1         = plt.subplot(2, 2, 1)
    cax         = ax1.matshow(np.rot90(MSE_percentages[60,:,:]), vmax=6, vmin=0, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 2)
    cax         = ax2.matshow(np.rot90(MSE_percentages[:,72,:]), vmax=6, vmin=0, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 3)
    cax         = ax2.matshow(np.rot90(MSE_percentages[:,:,60]), vmax=6, vmin=0, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)
    plt.suptitle('Prediction mean squared error (MSE) percentage')

    plt.show()