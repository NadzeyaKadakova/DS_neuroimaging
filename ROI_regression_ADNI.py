import pandas as pd
import glob
import nibabel as nib
from compute_atlas_ROIs import load_ROI_matrix
import numpy as np
import matplotlib.pyplot as plt
import basic_ADNI_analysis

import statsmodels.api as sm
import os

#prepare the a DataFrame with the ROI-based features we'll need in this analysis
def prep_data():

    #***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    df_raw              = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit']  = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns  = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)

    #we care about all the basics as before, plus Image UID
    measures_we_care_about = ['RID', 'VISCODE','Age_visit','Years_bl', 'Sex', 'Education_Years',  'DX', 'MMSE', 'IMAGEUID']
    df                  = df_raw[measures_we_care_about]
    print(df.shape)

    #****************** IMAGING PART

    #get the file names we want to load
    nii_files               = glob.glob('../data_lesson3/ADNI_60_mwrc1/mwrc1*.nii')

    #get the atlas - in this case it's the Harvard-Oxford cortical atlas
    nii_atlas               = nib.load('../data_lesson3/HarvardOxford_Cortical_warped.nii')
    atlas_mat               = nii_atlas.get_fdata()

    df_atlas                = pd.read_csv('../data_lesson3/HarvardOxford_mapping.csv')

    #create dictionary, mapping label int to label string
    dict_atlas              = {}
    for i, roi in enumerate(df_atlas['ROI']):
        dict_atlas[df_atlas['Label'][i]] = df_atlas['ROI'][i]

    #compute ROI means based on the atlas
    roi_means, atlas_num_vals = load_ROI_matrix(nii_files, atlas_mat)

    #get the Image UIDs from the raw filenames
    # example filename: #ADNI_024_S_4084_MR_MT1__GradWarp__N3m_Br_20111008155405044_S122287_I260273.nii
    imageUIDs               = np.array([int(str_i.split('_')[-1].split('.')[0][1:]) for str_i in nii_files])

    #get sorted indices and re-index roi_means rows based on it
    index_sorted            = np.argsort(imageUIDs)
    imageUIDs               = imageUIDs[index_sorted]
    roi_means               = roi_means[index_sorted, :]

    #make sure the roi_means labels are in the same order as the dict_atlas labels
    assert(np.all(df_atlas['Label'] == atlas_num_vals))
    df_roi_means            = pd.DataFrame(roi_means, columns=df_atlas['ROI'])
    df_roi_means['IMAGEUID'] = imageUIDs

    #do an inner merge (intersection of rows) between ADNIMERGE and the ROI data frame
    df                      = pd.merge(df, df_roi_means, how = 'inner', on='IMAGEUID')

    return df

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

    #Excluded - but in theory it often helps to mean center the column of the design matrix
    #This helps prevent the conditioning problem in OLS regression
    # X_train_1               = np.c_[df.Age_visit, sex_col, df.Education_Years, is_not_CN, is_AD]
    # X_train_1               = X_train_1 - np.mean(X_train_1, axis=0)
    # X_train                 = np.c_[np.ones((df.shape[0], 1)), X_train_1]

    #the design matrix will be:
    # intercept + age + sex (coding variable) + education years + DX (coding variable)
    X_train                 = np.c_[np.ones((df.shape[0], 1)), df.Age_visit, sex_col, df.Education_Years, is_not_CN, is_AD]

    Bonferroni_factor       = len(list_label_ints)

    t_mat                   = np.zeros(atlas_mat.shape)
    nlog_p_val_mat          = np.zeros(atlas_mat.shape)

    for i, label_int_i in enumerate(list_label_ints):
        label_string_i      = list_label_strings[i]

        #create the design matrix for this ROI
        y_train_i           = df[label_string_i].to_numpy().reshape(df.shape[0],1)

        #needs: pip install statsmodels
        model_i             = sm.OLS(y_train_i, X_train)
        results_i           = model_i.fit()

        #you can check that this is the same as our OLS estimate
        #beta_sm_i           = np.vstack(results_i.params)
        #beta_my_ols_i       = np.linalg.pinv(X_train) @ y_train_i
        #print('OLS param sum |diff|: ' + str(np.sum(np.abs(beta_sm_i-beta_my_ols_i))))

        #save the t-stat and corrected p-value from the AD vs (CN + MCI) contrast
        t_i                 = results_i.tvalues[-1]
        #multiply raw p-value by Bonferroni factor - increase the threshold for significance
        p_corr_i            = min(results_i.pvalues[-1] * Bonferroni_factor, 1)

        str_sig             = '*' if p_corr_i < 0.05 else ' '
        nlog10_p            = -np.log10(p_corr_i)

        t_mat[atlas_mat == label_int_i]             = t_i
        nlog_p_val_mat[atlas_mat == label_int_i]    = nlog10_p

        #print out the t-stat and uncorrected p-value from the (MCI + ADN) vs CN contrast
        #these aren't significant so we won't save or visualize them
        t_MCI_i             = results_i.tvalues[-2]
        p_MCI_i             = results_i.pvalues[-2]

        print('%50s AD vs (MCI + CN) t = %4.1f, corr. p-val: %.5f %s -log10: %4.1f, (AD + MCI) vs CN t = %4.1f, p-val: %.5f' % \
                            (label_string_i, t_i, p_corr_i, str_sig, nlog10_p, t_MCI_i, p_MCI_i))

    #********* VISUALIZE -LOG10 P-VALUES
    #visualize the negative log p-val matrix on the same scale (-10 log p from 0 to 4.5)
    plt.figure(1)
    ax1         = plt.subplot(2, 2, 1)
    cax         = ax1.matshow(np.rot90(nlog_p_val_mat[60,:,:]), vmax=0, vmin=4.5, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 2)
    cax         = ax2.matshow(np.rot90(nlog_p_val_mat[:,72,:]), vmax=0, vmin=4.5, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 3)
    cax         = ax2.matshow(np.rot90(nlog_p_val_mat[:,:,60]), vmax=0, vmin=4.5, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)
    plt.suptitle('AD vs (CN + MCI) -log10(corrected p-value)')

    #********* VISUALIZE t-stats on a -5.5 to 0 scale
    plt.figure(2)
    ax1         = plt.subplot(2, 2, 1)
    cax         = ax1.matshow(np.rot90(t_mat[60,:,:]), vmax=-5.5, vmin=0, cmap=plt.cm.Reds)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 2)
    cax         = ax2.matshow(np.rot90(t_mat[:,72,:]), vmax=-5.5, vmin=0, cmap=plt.cm.Reds)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 3)
    cax         = ax2.matshow(np.rot90(t_mat[:,:,60]), vmax=-5.5, vmin=0, cmap=plt.cm.Reds)
    plt.colorbar(cax)
    plt.suptitle('AD vs (CN + MCI) t-stats')

    plt.show()


    #Inspect the model for the parahippocampal gyrus - the ROI with the biggest drop in significance compared to the
    #previous ANOVA analysis we did
    y           = df['Parahippocampal Gyrus posterior division'].to_numpy().reshape(df.shape[0],1)
    #model: y ~ intercept + age + sex + education years + is_not_CN + is_AD
    model       = sm.OLS(y, X_train)
    result      = model.fit()
    print(result.summary())

    print('done')