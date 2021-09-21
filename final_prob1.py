import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import plot_tools


def prep_data():
    # ***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    # df_raw              = pd.read_excel('../data/ADNIMERGE_final_proj.xlsx')
    df_raw = pd.read_excel('../data/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit'] = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns={'PTGENDER': 'Sex', 'PTEDUCAT': 'Education_Years'}, inplace=True)

    measures_we_care_about = ['RID', 'VISCODE', 'Age_visit', 'Years_bl', 'Sex', 'Education_Years', 'DX',
                              'IMAGEUID']  # 'MMSE',
    df = df_raw[measures_we_care_about]

    # ****************** Freesurfer part
    df_freesurfer = pd.read_excel('../data/ADNI_Freesurfer.xlsx')

    # read features file
    df_lh_rh = pd.read_excel('../data/freesurfer_fields_lh_rh.xlsx')

    # combine left and right hemisphere ROIs
    vol_features = []
    for i in range(df_lh_rh.shape[0]):
        field_i = df_lh_rh.LEFT_HEMISPHERE_ROI[i][3:]
        df_freesurfer[field_i] = df_freesurfer[df_lh_rh.LEFT_HEMISPHERE_ROI[i]] + df_freesurfer[
            df_lh_rh.RIGHT_HEMISPHERE_ROI[i]]

        vol_features.append(field_i)

    measures_we_care_about_FS = ['IMAGEUID'] + vol_features
    df_freesurfer = df_freesurfer[measures_we_care_about_FS]

    # ****************** Merge

    # clean up both first!
    # otherwise the merge doesn't work!
    df = df.dropna(subset=measures_we_care_about)
    df_freesurfer = df_freesurfer.dropna(subset=measures_we_care_about_FS)

    df_out = pd.merge(df, df_freesurfer, how='inner', on='IMAGEUID')

    return df_out


def compare_OLS_ridge(df_bl_CN, vol_features):
    n_bl_CN = df_bl_CN.shape[0]

    n_train = np.round(n_bl_CN * 0.50)
    n_validation = np.round(n_bl_CN * 0.25)
    n_test = n_bl_CN - n_train - n_validation

    index_all = np.arange(n_bl_CN)
    fraction_remain = (n_validation + n_test) / n_bl_CN
    index_train, index_remain = train_test_split(index_all, test_size=fraction_remain, random_state=1)

    fraction_test = n_test / (n_validation + n_test)
    index_validate, index_test = train_test_split(index_remain, test_size=fraction_test, random_state=1)

    # don't forget to scale training/validation data properly
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

    X = np.array(df_bl_CN[
                     vol_features])  # np.c_[df_bl_CN.Age_visit, sex_col, df_bl_CN.Education_Years, df_bl_CN[vol_features]] #
    y = np.array(df_bl_CN.Age_visit).reshape(df_bl_CN.shape[0], 1)

    X_train = X[index_train, :]
    X_validate = X[index_validate, :]
    X_test = X[index_test, :]


    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_validate = scaler.transform(X_validate)
    X_test = scaler.transform(X_test)

    y_train = y[index_train]
    y_validate = y[index_validate]
    y_test = y[index_test]

    # **************** linear regression version
    ols_model = LinearRegression().fit(X_train, y_train)
    pred_ols = ols_model.predict(X_test)

    # RMSE
    error_ols = np.sqrt(np.mean((y_test - pred_ols) ** 2))  # np.mean(np.abs(y_test - pred_ols)) #

    # **************** ridge regression version
    lambda_vals = np.array([1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8])

    # loop through ridge lambda values, train up ridge regression and predict validation set targets, save RMSE
    error_validation = np.zeros((len(lambda_vals), 1))
    for i, lambda_i in enumerate(lambda_vals):
        ridge_model_i = Ridge(alpha=lambda_i)
        ridge_model_i.fit(X_train, y_train)

        pred_i = ridge_model_i.predict(X_validate)

        error_validation[i] = np.sqrt(np.mean((pred_i - y_validate) ** 2))  # np.mean(np.abs(y_validate - pred_i)) #

    # choose best ridge model based on validation set error
    iBest = np.argmin(error_validation)
    lambda_best = lambda_vals[iBest]
    ridge_model_best = Ridge(alpha=lambda_best)
    ridge_model_best.fit(X_train, y_train)
    print('Best ridge model is lambda = %f, with RMSE validation error of %.2f years' % (
    lambda_best, error_validation[iBest]))

    # predict test set targets using best ridge model and calculate test set RMSE
    pred_ridge = ridge_model_best.predict(X_test)
    error_ridge = np.sqrt(np.mean((pred_ridge - y_test) ** 2))  # np.mean(np.abs(pred_ridge - y_test)) #

    print('Brain age model comparision: OLS out-of-sample RMSE: %.2f, ridge out-of-sample RMSE: %.2f' % (
    error_ols, error_ridge))


if __name__ == '__main__':

    # ********* PREP DATA
    dataFilename = '../data/df_final_prob1.xlsx'

    # if the DataFrame we need hasn't been prepared yet, do it
    if not os.path.exists(dataFilename):
        df = prep_data()

        # needs: pip install openpyxl
        df.to_excel(dataFilename)
    else:
        df = pd.read_excel(dataFilename)

    np.random.seed(1)

    # *********************
    # read features file
    df_lh_rh = pd.read_excel('../data/freesurfer_fields_lh_rh.xlsx')
    vol_features = [str[3:] for str in df_lh_rh.LEFT_HEMISPHERE_ROI]  # df_lh_rh.LEFT_HEMISPHERE_ROI[i][3:]

    df = df.sort_values(by=['RID', 'Years_bl'])

    # find subjects who were CN at first visit
    df_bl = df.drop_duplicates(subset=['RID'])
    df_bl_CN = df_bl.loc[df.DX == 'CN']

    # function to compare OLS vs ridge
    compare_OLS_ridge(df_bl_CN, vol_features)

    # take 75% of baseline CN's for training
    n_bl_CN = df_bl_CN.shape[0]
    df_bl_CN_train = df_bl_CN.sample(int(n_bl_CN * 0.75))
    X_train = np.array(df_bl_CN_train[vol_features])
    y_train = np.array(df_bl_CN_train.Age_visit).reshape(df_bl_CN_train.shape[0], 1)

    # remaining 25% bl CN + the rest for testing
    df_bl_rest = df_bl.loc[~np.isin(df_bl.RID, df_bl_CN_train.RID)]
    X_test = np.array(df_bl_rest[vol_features])
    y_test = np.array(df_bl_rest.Age_visit).reshape(df_bl_rest.shape[0], 1)

    # don't forget to scale training/validation data properly
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # train brain age model
    ols_model = LinearRegression().fit(X_train, y_train)

    # see how well it predicts training data
    pred_train = ols_model.predict(X_train)

    # plot y = x line
    x_line = np.linspace(60, 90, 100)
    y_line = x_line

    # *************** plot predicted vs. actual for training data
    plt.figure(1)
    plt.plot(y_train, pred_train, '.')
    plt.plot(x_line, y_line, '--k')
    plt.xlabel('Chronological age')
    plt.ylabel('Estimated brain age')
    plt.title('Brain age vs. chronological age in training set')
    plt.show()

    # predict test subjects' brain age
    pred_test = ols_model.predict(X_test)
    brain_age_gap = pred_test.ravel() - df_bl_rest.Age_visit.ravel()

    # *************** boxplots of brain age gap across diagnostic groups
    # visualize the samples via histograms
    nbins = 30
    brain_ages_all = [brain_age_gap[df_bl_rest.DX == 'CN'], brain_age_gap[df_bl_rest.DX == 'MCI'],
                      brain_age_gap[df_bl_rest.DX == 'Dementia']]
    plt.figure(2)
    ax1 = plt.subplot(1, 1, 1)
    plot_tools.boxplot_scatter(ax1, brain_ages_all, ['CN', 'MCI', 'AD'])
    plt.title('Boxplots of brain age gap across diagnostic groups')
    plt.show()

    # F_stat, p_val   = stats.f_oneway(brain_age_gap[df_bl_rest.DX=='CN'], brain_age_gap[df_bl_rest.DX=='MCI'], brain_age_gap[df_bl_rest.DX=='Dementia'])#pred_test[df_rest.DX=='CN'], pred_test[df_rest.DX=='MCI'], pred_test[df_rest.DX=='Dementia'])
    # print('Differences in predicted brain age gap between CN/MCI/AD, F = %.1f, p = %f, -log10p = %f' %(F_stat, p_val, -np.log10(p_val)))

    # ******************** compare brain age gap between MCI converters/non-converters
    # MCI at baseline
    # df_rest         = df_rest.sort_values(by=['RID', 'Years_bl'])
    # df_rest_bl      = df_rest.drop_duplicates(subset=['RID'])

    # for simplicity, restrict to those who are MCI at baseline (years_bl=0)
    df_rest_MCI_bl = df_bl_rest.loc[(df_bl_rest.DX == 'MCI') & (df_bl_rest.Years_bl == 0)]

    RID_MCI_bl = np.unique(df_rest_MCI_bl.RID)

    df_rest_MCI = df.loc[np.isin(df.RID, RID_MCI_bl)]

    # get first visit after 2 years
    df_rest_MCI_2year = df_rest_MCI.loc[df_rest_MCI.Years_bl >= 2]
    df_rest_MCI_2year = df_rest_MCI_2year.sort_values(by=['RID', 'Years_bl'])
    df_rest_MCI_2year = df_rest_MCI_2year.drop_duplicates(subset=['RID'])

    # retain those who have 2 year+ follow-up
    df_rest_MCI_bl_2year = df_rest_MCI_bl[np.isin(df_rest_MCI_bl.RID, df_rest_MCI_2year.RID)]
    assert (np.all(df_rest_MCI_2year.RID.values == df_rest_MCI_bl_2year.RID.values))

    X_test_new = np.array(df_rest_MCI_bl_2year[vol_features])  # np.array(df_rest_MCI_2year[vol_features])
    X_test_new = scaler.transform(X_test_new)

    # predict test subjects' brain age
    pred_test = ols_model.predict(X_test_new)
    brain_age_gap = pred_test.ravel() - df_rest_MCI_bl_2year.Age_visit.ravel()

    # *************** boxplots of brain age gap across diagnostic groups
    # visualize the samples via histograms
    nbins = 30
    brain_ages_all = [brain_age_gap[df_rest_MCI_2year.DX == 'MCI'], brain_age_gap[df_rest_MCI_2year.DX == 'Dementia']]

    plt.figure(3)
    ax1 = plt.subplot(1, 1, 1)
    plot_tools.boxplot_scatter(ax1, brain_ages_all, ['MCI non-conv.', 'MCI conv.'])
    plt.title('Boxplots of baseline brain age gap for 2+ year MCI non-conv./conv.')
    plt.show()

    F_stat, p_val = stats.f_oneway(brain_age_gap[df_rest_MCI_2year.DX == 'MCI'],
                                   brain_age_gap[df_rest_MCI_2year.DX == 'Dementia'])
    print('Differences in baseline brain age gap between MCI non-conv/conv: F = %.1f, p = %f, -log10p = %f' % (
    F_stat, p_val, -np.log10(p_val)))

    print('done.')
