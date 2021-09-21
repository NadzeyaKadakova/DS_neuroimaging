import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    df_raw                  = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit']     = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns   = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)

    #convert cubic mm to cubic cm
    df_raw.ICV              = df_raw.ICV/1e3
    df_raw.Hippocampus      = df_raw.Hippocampus/1e3
    df_raw.WholeBrain       = df_raw.WholeBrain/1e3

    #retain 3T scans
    df_raw                      = df_raw.loc[df_raw.FLDSTRENG=='3 Tesla MRI']

    #we care about all the basics as before, plus Freesurfer-derived ROIs
    measures_we_care_about = ['RID', 'VISCODE','Age_visit','Years_bl', 'Sex', 'Education_Years',  'DX', 'Hippocampus']
    df                      = df_raw[measures_we_care_about]

    #throw out rows with missing data in any of these fields
    df                      = df.dropna(subset=measures_we_care_about)

    df                      = df.loc[(df.DX=='Dementia') | (df.DX == 'CN')]

    #sort df by RID and Years_bl then drop duplicates to get first available measure for each subject
    df                      = df.sort_values(by=['RID', 'Years_bl'])
    df                      = df.drop_duplicates(subset=['RID'])

    #setup the design matrix used everywhere
    Sex                     = (df.Sex=='Female').astype(int)

    is_AD                   = (df.DX == 'Dementia').astype(int)

    X_design                = np.c_[np.ones((df.shape[0], 1)), df.Age_visit, Sex, df.Education_Years, df.Hippocampus]
    x_names                 = ['Intercept', 'Age at visit', 'Sex', 'Ed. Years', 'Hippocampus']
    model                   = sm.Logit(is_AD, X_design)
    results                 = model.fit()
    print(results.summary(xname=x_names, yname='is_AD'))

    #np.mean(df.Hippocampus[df.DX=='CN'])

    # np.exp(7.5 * -0.01 * -1.3583)
    # 1.11
    # np.exp(7.5 * -0.05 * -1.3583)
    # 1.66

    #*************************************************

    X_design                = np.c_[np.ones((df.shape[0], 1)), df.Hippocampus]
    model                   = sm.Logit(is_AD, X_design)
    results                 = model.fit()
    print(results.summary())

    x_line                  = np.linspace(3,10,100)
    X_line                  = np.c_[np.ones(x_line.shape), x_line]

    #create the logistic regression line
    #y = 1/(1 + np.exp(-1*x))
    y_line                  = 1/(1 + np.exp(-1*X_line@results.params))

    x_CN                    = df.Hippocampus[df.DX=='CN']
    y_CN                    = np.zeros(x_CN.shape)

    x_AD                    = df.Hippocampus[df.DX=='Dementia']
    y_AD                    = np.ones(x_AD.shape)

    plt.figure(3)
    plt.plot(x_CN, y_CN, '.b', label='CN (y=0)')
    plt.plot(x_AD, y_AD, '.r', label='AD (y=1)')
    plt.legend(loc='upper right', fontsize=12)
    plt.plot(x_line, y_line)

    plt.xlabel('Hippocampal volume (cm3)', fontsize=12)
    plt.ylabel('P(AD|hipp. vol.)', fontsize=12)
    plt.show()

    print('done.')

