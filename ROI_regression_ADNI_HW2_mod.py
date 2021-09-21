import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

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

    #we care about all the basics as before, plus a few Freesurfer-derived ROIs
    measures_we_care_about = ['RID', 'VISCODE','Age_visit','Years_bl', 'Sex', 'Education_Years',  'DX', 'Hippocampus', 'WholeBrain', 'ICV']
    df                      = df_raw[measures_we_care_about]

    #throw out rows with missing data in any of these fields
    df                      = df.dropna(subset=measures_we_care_about)

    #sort df by RID and Years_bl then drop duplicates to get first available measure for each subject
    df                      = df.sort_values(by=['RID', 'Years_bl'])
    df                      = df.drop_duplicates(subset=['RID'])

    #setup the design matrix used everywhere
    is_Female               = (df.Sex=='Female').astype(int)

    #because we have three groups (CN, MCI, AD) we can investigate when GM density actually changes
    #to do this we create two coding variables:
    #coding (MCI + AD) vs CN
    is_not_CN               = (df.DX != 'CN').astype(int)
    #coding AD vs (CN + MCI)
    is_AD                   = (df.DX == 'Dementia').astype(int)

    X_design                = np.c_[np.ones((df.shape[0], 1)), df.Age_visit, is_Female, df.Education_Years, df.ICV, is_not_CN, is_AD]

    y_Hippocampus           = df['Hippocampus'].to_numpy().reshape(df.shape[0],1)
    model_Hipp              = sm.OLS(y_Hippocampus, X_design)
    results_Hipp            = model_Hipp.fit()
    print(results_Hipp.summary())

    y_WholeBrain            = df['WholeBrain'].to_numpy().reshape(df.shape[0], 1)
    model_WholeBrain        = sm.OLS(y_WholeBrain, X_design)
    results_WholeBrain      = model_WholeBrain.fit()
    print(results_WholeBrain.summary())

    print('done.')

