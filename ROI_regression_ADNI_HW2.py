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

    #sort df by RID and Years_bl then drop duplicates to get first available measure for each subject

    #create Sex (e.g. 1 for Female, 0 Male), is_not_CN, is_AD coding variables

    #design matrix columns: intercept, age, sex, education, intra-cranial volume (ICV), coding for not CN, coding for AD

    #form the training targets for the Hippocampus and WholeBrain ROI volumes

    #Build OLS models for Hippocampus and WholeBrain ROIs and print summary reports

    print('done.')

