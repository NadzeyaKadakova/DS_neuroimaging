import pandas as pd
import numpy as np

#FILL IN THIS FUNCTION THAT CALCULATES FIND MCI CONVERTERS, NON-CONVERTERS AND REGRESSORS
#THEN PRINTS OUT THE RATE OF CONVERSION/NON-CONVERSION/REGRESSION AS A PERCENTAGE
def calc_MCI_rates(df):

    #ADD COMMENTED CODE TO FIND MCI CONVERTERS, NON-CONVERTERS AND REGRESSORS
    df_MCI_converters       = df.loc[df.DX=='Dementia']
    df_MCI_non_converters   = df.loc[df.DX=='MCI']
    df_MCI_regressors       = df.loc[df.DX=='CN']

    #CALCULATE RATE OF CONVERSION AND REGRESSION
    conversion_rate         = np.round(df_MCI_converters.shape[0]        /df.shape[0] * 100)
    non_conversion_rate     = np.round(df_MCI_non_converters.shape[0]    /df.shape[0] * 100)
    regression_rate         = np.round(df_MCI_regressors.shape[0]        /df.shape[0] * 100)

    print('MCI conversion rate: %d%%, non-conversion rate: %d%%, regression rate: %d%%' % (conversion_rate, non_conversion_rate, regression_rate))


if __name__ == '__main__':

    # needs: pip install xlrd==1.2.0
    df_raw                  = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    #keep the measures we care about, drop missing data and sort by RID/Years_bl
    measures_we_care_about  = ['RID', 'Years_bl', 'DX']
    df                      = df_raw[measures_we_care_about]
    df                      = df.dropna(subset=measures_we_care_about)

    #ADD COMMENT:
    df_bl_MCI               = df.loc[(df.DX=='MCI') & (df.Years_bl==0)]
    RID_MCI_bl              = np.unique(df_bl_MCI.RID)

    #ADD COMMENT:
    df                      = df.loc[np.in1d(df.RID, RID_MCI_bl)]
    df                      = df.sort_values(by = ['RID','Years_bl'])

    #ADD COMMENT:
    df_2year                = df.loc[df.Years_bl >= 2]
    df_2year                = df_2year.sort_values(by = ['RID','Years_bl'])
    df_2year                = df_2year.drop_duplicates(subset=['RID'])

    calc_MCI_rates(df_2year)

    #ADD COMMENT:
    df_4year                = df.loc[df.Years_bl >= 4]
    df_4year                = df_4year.sort_values(by = ['RID','Years_bl'])
    df_4year                = df_4year.drop_duplicates(subset=['RID'])

    calc_MCI_rates(df_4year)
