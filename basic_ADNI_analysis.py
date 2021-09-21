import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_tools
import scipy.stats as stats

def plot_groups(df, plot_columns):

    df_CN               = df.loc[df.DX == 'CN']
    df_MCI              = df.loc[df.DX == 'MCI']
    df_AD               = df.loc[df.DX == 'Dementia']

    n_cols              = len(plot_columns)
    n_bins              = 20

    for i, col_i in enumerate(plot_columns):

        list_cols       = [df_CN[plot_columns[i]], df_MCI[plot_columns[i]], df_AD[plot_columns[i]]]
        list_names      = ['CN', 'MCI', 'AD']

        if plot_columns[i]=='Sex':

            plt.figure(i+1)

            n_CN, n_MCI, n_AD = df_CN.shape[0], df_MCI.shape[0], df_AD.shape[0]
            bar_CN      = [np.sum(df_CN.Sex =='Female')/n_CN,   np.sum(df_CN.Sex =='Male')/n_CN]
            bar_MCI     = [np.sum(df_MCI.Sex=='Female')/n_MCI,  np.sum(df_MCI.Sex=='Male')/n_MCI]
            bar_AD      = [np.sum(df_AD.Sex =='Female')/n_AD,   np.sum(df_AD.Sex =='Male')/ n_AD]

            ax_i1       = plt.subplot(1, 1, 1)

            x           = np.arange(0, 2)
            WIDTH       = 0.25

            b1          = plt.bar(x,              bar_CN,   width=WIDTH, label='CN')
            b2          = plt.bar(x + WIDTH,      bar_MCI,  width=WIDTH, label='MCI')
            b3          = plt.bar(x + 2*WIDTH,    bar_AD,   width=WIDTH, label='AD')
            ax_i1.set_xticks(x + WIDTH)
            ax_i1.set_xticklabels(['Female %', 'Male %'])
            plt.ylim((0, 0.75))
            plt.legend([b1,b2,b3], ['CN','MCI','AD'], loc='upper left', fontsize=12)
            plt.title('Sex', fontsize=12)

            continue

        # create histogram
        plt.figure(i+1)
        ax_i1           = plt.subplot(1, 2, 1)

        ax_i1.hist(list_cols, n_bins, label=list_names, histtype='bar')
        ax_i1.legend(loc='upper left', fontsize=12)
        plt.title('Hist ' + plot_columns[i], fontsize=12)

        #create boxplot
        ax_i2           = plt.subplot(1, 2, 2)
        plot_tools.boxplot_scatter(ax_i2, list_cols, ['CN', 'MCI', 'AD'])
        plt.title('Boxplot ' + plot_columns[i], fontsize=12)

    plt.show()

def calc_group_diffs(df, cols):

    df_CN               = df.loc[df.DX == 'CN']
    df_MCI              = df.loc[df.DX == 'MCI']
    df_AD               = df.loc[df.DX == 'Dementia']

    print('Running one-way ANOVAs...')
    for col_i in cols:

        print('****** {} ******'.format(col_i))

        #the describe() method gives you a bunch of super useful summaries of each column in your data frame
        #you get: count, mean, std, min, 25th percentile, median (50th percentile), 75th percentile, max
        cn_i            = df_CN[col_i].describe()
        mci_i           = df_MCI[col_i].describe()
        ad_i            = df_AD[col_i].describe()
        print('mean (std): CN: %.1f (%.1f), MCI: %.1f (%.1f), AD: %.1f (%.1f)'% \
                (cn_i['mean'], cn_i['std'], mci_i['mean'], mci_i['std'], ad_i['mean'], ad_i['std']))

        #f_oneway function from scipy.stats
        #accepts an arbitrary number of numpy arrays (here 3), each a different group
        #runs a one-way ANOVA and returns the F statistic and associated p-value
        F_i, p_i = stats.f_oneway(df_CN[col_i],df_MCI[col_i], df_AD[col_i])

        print('ANOVA: F = %.1f, p-val: %f' % (F_i, p_i))

    print('done.')

if __name__ == '__main__':

    # needs: pip install xlrd==1.2.0
    df_raw              = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit']  = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns  = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)


    #How big is the dataset
    print(df_raw.shape)

    #What kind of measures do we have?
    print('Columns: ', list(df_raw.columns))

    #Which measures do we care about?
    #Basics: RID, age, sex, education
    #Diagnosis (DX) and cognitive measures (ADAS13/MMSE)
    measures_we_care_about = ['RID', 'VISCODE','Age_visit','Years_bl', 'Sex', 'Education_Years',  'DX', 'MMSE']
    df                  = df_raw[measures_we_care_about]
    print(df.shape)

    #How many subjects are there?
    unique_RIDs         = np.unique(df.RID)
    print('Number of unique subjects: ' + str(len(unique_RIDs)))

    #How many visits per subject?
    dup_series          = df.pivot_table(index=['RID'], aggfunc='size')
    vec_dups            = dup_series.to_numpy()
    plt.hist(vec_dups, np.arange(-0.5, max(vec_dups)+0.5), rwidth=0.75)
    plt.title('Histogram of # visits per subject')
    plt.show()

    #Some subjects have had over 20 visits!
    RID_lots_of_visits   = dup_series.index[dup_series > 20]
    print(str(len(RID_lots_of_visits))+ ' subjects have over 20 visits!')

    #inspect this table
    df_lots_of_visits    = df.loc[np.in1d(df.RID, RID_lots_of_visits)]
    df_lots_of_visits    = df_lots_of_visits.sort_values(by=['RID', 'Years_bl'])

    #How many in each diagnostic group? Just looking at first available visit (usually baseline)
    df                  = df.sort_values(by = ['RID','Years_bl'])
    #do this before dropping duplicates
    df_0                = df.dropna(subset=['RID', 'Age_visit', 'Education_Years', 'Sex', 'DX', 'MMSE'])
    df_0                = df_0.drop_duplicates(subset=['RID'])

    #make sure they all have a diagnosis we expect
    assert(np.all(np.in1d(df_0.DX, ['CN', 'MCI', 'Dementia'])))

    #plot basic info via histograms/boxplots/bar plots
    plt.figure(1)
    plot_groups(df_0, ['Age_visit', 'Sex', 'Education_Years', 'MMSE'])

    #finaly, run some statistical tests (ANOVAs) for group differences
    calc_group_diffs(df_0, ['Age_visit', 'Education_Years', 'MMSE'])

    print('done')