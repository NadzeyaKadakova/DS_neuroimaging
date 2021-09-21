import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import plot_tools
import scipy.stats as stats

def prep_data_prob2():

    measures_we_care_about = ['RID', 'VISCODE', 'Age_visit', 'Years_bl', 'Sex', 'Education_Years', 'DX', 'ADAS13', \
                              'MMSE', 'ABETA', 'TAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 'ICV']


    #***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    df_raw              = pd.read_excel('../data/ADNIMERGE_final_proj.xlsx')

    df_raw['Age_visit']  = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns  = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)

    #measures_we_care_about = ['RID', 'VISCODE','Age_visit','Years_bl', 'Sex', 'Education_Years',  'DX', 'ADAS13', 'MMSE', 'TAU', 'FDG', 'AV45', 'Hippocampus', 'Ventricles', 'WholeBrain', 'ICV']

    df                  = df_raw[measures_we_care_about]
    df                  = df.dropna(subset=measures_we_care_about)

    return df

#FROM: https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
from matplotlib.patches import Ellipse
def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs    = np.linalg.eigh(cov)
    order               = eigvals.argsort()[::-1]
    eigvals, eigvecs    = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy              = eigvecs[:,0][0], eigvecs[:,0][1]
    theta               = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height       = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height, angle=np.degrees(theta), **kwargs)

#run the PCA+GMM model and create log odds scatter plots for given feature set
def run_analysis(df, features):
    np.random.seed(1)

    df_full         = df.sort_values(by=['RID', 'Years_bl'])

    #get baseline
    df_bl           = df_full.loc[df.Years_bl==0]

    X               = np.array(df_bl[features])

    scaler          = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X)
    X               = scaler.transform(X)

    #***** PCA
    pca                     = PCA()
    pca.fit(X)

    #plot cumulative explained variance
    plt.figure(1)
    plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), '.-')
    plt.xlabel('Number of PCs')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    T                       = pca.components_[0:2, :].T

    X_proj                  = X @ T

    #****** GMM PART
    K_vals                  = [1, 2, 3, 4, 5]
    BICs                    = []
    AICs                    = []

    for K_i in K_vals:
        gmm_model           = GaussianMixture(n_components=K_i, random_state=1)
        gmm_model           = gmm_model.fit(X_proj)

        BICs.append(gmm_model.bic(X_proj))
        AICs.append(gmm_model.aic(X_proj))

    plt.figure(2)
    plt.subplot(1,2,1)
    plt.plot(K_vals, BICs, '.-')
    plt.xlabel('K')
    plt.title('BIC')
    plt.subplot(1,2,2)
    plt.plot(K_vals, AICs, '.-')
    plt.xlabel('K')
    plt.title('AIC')
    plt.show()

    #choosing K = 2
    gmm_model               = GaussianMixture(n_components=2, random_state=1)
    gmm_model               = gmm_model.fit(X_proj)

    MEAN1                   = gmm_model.means_[0, :]
    COV1                    = gmm_model.covariances_[0, :, :]

    MEAN2                   = gmm_model.means_[1, :]
    COV2                    = gmm_model.covariances_[1, :, :]

    labels_gmm              = gmm_model.predict(X_proj)


    NUM_STDs_ELLIPSE        = 2

    plt.figure(3)
    f                       = plt.figure(figsize=(7, 3))

    ax                      = f.add_subplot(121)
    X_proj_CN               = X_proj[df_bl.DX == 'CN',:]
    X_proj_MCI              = X_proj[df_bl.DX == 'MCI', :]
    X_proj_AD               = X_proj[df_bl.DX == 'Dementia', :]

    plt.plot(X_proj_CN[:,0],    X_proj_CN[:,1],     '.b', label='CN')
    plt.plot(X_proj_MCI[:, 0],  X_proj_MCI[:, 1],   '.m', label='MCI')
    plt.plot(X_proj_AD[:, 0],   X_proj_AD[:, 1],    '.r', label='AD')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Scatter plot by diagnostic group')
    plt.legend()

    #number of std. dev.'s enclosed by ellipses
    ax2                      = f.add_subplot(122)
    ax2.scatter(X_proj[labels_gmm==0, 0],    X_proj[labels_gmm==0,1], color='r', s=4, label='cluster 1')
    ellipse1                = get_cov_ellipse(COV1, MEAN1, NUM_STDs_ELLIPSE, fc='r', alpha=0.1)
    ax2.add_patch(ellipse1)
    ax2.scatter(X_proj[labels_gmm==1, 0],    X_proj[labels_gmm==1,1], color='b', s=4, label='cluster 2')
    ellipse2                = get_cov_ellipse(COV2, MEAN2, NUM_STDs_ELLIPSE, fc='b', alpha=0.1)
    ax2.add_patch(ellipse2)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Scatter plot by GMM cluster')
    plt.legend()
    plt.show()

    #find all MCIs in the dataset
    MCI_RIDs                = df_bl.RID[df_bl.DX=='MCI']

    #find those who have a follow-up sample 2+ years later, retain first sample after that time
    df_MCI_2year            = df_full.loc[np.isin(df_full.RID, MCI_RIDs) & (df_full.Years_bl >= 2)]
    df_MCI_2year            = df_MCI_2year.sort_values(by=['RID', 'Years_bl'])
    df_MCI_2year            = df_MCI_2year.drop_duplicates(subset=['RID'])

    #retain those who have 2 year+ follow-up
    df_MCI_bl_2year         = df_bl[np.isin(df_bl.RID, df_MCI_2year.RID)]
    assert(np.all(df_MCI_bl_2year.RID.values==df_MCI_2year.RID.values))

    #get the baseline projected features for these subjects
    #probably some faster way to do this, but lets just do a loop
    responsibilities_mat    = np.zeros((df_MCI_2year.shape[0], 2))
    for i in range(df_MCI_2year.shape[0]):
        X_proj_i             = X_proj[df_bl.RID==df_MCI_bl_2year.RID.values[i], :]
        responsibilities_mat[i, :] = gmm_model.predict_proba(X_proj_i)

    #calc log odds of being in disease cluster = log(p(disease)/1-p(disease))
    #                                          = log(responsibility of disease cluster over sample/resp. of CN cluster)
    log_odds_AD             = np.log(responsibilities_mat[:, 0] / responsibilities_mat[:, 1])

    #get log odds of converters (those with Dementia at 2+ year follow-up) vs non-conv. (those still MCI)
    log_odds_conv           = log_odds_AD[df_MCI_2year.DX == 'Dementia']
    log_odds_non_conv       = log_odds_AD[df_MCI_2year.DX == 'MCI']

    #create boxplot+scatter plot of two groups and compute group diffs via one-way ANOVA
    plt.figure(4)
    ax                      = plt.subplot(1, 1, 1)
    plot_tools.boxplot_scatter(ax, [log_odds_non_conv, log_odds_conv], ['MCI non-conv.', 'MCI conv.'])
    plt.title('Log odds of being in disease cluster for 2 year MCI non-conv./converters.')
    plt.show()

    F_stat, p_val           = stats.f_oneway(log_odds_non_conv, log_odds_conv)
    print('Differences in disease cluster log odds between MCI non-conv/conv, F = %.1f, p = %f, -log10p = %f' %(F_stat, p_val, -np.log10(p_val)))

    print('done.')

if __name__ == '__main__':

    #********* PREP DATA
    dataFilename    = '../data/df_final_prob2.xlsx'

    #if the DataFrame we need hasn't been prepared yet, do it
    if not os.path.exists(dataFilename):
        df          = prep_data_prob2()

        #needs: pip install openpyxl
        df.to_excel(dataFilename)
    else:
        df          = pd.read_excel(dataFilename)

    #to keep things modular I created a run_analysis function that does all the analysis for problem 2
    #accepting the desired feature set as a parameter

    #run the analysis for full feature set
    all_features        = ['ADAS13', 'MMSE', 'ABETA', 'TAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 'ICV']
    run_analysis(df, all_features)

    #... and for reduced 'MRI-only' feature set
    mri_features        = ['Hippocampus', 'Ventricles', 'WholeBrain', 'ICV']
    run_analysis(df, mri_features)