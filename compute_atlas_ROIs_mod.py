import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

def load_ROI_matrix(nii_filenames, atlas_mat):

    n               = len(nii_filenames)

    #get unique values in atlas, remove 0 value
    atlas_values    = np.unique(atlas_mat).astype(int)
    atlas_values    = np.array(list(set(atlas_values).difference({0})))

    d               = len(atlas_values)

    out_mat         = np.zeros((n, d))

    #loop through the filenames
    for i, file_i in enumerate(nii_filenames):
        print('Loading ' + file_i)
        nii_i       = nib.load(file_i)

        img_i       = nii_i.get_fdata()

        assert(np.all(img_i.shape == atlas_mat.shape))

        #for each image, loop through all atlas values
        for j, atlas_val_j in enumerate(atlas_values):

            #take the mean of the image at this value's locations
            out_mat[i,j]  = np.mean(img_i[atlas_mat == atlas_val_j])

    return out_mat, atlas_values

#Loop through the keys in atlas_dict, printing ROI means for given subjects
#accepts:
#   roi_means       - (n x d) matrix of ROI means for each subject
#   atlas_num_vals  - the d ROI indices used in roi_means
#   atlas_dict      - (d -> d) dictionary mapping ROI name to ROI index
#   subj_indices    - a container (list/np array) of subject indices whose ROIs to print
#
# if subj_indices = [0, 10], output should look like:
# Subject 0
#   Frontal Pole, x
#   Insular Cortex, y
#   Superior Frontal Gyrus, z
#   etc.
# Subject 10
#   Frontal Pole, x
#   Insular Cortex, z
#   Superior Frontal Gyrus, z
#   etc.
def print_ROI_means(roi_means, atlas_num_vals, atlas_dict, subj_indices):

    #loop through subjects
    for subj_i in subj_indices:

        #make sure it's a valid index
        assert (subj_i >= 0 & subj_i < roi_means.shape[0])

        roi_i = roi_means[subj_i, :]

        print('Subject ' + str(subj_i))

        #loop through keys
        for key_i in atlas_dict.keys():

            #get ROI where index of atlas_num_vals that matches this key's value
            val_i   = roi_i[atlas_num_vals == atlas_dict[key_i]]
            print("\t%s, %.2f" % (key_i, val_i.item()))

if __name__ == '__main__':

    nii_template    = nib.load('../data_lesson3/ADNI_60_mwrc1/Template_6.nii')
    nii_atlas       = nib.load('../data_lesson3/HarvardOxford_Cortical_warped.nii')

    GM_template_mat = nii_template.get_fdata()[:,:,:,0]
    atlas_mat       = nii_atlas.get_fdata()

    print(GM_template_mat.shape)
    print(atlas_mat.shape)

    ax1         = plt.subplot(1, 2, 1)
    ax1.matshow(GM_template_mat[:,72,:], cmap=plt.cm.gray)
    plt.title('Template')

    ax2         = plt.subplot(1, 2, 2)
    ax2.matshow(atlas_mat[:,72,:], cmap=plt.cm.gray)
    plt.title('Atlas')
    plt.show()

    nii_files = glob.glob('../data_lesson3/ADNI_60_mwrc1/mwrc1*.nii')

    #calculate ROI means for each subject
    roi_means, atlas_num_vals = load_ROI_matrix(nii_files, atlas_mat)

    df_atlas = pd.read_csv('../data_lesson3/HarvardOxford_mapping.csv')

    #build up a dictionary from pandas data frame columns
    atlas_dict = {}
    for i, roi in enumerate(df_atlas['ROI'].values):
        atlas_dict[roi] = df_atlas['Label'].values[i]

    #call print_ROI_means(roi_means, atlas_num_vals, atlas_dict, 0)
    #what does it print?
    print_ROI_means(roi_means, atlas_num_vals, atlas_dict, [0, 10])