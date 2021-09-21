import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt

#load a set of Nifti images, mask them (if mask has been passed in)
#and return a matrix
#assumption: all images have the same dimensionality
def load_image_matrix(nii_filenames, mask_mat=None):

    n               = len(nii_filenames)

    #load mask if it's been passed in
    if mask_mat is not None:
        #raw dimensions of the mask - for checking later
        raw_dims    = mask_mat.shape

        mask_vec    = mask_mat.flatten()
        index_nz    = np.where(mask_vec > 0)[0]
        d           = len(index_nz)

        #allocate the output matrix
        img_mat     = np.zeros((n, d))

    #loop through the filenames
    for i, file_i in enumerate(nii_filenames):
        print('Loading ' + file_i)
        nii_i       = nib.load(file_i)

        #we haven't preallocated the image matrix yet in this case
        if i == 0 and mask_mat is None:
            raw_dims = nii_i.shape

            #no mask - dimensinality is product of dimensions
            d        = np.prod(raw_dims)

            #allocate out matrix and mask vector
            img_mat  = np.zeros((n,d))
            mask_vec = np.ones((d,))
        else:
            assert(np.all(nii_i.shape == raw_dims))

        img_i       = nii_i.get_fdata().flatten()
        img_mat[i,:] = img_i[mask_vec > 0]

    return img_mat

if __name__ == '__main__':

    #****** masking

    #load the template from DARTEL - it's actually a 4D matrix (3D x 2)
    #first 3D matrix is GM template
    #second 3D matrix is WM template
    mask_nii = nib.load('../data_lesson3/ADNI_60_mwrc1/Template_6.nii')

    #We're going to use the GM template
    GM_template_mat = mask_nii.get_fdata()[:,:,:,0]

    #raw GM template
    ax1         = plt.subplot(2, 2, 1)
    ax1.matshow(GM_template_mat[:,:,60], cmap=plt.cm.gray)
    plt.title('Raw Template, d = ' + str(np.prod(GM_template_mat.shape)))

    #*** visuzlize some potential masks
    mask_mat    = np.zeros(GM_template_mat.shape)
    mask_mat[GM_template_mat > 0.01] = 1
    ax2         = plt.subplot(2, 2, 2)
    ax2.matshow(mask_mat[:,:,60], cmap=plt.cm.gray)
    plt.title('Mask 0.01, d = ' + str(np.sum(mask_mat==1)))

    mask_mat = np.zeros(GM_template_mat.shape)
    mask_mat[GM_template_mat > 0.05] = 1
    ax3         = plt.subplot(2, 2, 3)
    ax3.matshow(mask_mat[:,:,60], cmap=plt.cm.gray)
    plt.title('Mask 0.05, d = ' + str(np.sum(mask_mat==1)))

    mask_mat = np.zeros(GM_template_mat.shape)
    mask_mat[GM_template_mat > 0.15] = 1
    ax4         = plt.subplot(2, 2, 4)
    ax4.matshow(mask_mat[:,:,60], cmap=plt.cm.gray)
    plt.title('Mask 0.15, d = ' + str(np.sum(mask_mat==1)))

    plt.show()

    #**** loading images

    nii_files           = glob.glob('../data_lesson3/ADNI_60_mwrc1/mwrc1*.nii')

    #load masked features
    image_mat_masked   = load_image_matrix(nii_files, mask_mat)

    #load raw features
    image_mat_raw       = load_image_matrix(nii_files)

    print('Loaded masked image matrix of size ' + str(image_mat_masked.shape[0])    + ' x ' + str(image_mat_masked.shape[1]))
    print('Loaded raw image matrix of size '    + str(image_mat_raw.shape[0])       + ' x ' + str(image_mat_raw.shape[1]))