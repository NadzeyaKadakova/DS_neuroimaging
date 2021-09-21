import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def open_view_nii(nii_filename):
    nii     = nib.load(nii_filename)

    print(type(nii))
    print(nii.header)

    #get the 3D matrix of data
    nii_data    = nii.get_fdata()

    #get the shape of the data - number of voxels in each dimension
    data_shape = nii_data.shape
    print(data_shape)

    #get the middle slice along each dimension, use floor division
    mid_indices = np.array(data_shape)//2

    assert(len(mid_indices)==3 and np.all(mid_indices>0))

    #sagittal plane view
    ax1         = plt.subplot(2, 2, 1)
    ax1.matshow(nii_data[mid_indices[0],:,:], cmap=plt.cm.gray)

    #coronal plane view
    ax2         = plt.subplot(2, 2, 2)
    ax2.matshow(nii_data[:,mid_indices[1],:], cmap=plt.cm.gray)

    #axial plane view
    ax3         = plt.subplot(2, 2, 3)
    ax3.matshow(nii_data[:,:,mid_indices[2]], cmap=plt.cm.gray)

    plt.show()

    print('done.')

if __name__ == '__main__':

    open_view_nii('../data_lesson3/example1.nii')