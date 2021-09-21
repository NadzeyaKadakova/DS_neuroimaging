import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def QCmovie(list_nii):

    #get the figure and axis handles
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    #turn interactive mode on, show figure
    plt.ion()
    plt.show()

    qc_notes = []

    #THIS IS A NESTED LOOP
    #the outer loop iterates over a list of *.nii filenames
    #the inner loop slices (in steps) through the third dimension of the matrix
    for i in range(len(list_nii)):
        nii_filename_i = list_nii[i]

        print('{}/{} loading {}'.format(i+1, len(list_nii), nii_filename_i))

        #load nifti and get image matrix
        img_i       = nib.load(nii_filename_i)
        data_i      = img_i.get_fdata()

        #get the number of voxels in dimension 1
        size_dim_0 = data_i.shape[0]

        #get start and end MRI slices
        startNum    = np.round(size_dim_0 * 0.10).astype(int)
        endNum      = np.round(size_dim_0 * 0.90).astype(int)
        iterStep    = 5

        #loop through slices in steps of iterStep
        for j in range(startNum, endNum, iterStep):
            #update data, draw and pause for user
            ax.clear()

            #rotate the figure 90 degrees
            ax.matshow(np.rot90(data_i[j,:,:]), cmap=plt.cm.gray)
            plt.draw()
            plt.pause(0.15)

        #get user's QC note and append to list
        qc_note_i = input('Enter QC note: ')
        qc_notes.append(qc_note_i)

    #close figure
    plt.close(fig)

    return qc_notes