import glob
from QCmovie import QCmovie
from datetime import datetime

if __name__ == '__main__':

    #find all *.nii files in the 'subset_ADNI_images' subfolder of 'data'
    nii_files = glob.glob('../data_lesson3/some_raw_images/*.nii')

    #sort the filenames list in-place
    nii_files.sort()

    #get the image ids from the filename
    image_ids = [str_i.split('/')[-1].split('.')[0]  for str_i in nii_files]

    #run the QC movie
    #qc_notes = [''] * len(nii_files)
    qc_notes = QCmovie(nii_files)

    #create timestamped QC notes file
    out_filename =  '../data_lesson3/example_QC_notes_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'
    with open(out_filename, 'w') as f:
        f.write('Image UID\tQC Note\n')
        for i in range(len(image_ids)):
            f.write('{}\t{}\n'.format(image_ids[i], qc_notes[i]))