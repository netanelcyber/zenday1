from glob import glob

import os



def get_openneuro_data(base_dir, label):

    file_paths = []

    labels = []



    for func_file in glob(os.path.join(base_dir, "sub-*", "func", "*.nii.gz")):

        file_paths.append(func_file)

        labels.append(label)



    return file_paths, labels



# Change these to your actual paths after download

abide_paths, abide_labels = get_openneuro_data("ds000228", label=1)  # ASD

cobre_paths, cobre_labels = get_openneuro_data("ds000030", label=0)  # SCZ



file_paths = cobre_paths + abide_paths

labels = cobre_labels + abide_labels


