import os
import nibabel as nib
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    default="./data/srphs_sh_h",
    type=str,
)


def rename(directory):
    for file in os.listdir(directory):
        curr_patient_mri = directory + f"/{file}/rsfmri"
        for subfile in os.listdir(curr_patient_mri):
            name = subfile.split(".")[0]
            new_name = curr_patient_mri + f"/{name}.nii"
            prev_name = curr_patient_mri + f"/{subfile}"

            os.rename(prev_name, new_name)


def merge_all(directory):
    for file in os.listdir(directory):
        curr_patient_mri = directory + f"/{file}/rsfmri"

        nii_files = curr_patient_mri + "/vol_%00*d.nii"
        images = merge_nii_files(nii_files, range(1, 241))

        save_path = directory + f"/{file}/raw_mri"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        nib.save(images, save_path + "/fmri.nii")


def merge_nii_files(sfile, ns):
    # This will load the first image for header information
    img = nib.load(sfile % (3, ns[0]))
    dshape = list(img.shape)
    dshape.append(len(ns))
    data = np.empty(dshape, dtype=img.get_data_dtype())

    header = img.header
    equal_header_test = True

    # Now load all the rest of the images
    for n, i in enumerate(ns):
        try:
            img = nib.load(sfile % (3, i))
            print(img.shape)
            equal_header_test = equal_header_test and img.header == header
            data[..., n] = np.array(img.dataobj)
        except FileNotFoundError:
            print("HERE, at", n)
            imgs = nib.Nifti1Image(data, img.affine, header=header)
            return imgs

    imgs = nib.Nifti1Image(data, img.affine, header=header)
    if not equal_header_test:
        print("WARNING: Not all headers were equal!")
    return imgs

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.data_path
    rename(data_path)
    merge_all(data_path)
