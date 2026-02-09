import os
import numpy as np
import nibabel as nib
import h5py

origin_dir = "./origin"
list_dir = "../../lists/lists_knee"

train_dir = "train_npz"
test_dir = "test_vol_h5"
# prepare train and test data,change train data into npz format
with open(os.path.join(list_dir, "train.txt")) as f:
    train_cases = [line.strip().replace(".npy.h5", "") for line in f]

for case in train_cases:
    ct_path = os.path.join(origin_dir, case, "case00.nii")
    seg_path = os.path.join(origin_dir, case, "case00.label.nii")

    ct_nii = nib.load(ct_path).get_fdata()
    seg_nii = nib.load(seg_path).get_fdata()

    num_slices = ct_nii.shape[2]

    for i in range(num_slices):
        image_slice = ct_nii[:, :, i].astype(np.float32)
        label_slice = seg_nii[:, :, i].astype(np.uint8)

        npz_name = f"{case}_slice{i:03d}.npz"
        np.savez_compressed(
            os.path.join(train_dir, npz_name), image=image_slice, label=label_slice
        )

    print(f"✔ train case {case}: {num_slices} slices")

with open(os.path.join(list_dir, "test_vol.txt")) as f:
    test_cases = [line.strip().replace(".npy.h5", "") for line in f]

for case in test_cases:
    ct_path = os.path.join(origin_dir, case, "case00.nii")
    seg_path = os.path.join(origin_dir, case, "case00.label.nii")

    ct_nii = nib.load(ct_path).get_fdata()
    seg_nii = nib.load(seg_path).get_fdata()

    h5_path = os.path.join(test_dir, f"{case}.npy.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("image", data=ct_nii.astype(np.float32))
        f.create_dataset("label", data=seg_nii.astype(np.uint8))

    print(f"✔ test case {case}: saved volume h5")
