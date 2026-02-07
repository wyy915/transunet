import os
import numpy as np
import nibabel as nib
import h5py

# 配置
origin_dir = "./origin"
save_dir = "../datasets/knee"
case_name = "case0000"

# 输出目录
train_npz_dir = os.path.join(save_dir, "train_npz")
test_h5_dir = os.path.join(save_dir, "test_vol_h5")
os.makedirs(train_npz_dir, exist_ok=True)
os.makedirs(test_h5_dir, exist_ok=True)
# 读取 NIfTI
ct_path = os.path.join(origin_dir, "case00.nii")
seg_path = os.path.join(origin_dir, "case00.Labels.nii")
ct_nii = nib.load(ct_path).get_fdata()
seg_nii = nib.load(seg_path).get_fdata()

# -------------------------------
# 生成训练用 slice npz
# -------------------------------
num_slices = ct_nii.shape[2]  # z 轴切片
for i in range(num_slices):
    image_slice = ct_nii[:, :, i].astype(np.float32)
    label_slice = seg_nii[:, :, i].astype(np.uint8)
    npz_name = f"{case_name}_slice{i:03d}"
    np.savez_compressed(
        os.path.join(train_npz_dir, npz_name), image=image_slice, label=label_slice
    )

print(f"✔ 已生成 {num_slices} 张训练 slice npz")

# -------------------------------
# 生成测试用完整 volume h5
# -------------------------------
h5_path = os.path.join(test_h5_dir, f"{case_name}")
with h5py.File(h5_path, "w") as f:
    f.create_dataset("image", data=ct_nii.astype(np.float32))
    f.create_dataset("label", data=seg_nii.astype(np.uint8))

print(f"✔ 已生成测试 volume h5: {h5_path}")
