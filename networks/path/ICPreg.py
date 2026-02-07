import numpy as np
import open3d as o3d
from scipy import ndimage
from skimage import measure
import nibabel as nib
import cv2
import os
class SurgicalRegistration:
    def __init__(self):
        self.transformation = None
        self.registration_error = None

    def load_nifti_mask(mask_path):
        """加载 NIfTI 文件并返回 numpy array"""
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)
        nii = nib.load(mask_path)
        data = nii.get_fdata()
        return data, nii.affine if hasattr(nii, "affine") else None

    def preprocess_mask(mask, threshold=0.5, smooth_sigma=None):
        """
        将 mask 或密度图转为二值（0/1）
        - mask: numpy array (3D)
        - threshold: 阈值。对于 segmentation mask 通常 0.5；对于 CT Hounsfield 可能设置为 300 等
        - smooth_sigma: 如果不为 None，先做 Gaussian 平滑（避免 marching_cubes 的噪点）
        """
        if smooth_sigma is not None and smooth_sigma > 0:
            mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=smooth_sigma)
        binary = (mask >= threshold).astype(np.uint8)
        return binary

    def extract_surface_marching_cubes(binary_mask, spacing=(1.0, 1.0, 1.0)):
        """
        使用 scikit-image 的 marching_cubes 提取等值面
        返回 vertices (N,3) 和 faces (M,3)
        spacing: voxel size (z,y,x) 或 (x,y,z) 取决于数据——我们使用 (z,y,x) -> 请确保传入正确顺序
        注意：skimage.measure.marching_cubes 的 spacing 参数通常是 (sx, sy, sz) 对应 array 的轴顺序。
        """
        # skimage 的 marching_cubes 要求输入为连续数据；二值图也可以
        # level 为 0.5 (二值图的分界)
        verts, faces, normals, values = measure.marching_cubes(binary_mask, level=0.5, spacing=spacing,
                                                               allow_degenerate=False)
        return verts, faces

def mesh_to_pointcloud(verts, faces=None, sample_points=None):
    """
    将 mesh 顶点/三角面片转换为 Open3D 点云
    - 如果 sample_points 指定，按三角面积采样点（返回点云）；否则直接使用顶点作为点云
    """
    if sample_points is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        return pcd
    else:
        # 若需要从 mesh 中采样更多点（均匀采样）
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        return pcd

def visualize_surface_open3d(mask_path,
                             label_epi=3,
                             label_meta=4,
                             voxel_spacing=(1.0, 1.0, 1.0)):
    """使用 Open3D 可视化 MC 表面点云"""

    reg = SurgicalRegistration()

    # 加载 mask
    print("正在加载 mask...")
    mask_data = reg.load_mask_data(mask_path)
    if mask_data is None:
        return None

    print(f"mask 唯一值: {np.unique(mask_data)}")

    # 提取表面
    if mask_data.dtype == bool or len(np.unique(mask_data)) <= 2:
        print("检测到二值mask，提取整体表面")
        epi_surface = reg.extract_surface_from_mask(mask_data, 1, voxel_spacing)
        meta_surface = None
    else:
        print("检测到多标签mask，分别提取 surface")
        epi_surface = reg.extract_surface_from_mask(mask_data, label_epi, voxel_spacing)
        meta_surface = reg.extract_surface_from_mask(mask_data, label_meta, voxel_spacing)

    geometries = []

    # Epiphysis
    if epi_surface is not None:
        pcd_epi = o3d.geometry.PointCloud()
        pcd_epi.points = o3d.utility.Vector3dVector(epi_surface)
        pcd_epi.paint_uniform_color([0.4, 0.8, 1.0])  # light blue
        geometries.append(pcd_epi)

    # Metaphysis
    if meta_surface is not None:
        pcd_meta = o3d.geometry.PointCloud()
        pcd_meta.points = o3d.utility.Vector3dVector(meta_surface)
        pcd_meta.paint_uniform_color([0.6, 1.0, 0.6])  # green
        geometries.append(pcd_meta)

    if len(geometries) == 0:
        print("未提取到任何表面")
        return None

    # Open3D 可视化
    o3d.visualization.draw_geometries(
        geometries,
        window_name="MC Surface (Open3D版)",
        width=900,
        height=700
    )

    return {
        "epi_surface": epi_surface,
        "meta_surface": meta_surface,
        "mask_data": mask_data
    }
# -------------------------
# 2) 载入 NDI 点 (CSV/ TXT)
# -------------------------
def load_ndi_points_from_csv(csv_path, delimiter=','):
    """
    期望 CSV 文件列为: x,y,z (可有头行)
    返回 Open3D 点云
    """
    arr = np.loadtxt(csv_path, delimiter=delimiter)
    # 若文件有 header 或单列异常，请手动处理
    if arr.ndim == 1:
        raise ValueError("NDI 文件似乎不是 3 列坐标，请检查格式")
    if arr.shape[1] < 3:
        raise ValueError("NDI 文件列数小于 3")
    points = arr[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd
# -------------------------
# 3) 预处理：下采样、估计法线、FPFH 特征
# -------------------------
#
def preprocess_point_cloud(pcd, voxel_size):
    """
    下采样、估法线、计算 FPFH
    返回 (pcd_down, fpfh)
    """
    print(f"Downsampling with voxel size = {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.0
    print(f"Estimating normals with radius = {radius_normal}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5.0
    print(f"Computing FPFH with radius = {radius_feature}")
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, fpfh

# -------------------------
# 4) 粗配准：RANSAC based on FPFH
#FPFH 特征匹配:Open3D 会对 source 与 target 的每个点寻找特征空间中最近邻点。
#在不知道两个点云如何对齐的前提下，找到一个初始的、足够接近的刚性变换，使得后续 ICP 可以收敛到正确结果。
# -------------------------
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(f"RANSAC 初配准，distance_threshold = {distance_threshold}")
    #随机挑选4对匹配点
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result
# -------------------------
# 5) 精配准：点到面 ICP (point-to-plane)
# -------------------------
def refine_registration_icp(source, target, init_transformation, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(f"ICP 精配准，distance_threshold = {distance_threshold}")
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result
# -------------------------
# 6) 误差评估
# -------------------------
def compute_rmse(source, target, transformation, max_correspondence_distance=np.inf):
    """
    计算点到点平均误差（使用最近邻）
    """
    src = source.transform(transformation)
    # 需要用 KDTree 对 target 建树
    tgt_kd = o3d.geometry.KDTreeFlann(target)
    src_pts = np.asarray(src.points)
    dists = []
    for p in src_pts:
        [k, idx, dist2] = tgt_kd.search_knn_vector_3d(p, 1)
        if k > 0:
            dists.append(dist2[0])
    if len(dists) == 0:
        return float('inf')
    rmse = np.sqrt(np.mean(dists))
    return rmse

# -------------------------
# 7) 可视化配准前后
# -------------------------
def visualize_registration(source, target, transformation=np.eye(4), window_name="Registration"):
    src_temp = source.copy()
    tgt_temp = target.copy()
    src_temp.paint_uniform_color([1, 0.706, 0])  # orange
    tgt_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    src_temp.transform(transformation)
    o3d.visualization.draw_geometries([src_temp, tgt_temp], window_name=window_name)


# -------------------------
# 8) 主流程：CT -> extract -> reg with NDI
# -------------------------
def run_registration_pipeline(
    ct_mask_path,
    ndi_points_path,
    voxel_size=1.0,
    ct_threshold=0.5,
    ct_smooth_sigma=None,
    sample_points_from_mesh=200000,
    ndi_delimiter=',',
    save_temp_ply=None
):
    """
    完整流水线
    - ct_mask_path: NIfTI file (mask or CT density)
    - ndi_points_path: CSV/TXT with x,y,z (NDI 点)
    - voxel_size: 下采样体素大小（和数据尺度相关，单位与 voxel_spacing 一致）
    - ct_threshold: 二值阈值，用于 mask（若是原始 CT，请先用阈值选择骨）
    - sample_points_from_mesh: 从 mesh 采样多少点（None 则使用顶点）
    - save_temp_ply: 如果指定，将保存 CT 提取的点云到 ply，便于调试
    """
    # 1) load ct
    print("加载 CT/Mask...")
    reg=SurgicalRegistration()
    mask, affine=reg.load_nifti_mask(mask_path)
    print(f"mask shape = {mask.shape}, affine = {affine is not None}")

    # 2) preprocess mask -> binary
    binary = reg.preprocess_mask(mask, threshold=ct_threshold, smooth_sigma=ct_smooth_sigma)
    print(f"binary sum = {binary.sum()}")

    # 3) marching cubes
    # NOTE: spacing 通常基于 nifti header 的 voxel sizes；如果你不知道，可传 (1,1,1) 或从 header 读取
    # nibabel 的 header: nii.header.get_zooms() -> (x_spacing, y_spacing, z_spacing)
    try:
        import nibabel as nib2
        nii = nib2.load(mask_path)
        zooms = nii.header.get_zooms()
        # skimage expects spacing in same axis order as array (z,y,x) or (x,y,z) depending - careful
        # We will pass spacing = (zooms[2], zooms[1], zooms[0]) if marching_cubes expects z,y,x ordering
        spacing = zooms  # user should verify; common case: (x, y, z)
        print(f"Voxel spacing from header: {spacing}")
    except Exception:
        spacing = (1.0, 1.0, 1.0)
        print("无法读取 header spacing，使用默认 spacing = (1,1,1)")

    verts, faces = reg.extract_surface_marching_cubes(binary, spacing=spacing)
    print(f"Extracted mesh: vertices={len(verts)}, faces={len(faces)}")

    # 4) mesh -> point cloud (表面点云采样)
    if sample_points_from_mesh is None:
        pcd_ct = mesh_to_pointcloud(verts, faces, sample_points=None)
    else:
        pcd_ct = mesh_to_pointcloud(verts, faces, sample_points=sample_points_from_mesh)
    print(f"CT point cloud points: {len(pcd_ct.points)}")

    if save_temp_ply is not None:
        o3d.io.write_point_cloud(save_temp_ply, pcd_ct)
        print(f"Saved CT point cloud to {save_temp_ply}")

    # 5) load NDI points——用探针采集的点
    pcd_ndi = load_ndi_points_from_csv(ndi_points_path, delimiter=ndi_delimiter)
    print(f"NDI point cloud points: {len(pcd_ndi.points)}")

    # 6) preprocess both
    print("预处理点云（下采样、估法线、FPFH）")
    source_down, source_fpfh = preprocess_point_cloud(pcd_ct, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_ndi, voxel_size)

    # 7) coarse registration (RANSAC)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("RANSAC result:")
    print(result_ransac)
    print("RANSAC transformation:\n", result_ransac.transformation)

    # visualize coarse
    visualize_registration(pcd_ct, pcd_ndi, result_ransac.transformation, window_name="Coarse Registration (RANSAC)")

    # 8) refine using ICP (point-to-plane)
    result_icp = refine_registration_icp(pcd_ct, pcd_ndi, result_ransac.transformation, voxel_size)
    print("ICP result:")
    print(result_icp)
    print("ICP transformation:\n", result_icp.transformation)

    # 9) compute RMSE
    rmse_before = compute_rmse(pcd_ct, pcd_ndi, np.eye(4))
    rmse_after = compute_rmse(pcd_ct, pcd_ndi, result_icp.transformation)
    print(f"RMSE before: {rmse_before:.4f}, RMSE after: {rmse_after:.4f}")

    # 10) visualize refined
    visualize_registration(pcd_ct, pcd_ndi, result_icp.transformation, window_name="Refined Registration (ICP)")

    return {
        "ransac": result_ransac,
        "icp": result_icp,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "transformation": result_icp.transformation,
        "pcd_ct": pcd_ct,
        "pcd_ndi": pcd_ndi
    }



if __name__ == "__main__":
    image_path = "../../data/origin/TEDIEGW3.nii"
    mask_path = "../../data/mask/TEDIEGW3.Labels.nii"

    ndi_points_path = "path/to/ndi_points.csv"  # CSV: x,y,z
    # 核心参数（需要根据你的数据尺度调整）
    voxel_size = 2.0  # 下采样大小（单位与 CT spacing 一致），对 RANSAC/ICP 敏感
    ct_threshold = 0.5  # 如果是 segmentation mask 用 0.5；若原始 CT，请用 HU 阈值 (如 300)
    ct_smooth_sigma = 1.0  # 可选，平滑以去噪
    sample_points = 200000  # 从 mesh 上采样点数（None 则使用顶点）
    res = run_registration_pipeline(
        ct_mask_path=mask_path,
        ndi_points_path=ndi_points_path,
        voxel_size=voxel_size,
        ct_threshold=ct_threshold,
        ct_smooth_sigma=ct_smooth_sigma,
        sample_points_from_mesh=sample_points,
        save_temp_ply="ct_extracted.ply"
    )

    print("Final transformation:\n", res["transformation"])
    print(f"Final RMSE: {res['rmse_after']:.4f}")



