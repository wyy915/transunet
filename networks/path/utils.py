import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
import pyvista as pv
import pandas as pd
import os
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree
from scipy import ndimage
from scipy.linalg import svd
def load_nii_files(image_path, mask_path):
    """加载NII文件并返回图像数据和掩码数据"""
    try:
        image = nib.load(image_path)
        mask = nib.load(mask_path)
        image_data = image.get_fdata()
        mask_data = mask.get_fdata()

        print(f"图像数据形状: {image_data.shape}")
        print(f"掩码数据形状: {mask_data.shape}")

        return image_data, mask_data, image.affine
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None, None, None


def calculate_density_centroids_sagittal(image_data, mask_data, label=3):
    """计算骨骺矢状面上的密度中心（考虑灰度值的加权中心）"""
    if image_data is None or mask_data is None:
        return np.array([]), []

    centroids = []
    valid_slices = []

    print(f"计算密度中心，遍历 {mask_data.shape[0]} 个矢状面切片...")

    for x in range(mask_data.shape[0]):
        try:
            slice_mask = mask_data[x, :, :]
            slice_image = image_data[x, :, :]

            if np.any(slice_mask == label):
                weights = np.where(slice_mask == label, slice_image, 0)

                if np.sum(weights) > 0:
                    y_coords, z_coords = np.mgrid[:slice_mask.shape[0], :slice_mask.shape[1]]
                    total_weight = np.sum(weights)
                    y_center = np.sum(y_coords * weights) / total_weight
                    z_center = np.sum(z_coords * weights) / total_weight

                    centroids.append((x, y_center, z_center))
                    valid_slices.append(x)

        except Exception as e:
            continue

    print(f"找到 {len(centroids)} 个密度中心")
    return np.array(centroids), valid_slices


def calculate_density_centroids_axial(image_data, mask_data, label, trim_ratio):
    """
    计算干骺端横断面 (axial, z 方向) 的密度中心 (灰度加权中心)。
    去掉干骺端范围内的前后若干比例，只保留中间部分。
    """
    if image_data is None or mask_data is None:
        return np.array([]), []

    centroids = []
    valid_slices = []

    # 找到干骺端在 z 方向上的范围
    indices = np.where(mask_data == label)
    if len(indices[0]) == 0:
        print("未找到干骺端区域")
        return np.array([]), []

    z_min, z_max = indices[2].min(), indices[2].max()

    # 在干骺端的范围里去掉两端
    z_len = z_max - z_min + 1
    start_idx = z_min + int(z_len * trim_ratio)
    end_idx   = z_max - int(z_len * trim_ratio)

    for z in range(start_idx, end_idx + 1):
        slice_mask = mask_data[:, :, z]
        slice_image = image_data[:, :, z]

        if np.any(slice_mask == label):
            weights = np.where(slice_mask == label, slice_image, 0)
            if np.sum(weights) > 0:
                x_coords, y_coords = np.mgrid[:slice_mask.shape[0], :slice_mask.shape[1]]
                total_weight = np.sum(weights)
                x_center = np.sum(x_coords * weights) / total_weight
                y_center = np.sum(y_coords * weights) / total_weight

                centroids.append((x_center, y_center, z))
                valid_slices.append(z)

    print(f"找到 {len(centroids)} 个横断面密度中心 (干骺端范围 {z_min}~{z_max}, 去掉 {trim_ratio*100:.0f}% 两端)")
    return np.array(centroids), valid_slices



def transform_to_ras_coordinates(points, affine):
    """
    将体素坐标转换到 RAS 世界坐标。
    支持 points 为 (N,3) 或 (3,)。
    """
    points = np.asarray(points, dtype=float)

    # 统一处理 (3,) -> (1,3)
    if points.ndim == 1:
        points = points.reshape(1, 3)

    if points.shape[0] == 0:
        return np.zeros((0,3), dtype=float)

    # 构造齐次坐标 (N,4)
    ones = np.ones((points.shape[0], 1), dtype=float)
    homogeneous = np.hstack([points, ones])  # (N,4)

    # 应用 affine（行向量乘以 affine.T）
    ras = homogeneous @ affine.T  # (N,4)

    return ras[:, :3].astype(float)



def fit_line_least_squares(centroids):
    """使用最小二乘法拟合直线，返回拟合点、直线中心点、方向向量"""
    if len(centroids) < 2:
        return np.array([]), None, None

    try:
        pca = PCA(n_components=1)
        pca.fit(centroids)
        direction = pca.components_[0]
        centroid = np.mean(centroids, axis=0)

        def residuals(params, points):
            x0, y0, z0, dx, dy, dz = params
            t = ((points[:, 0] - x0) * dx + (points[:, 1] - y0) * dy + (points[:, 2] - z0) * dz) / (
                        dx ** 2 + dy ** 2 + dz ** 2 + 1e-10)
            line_x = x0 + t * dx
            line_y = y0 + t * dy
            line_z = z0 + t * dz
            return np.sqrt((points[:, 0] - line_x) ** 2 + (points[:, 1] - line_y) ** 2 + (points[:, 2] - line_z) ** 2)

        initial_params = np.concatenate([centroid, direction])
        result = least_squares(residuals, initial_params, args=(centroids,))
        x0, y0, z0, dx, dy, dz = result.x

        # 单位化方向
        direction = np.array([dx, dy, dz])
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        line_point = np.array([x0, y0, z0])

        # 生成拟合路径上的点
        t_values = np.linspace(-len(centroids), len(centroids), 100)
        line_points = np.array([
            line_point + t * direction
            for t in t_values
        ])

        return line_points, line_point, direction

    except Exception as e:
        print(f"拟合直线时出错: {e}")
        return np.array([]), None, None


    except Exception as e:
        print(f"拟合直线时出错: {e}")
        return np.array([])


def export_for_slicer(line_points_ras, output_path):
    """导出为3D Slicer可以直接读取的CSV格式"""
    if len(line_points_ras) == 0:
        print("没有路径数据可导出")
        return False

    try:
        # 3D Slicer需要的CSV格式
        df = pd.DataFrame({
            'R': line_points_ras[:, 0],
            'A': line_points_ras[:, 1],
            'S': line_points_ras[:, 2],
            'Label': ['Path_Point'] * len(line_points_ras)
        })

        # 保存为CSV（确保使用正确的编码和格式）
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"路径已导出到: {output_path}")
        print(f"共 {len(line_points_ras)} 个点")
        print(f"RAS坐标范围: R[{df['R'].min():.2f}-{df['R'].max():.2f}], "
              f"A[{df['A'].min():.2f}-{df['A'].max():.2f}], "
              f"S[{df['S'].min():.2f}-{df['S'].max():.2f}]")

        return True

    except Exception as e:
        print(f"导出CSV文件时出错: {e}")
        return False


def export_as_markups_fcsv(line_points_ras, output_path):
    """导出为3D Slicer的Markups Fiducial格式 (.fcsv) - 推荐使用"""
    if len(line_points_ras) == 0:
        return False

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write("# Markups fiducial file version = 4.13\n")
            f.write("# CoordinateSystem = 0\n")
            f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")

            # 写入每个点
            for i, point in enumerate(line_points_ras):
                f.write(f"vtkMRMLMarkupsFiducialNode_{i},")
                f.write(f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f},")  # x,y,z
                f.write("0,0,0,1,")  # orientation (quaternion)
                f.write("1,1,0,")  # visible, selected, locked
                f.write(f"PathPoint_{i},,")  # label, description
                f.write("\n")

        print(f"Markups文件已导出到: {output_path}")
        return True

    except Exception as e:
        print(f"导出Markups文件时出错: {e}")
        return False

def extract_growth_plate_midpoints(mask_data, label_epi, label_meta, dilation_iter, dist_thresh):
    """
    提取骺板中层点：膨胀后的骨骺和干骺端表面之间的点对，如果距离小于某个值，就取他们的中点
    """
    struct = ndimage.generate_binary_structure(3, 1)

    # 膨胀后掩码
    epi_mask = (mask_data == label_epi)
    meta_mask = (mask_data == label_meta)
    epi_dil = ndimage.binary_dilation(epi_mask, structure=struct, iterations=dilation_iter)  #dilation_iter:膨胀的层数
    meta_dil = ndimage.binary_dilation(meta_mask, structure=struct, iterations=dilation_iter)

    # 提取表面点
    epi_surface = epi_dil & ~ndimage.binary_erosion(epi_dil, structure=struct)
    meta_surface = meta_dil & ~ndimage.binary_erosion(meta_dil, structure=struct)
    #生成点云
    epi_points = np.column_stack(np.nonzero(epi_surface))
    meta_points = np.column_stack(np.nonzero(meta_surface))

    if len(epi_points) == 0 or len(meta_points) == 0:
        print("未找到足够的表面点")
        return np.array([])

    # KDTree 最近邻搜索：找到距离在范围内的点对
    tree = cKDTree(meta_points)
    dist, idx = tree.query(epi_points, k=1)

    # 筛选在阈值内的点对
    mask = dist <= dist_thresh
    epi_close = epi_points[mask]
    meta_close = meta_points[idx[mask]]

    # 中点
    midpoints = (epi_close + meta_close) / 2.0
    return midpoints

# ---------- 在fit_plate_plane函数中调用，生成平面网格角点并导出 fcsv ----------
def generate_plane_corners_and_export(centroid_voxel, u_vec, v_vec, points_voxel, affine,
                                      fcsv_path_corners, fcsv_path_dense=None, scale_factor=1.2, grid_samples=(5,5)):
    """
    根据点云在 u/v 方向的投影范围生成一个矩形平面：
      - centroid_voxel: 平面中心 (voxel)
      - u_vec, v_vec: 基向量（voxel 单位方向）
      - points_voxel: 原始交界点云 (用于估算范围)
    输出:
      - 一个包含四个角的 .fcsv (可导入 Slicer)
      - 可选：导出平面上网格点的 .fcsv（更密集，便于可视化平面）
    """
    # 在 u/v 方向上对交界点投影，求范围
    proj_u = np.dot(points_voxel - centroid_voxel, u_vec)
    proj_v = np.dot(points_voxel - centroid_voxel, v_vec)
    u_min, u_max = proj_u.min(), proj_u.max()
    v_min, v_max = proj_v.min(), proj_v.max()

    # 使矩形略大一些
    half_u = (u_max - u_min) / 2.0 * scale_factor
    half_v = (v_max - v_min) / 2.0 * scale_factor

    # 四个角的 voxel 坐标
    corners = np.array([
        centroid_voxel +  (+half_u)*u_vec + (+half_v)*v_vec,
        centroid_voxel +  (+half_u)*u_vec + (-half_v)*v_vec,
        centroid_voxel +  (-half_u)*u_vec + (-half_v)*v_vec,
        centroid_voxel +  (-half_u)*u_vec + (+half_v)*v_vec,
    ])  # shape (4,3)

    # 把角点及密集网格转换到 RAS（如果 affine 提供）
    if affine is not None:
        # 转换函数
        from nibabel.affines import apply_affine
        corners_ras = apply_affine(affine, corners)
    else:
        corners_ras = corners.copy()

    # 导出四角 fcsv
    with open(fcsv_path_corners, 'w', encoding='utf-8') as f:
        f.write("# Markups fiducial file version = 4.13\n")
        f.write("# CoordinateSystem = 0\n")  # 0: RAS in some Slicer versions; 请按需调整
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, p in enumerate(corners_ras):
            f.write(f"vtkMRMLMarkupsFiducialNode_{i},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},0,0,0,1,1,1,0,PlaneCorner_{i},,\n")

    # 可选：导出平面上的密集点云
    if fcsv_path_dense is not None:
        u_lin = np.linspace(-half_u, +half_u, grid_samples[0])
        v_lin = np.linspace(-half_v, +half_v, grid_samples[1])
        grid_pts = []
        for uu in u_lin:
            for vv in v_lin:
                pt = centroid_voxel + uu*u_vec + vv*v_vec
                grid_pts.append(pt)
        grid_pts = np.array(grid_pts)
        if affine is not None:
            from nibabel.affines import apply_affine
            grid_ras = apply_affine(affine, grid_pts)
        else:
            grid_ras = grid_pts.copy()

        with open(fcsv_path_dense, 'w', encoding='utf-8') as f:
            f.write("# Markups fiducial file version = 4.13\n")
            f.write("# CoordinateSystem = 0\n")
            f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
            for i, p in enumerate(grid_ras):
                f.write(f"vtkMRMLMarkupsFiducialNode_{i},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},0,0,0,1,1,1,0,PlanePt_{i},,\n")

    return corners, grid_pts

# ----------在fit_plate_plane函数中调用， 对之前计算出的中点平面拟合（SVD） ----------
def fit_plane_svd(points_voxel, affine=None):
    """
    对输入点拟合平面 (voxel coordinates).
    返回：
      - centroid_voxel (3,)
      - normal (3,)  (单位向量，指向某侧)
      - u_vec, v_vec (两个平面内基向量，单位正交)
      - if affine provided: also returns corresponding RAS coords: centroid_ras, corners_ras...
    """
    if points_voxel.size == 0:
        raise ValueError("没有输入点用于拟合平面")

    centroid = points_voxel.mean(axis=0)
    pts_centered = points_voxel - centroid
    # SVD
    U, S, Vt = svd(pts_centered, full_matrices=False)
    # Vt 行为主成分方向： Vt[0] 最大方差方向，Vt[2] 最小方差方向 => 法向量
    u_vec = Vt[0]
    v_vec = Vt[1]
    normal = Vt[2]
    # 确保右手坐标系（可选）
    # normal = np.cross(u_vec, v_vec)
    return centroid, normal / (np.linalg.norm(normal) + 1e-12), u_vec / (np.linalg.norm(u_vec)+1e-12), v_vec / (np.linalg.norm(v_vec)+1e-12)

def fit_growth_plate_plane(midpoints_voxel, mask_data, affine,
                                  label_epi=3, output_dir="./slicer_export",
                                  split_axis="u"):
    """
    将骺板分为内外两部分（沿拟合平面的u或v方向），并分别拟合平面导出。
    参数:
      - split_axis: "u" 或 "v"，决定沿哪个方向划分内外骺板。
    返回:
      - dict(inner=..., outer=...)，每个包含(grid, centroid, normal, u_vec, v_vec)
    """
    if len(midpoints_voxel) < 10:
        print("中点不足，无法拟合平面")
        return None

    # ---------- 1. 拟合整体平面 ----------
    centroid, normal, u_vec, v_vec = fit_plane_svd(midpoints_voxel)

    # ---------- 2. 按 u 或 v 方向划分 ----------
    axis_vec = u_vec if split_axis == "u" else v_vec
    proj = np.dot(midpoints_voxel - centroid, axis_vec)
    median_val = np.median(proj)
    inner_pts = midpoints_voxel[proj < median_val]
    outer_pts = midpoints_voxel[proj >= median_val]

    print(f"内骺板点数: {len(inner_pts)}, 外骺板点数: {len(outer_pts)}")

    # ---------- 3. 取骨骺点云 ----------
    epi_points = np.column_stack(np.nonzero(mask_data == label_epi))
    if len(epi_points) == 0:
        print("未找到骨骺点")
        return None

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, pts in zip(["inner", "outer"], [inner_pts, outer_pts]):
        if len(pts) < 10:
            print(f"{name} 骺板点不足，跳过")
            continue

        # 再次拟合子平面
        c, n, u, v = fit_plane_svd(pts)
        corners_fcsv = os.path.join(output_dir, f"growth_plate_{name}_corners.fcsv")
        grid_fcsv = os.path.join(output_dir, f"growth_plate_{name}_grid.fcsv")

        corners_voxel, grid_voxel = generate_plane_corners_and_export(
            c, u, v, epi_points, affine,
            corners_fcsv, fcsv_path_dense=grid_fcsv,
            scale_factor=1.2, grid_samples=(21, 21)
        )

        results[name] = dict(
            grid=grid_voxel,
            centroid=c,
            normal=n,
            u_vec=u,
            v_vec=v,
            corners=corners_voxel
        )

        print(f"{name} 骺板平面已导出：{corners_fcsv}")

    return results


def mirror_points_across_plane(results, points, output_dir="./slicer_export", affine=None):
    """
    对 inner / outer 两个平面分别执行镜像操作
    参数:
      - results: fit_growth_plate_plane() 的返回值（包含 inner / outer）
      - points: (N,3) 要镜像的路径
      - output_dir: 导出目录
      - affine: 若提供，将结果点转换到 RAS 坐标输出
    输出:
      - dict(inner_mirrored, outer_mirrored)，每个为镜像点数组
    """
    mirrored_results = {}
    for name in ["inner", "outer"]:
        plane = results.get(name)
        if plane is None or "normal" not in plane:
            print(f"{name} 平面信息缺失，跳过镜像")
            continue

        centroid = plane["centroid"]
        normal = plane["normal"]

        # --- 镜像 ---
        v = points - centroid
        d = np.dot(v, normal)
        mirrored = points - 2 * d[:, np.newaxis] * normal

        # --- 转到 RAS 坐标（可选） ---
        if affine is not None:
            mirrored_ras = apply_affine(affine, mirrored)
        else:
            mirrored_ras = mirrored

        mirrored_results[name] = mirrored_ras

        # --- 导出 fcsv ---
    return mirrored_results


def move_path_to_line_in_plane(mirrored_path, line_point, line_dir, plane_normal, ref="centroid"):
    """
    沿平面平移路径，使路径与直线相交，参考点为直线与平面的交点
    """
    line_dir = line_dir / np.linalg.norm(line_dir)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # 方法1：使用路径上的特定点计算平面
    if ref == "start":
        p_ref = mirrored_path[0]
    elif ref == "end":
        p_ref = mirrored_path[-1]
    else:  # centroid
        p_ref = np.mean(mirrored_path, axis=0)

    # 创建通过参考点且法向量为plane_normal的平面
    # 平面方程: n·(x - p_ref) = 0

    # 计算直线与平面的交点
    denom = np.dot(plane_normal, line_dir)
    if abs(denom) < 1e-6:
        raise ValueError("直线与平面平行，没有交点")

    # 计算参数t: t = n·(p_ref - line_point) / (n·line_dir)
    t = np.dot(plane_normal, p_ref - line_point) / denom
    intersection_point = line_point + t * line_dir

    print(f"参考点: {p_ref}")
    print(f"交点: {intersection_point}")
    print(f"平移向量: {intersection_point - p_ref}")

    # 计算平面内平移向量（去除法线方向分量）
    T = intersection_point - p_ref
    T_plane = T - np.dot(T, plane_normal) * plane_normal

    # 平移路径
    moved_path = mirrored_path + T_plane

    # 验证：平移后的参考点应该在直线上
    if ref == "start":
        moved_ref = moved_path[0]
    elif ref == "end":
        moved_ref = moved_path[-1]
    else:
        moved_ref = np.mean(moved_path, axis=0)

    # 计算平移后参考点到直线的距离
    v = moved_ref - line_point
    projection = np.dot(v, line_dir)
    closest_point = line_point + projection * line_dir
    distance = np.linalg.norm(moved_ref - closest_point)
    print(f"验证距离: {distance}")

    if distance > 1e-3:
        print(f"警告：平移后参考点与直线距离为 {distance:.6f}")

    return moved_path


def visualize_in_3d(image_data, mask_data,
                    centroids_voxel1,
                    line_points_voxel1,
                    centroids_voxel2,
                    line_points_voxel2,
                    move_path1,
                    move_path2,
                    label_epi=3,
                    label_meta=2,
                    plane_points_voxel1=None,
                    plane_points_voxel2=None,
                    mirror_path1=None,
                    mirror_path2=None):
    """在3D窗口中可视化骨骺、干骺端、拟合路径、骺板平面、镜像路径"""
    try:
        plotter = pv.Plotter()

        # ---------- Epiphysis (骨骺) ----------
        indices = np.where(mask_data == label_epi)
        if len(indices[0]) > 0:
            epi_points = np.column_stack(indices)
            if len(epi_points) > 10000:
                sample_indices = np.random.choice(len(epi_points), 10000, replace=False)
                epi_points = epi_points[sample_indices]
            plotter.add_mesh(pv.PolyData(epi_points),
                             color='lightblue',
                             point_size=2,
                             opacity=0.3,
                             label='Epiphysis')

        # ---------- Metaphysis (干骺端) ----------
        indices = np.where(mask_data == label_meta)
        if len(indices[0]) > 0:
            meta_points = np.column_stack(indices)
            if len(meta_points) > 10000:
                sample_indices = np.random.choice(len(meta_points), 10000, replace=False)
                meta_points = meta_points[sample_indices]
            plotter.add_mesh(pv.PolyData(meta_points),
                             color='green',
                             point_size=2,
                             opacity=0.3,
                             label='Metaphysis')

        # ---------- 骨骺密度中心 ----------
        if len(centroids_voxel1) > 0:
            plotter.add_mesh(pv.PolyData(centroids_voxel1),
                             color='red',
                             point_size=6,
                             render_points_as_spheres=True,
                             label='Epiphysis Density Centers')

        # ---------- 骨骺拟合路径 ----------
        if len(line_points_voxel1) > 0:
            line = pv.lines_from_points(line_points_voxel1)
            plotter.add_mesh(line,
                             color='blue',
                             line_width=5,
                             label='Epiphysis Fitted Path')
        # ---------- 干骺端密度中心 ----------
        if len(centroids_voxel2) > 0:
            plotter.add_mesh(pv.PolyData(centroids_voxel2),
                             color='red',
                             point_size=6,
                             render_points_as_spheres=True,
                             label='metaphysis Density Centers')

        # ---------- 干骺端拟合路径 ----------
        if len(line_points_voxel2) > 0:
            line = pv.lines_from_points(line_points_voxel2)
            plotter.add_mesh(line,
                             color='black',
                             line_width=5,
                             label='metaphysis Fitted Path')

        # ---------- Growth plate plane (骺板平面) ----------
        if plane_points_voxel1 is not None and len(plane_points_voxel1) > 0:
            # 构造网格
            plane = pv.PolyData(plane_points_voxel1).delaunay_2d()
            plotter.add_mesh(plane,
                             color='yellow',
                             opacity=0.4,
                             label='inner Growth Plate Plane')
        if plane_points_voxel2 is not None and len(plane_points_voxel2) > 0:
            # 构造网格
            plane = pv.PolyData(plane_points_voxel2).delaunay_2d()
            plotter.add_mesh(plane,
                             color='blue',
                             opacity=0.4,
                             label='outer Growth Plate Plane')
        # ---------- 镜像路径 ----------
        if len(mirror_path1) > 0:
            line = pv.lines_from_points(mirror_path1)
            plotter.add_mesh(line,
                             color='green',
                             line_width=5,
                             label='inner Mirrored Path')
        if len(mirror_path2) > 0:
            line = pv.lines_from_points(mirror_path2)
            plotter.add_mesh(line,
                             color='green',
                             line_width=5,
                             label='outer Mirrored Path')
        # ---------- 移动镜像路径 ----------
        if len(mirror_path1) > 0:
            line = pv.lines_from_points(move_path1)
            plotter.add_mesh(line,
                             color='red',
                             line_width=5,
                             label='inner moved Mirrored Path')
        if len(mirror_path2) > 0:
            line = pv.lines_from_points(move_path2)
            plotter.add_mesh(line,
                             color='red',
                             line_width=5,
                             label='outer moved Mirrored Path')

        # ---------- 显示设置 ----------
        plotter.add_axes()
        plotter.add_legend()
        plotter.add_title('Knee Analysis\n(Voxel Coordinates)')
        plotter.show()

    except Exception as e:
        print(f"3D可视化时出错: {e}")

