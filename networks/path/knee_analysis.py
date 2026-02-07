from .utils import *


class KneePathAnalysis:
    def __init__(self):
        self.image_data = None
        self.mask_data = None
        self.affine = None
        self.results = {}

    def run_full_analysis(self, image_path, mask_path, output_dir="."):
        """主函数"""
        print("=== Knee Epiphysis Path Analysis ===")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        image_data, mask_data, affine = load_nii_files(image_path, mask_path)
        R_affine = np.asarray(affine, dtype=float)[:3, :3]

        if image_data is None or mask_data is None:
            return
        # 计算骨骺密度中心
        lb3_centroids_voxel, lb3_valid_slices = calculate_density_centroids_sagittal(
            image_data, mask_data, label=3
        )
        # 拟合骨骺进钉路径
        lb3_points_voxel, lb3_line_point, lb3_line_dir = fit_line_least_squares(
            lb3_centroids_voxel
        )
        # 转换为RAS坐标
        lb3_points_ras = transform_to_ras_coordinates(lb3_points_voxel, affine)

        # 计算干骺端密度中心
        lb2_centroids_voxel, lb2_valid_slices = calculate_density_centroids_axial(
            image_data, mask_data, label=4, trim_ratio=0.25
        )
        # 拟合路径
        lb2_points_voxel, lb2_line_point, lb2_line_dir = fit_line_least_squares(
            lb2_centroids_voxel
        )
        # 转换为RAS坐标
        lb2_line_point_ras = transform_to_ras_coordinates(
            lb2_line_point[np.newaxis, :], affine
        )[0]
        lb2_line_dir = np.asarray(lb2_line_dir, dtype=float).reshape(
            3,
        )
        lb2_line_dir_ras = R_affine.dot(lb2_line_dir)
        lb2_line_dir_ras = lb2_line_dir_ras / (np.linalg.norm(lb2_line_dir_ras) + 1e-12)

        # 导出两种格式（推荐使用.fcsv格式）
        csv_path = os.path.join(output_dir, "Epiphysis_path.csv")
        fcsv_path = os.path.join(output_dir, "fitted_Epiphysis_path.fcsv")
        export_for_slicer(lb3_points_ras, csv_path)
        export_as_markups_fcsv(lb3_points_ras, fcsv_path)
        print("骨骺路径已导出")

        csv_path = os.path.join(output_dir, "Metaphysis_path.csv")
        fcsv_path = os.path.join(output_dir, "fitted_Metaphysis_path.fcsv")
        export_for_slicer(lb2_line_point_ras, csv_path)
        export_as_markups_fcsv(lb2_line_point_ras, fcsv_path)
        print("干骺端路径已导出")

        # 提取中点并拟合平面
        midpoints = extract_growth_plate_midpoints(
            mask_data, label_epi=3, label_meta=4, dilation_iter=3, dist_thresh=7.0
        )
        # 拟合平面并导出 fcsv
        results = fit_growth_plate_plane(
            midpoints, mask_data, affine, label_epi=3, output_dir="./slicer_export"
        )
        inner_plane = results["inner"]
        outer_plane = results["outer"]

        # plane centroids / normals -> RAS
        inner_centroid_ras = transform_to_ras_coordinates(
            np.asarray(results["inner"]["centroid"], dtype=float).reshape(1, 3), affine
        )[0]
        outer_centroid_ras = transform_to_ras_coordinates(
            np.asarray(results["outer"]["centroid"], dtype=float).reshape(1, 3), affine
        )[0]

        inner_normal_vox = np.asarray(results["inner"]["normal"], dtype=float).reshape(
            3,
        )
        outer_normal_vox = np.asarray(results["outer"]["normal"], dtype=float).reshape(
            3,
        )

        inner_normal_ras = R_affine.dot(inner_normal_vox)
        inner_normal_ras = inner_normal_ras / (np.linalg.norm(inner_normal_ras) + 1e-12)
        outer_normal_ras = R_affine.dot(outer_normal_vox)
        outer_normal_ras = outer_normal_ras / (np.linalg.norm(outer_normal_ras) + 1e-12)

        mirrored_paths = mirror_points_across_plane(
            results, lb3_points_voxel, output_dir="./slicer_export", affine=None
        )
        inner_mirrored = mirrored_paths["inner"]
        outer_mirrored = mirrored_paths["outer"]
        inner_mirrored_ras = (
            transform_to_ras_coordinates(inner_mirrored, affine)
            if inner_mirrored is not None
            else None
        )
        outer_mirrored_ras = (
            transform_to_ras_coordinates(outer_mirrored, affine)
            if outer_mirrored is not None
            else None
        )

        # 导出 inner
        fcsv_path = os.path.join(output_dir, "inner_mirrored.fcsv")
        export_as_markups_fcsv(inner_mirrored_ras, fcsv_path)
        # 导出 outer
        fcsv_path = os.path.join(output_dir, "outer_mirrored.fcsv")
        export_as_markups_fcsv(outer_mirrored_ras, fcsv_path)
        print("镜像路径已导出")

        # 移动镜像路径
        inner_path = mirrored_paths["inner"]
        outer_path = mirrored_paths["outer"]
        inner_moved_path = move_path_to_line_in_plane(
            inner_path,
            lb2_line_point,
            lb2_line_dir,  # 中心轴线
            inner_normal_vox,
            ref="cen",
        )
        outer_moved_path = move_path_to_line_in_plane(
            outer_path,
            lb2_line_point,
            lb2_line_dir,  # 中心轴线
            outer_normal_vox,
            ref="cen",
        )
        inner_moved_ras = (
            transform_to_ras_coordinates(inner_moved_path, affine)
            if inner_mirrored is not None
            else None
        )
        outer_moved_ras = (
            transform_to_ras_coordinates(outer_moved_path, affine)
            if inner_mirrored is not None
            else None
        )

        fcsv_path = os.path.join(output_dir, "inner_moved_path.fcsv")
        export_as_markups_fcsv(inner_moved_ras, fcsv_path)
        fcsv_path = os.path.join(output_dir, "outer_moved_path.fcsv")
        export_as_markups_fcsv(outer_moved_ras, fcsv_path)
        print("移动后的路径已导出")

        # 在voxel坐标系可视化,注意是体素坐标系，RAS坐标系可以在3d slicer中验证
        visualize_in_3d(
            image_data,
            mask_data,
            lb3_centroids_voxel,
            lb3_points_voxel,
            lb2_centroids_voxel,
            lb2_points_voxel,
            inner_moved_path,
            outer_moved_path,
            label_epi=3,
            label_meta=4,
            plane_points_voxel1=results["inner"]["corners"],
            plane_points_voxel2=results["outer"]["corners"],
            mirror_path1=inner_path,
            mirror_path2=outer_path,
        )


if __name__ == "__main__":
    analyzer = KneePathAnalysis()
    image_path = "../../data/origin/TEDIEGW3.nii"
    mask_path = "../../data/mask/TEDIEGW3.Labels.nii"
    output_directory = "../../slicer_export"

    try:
        results = analyzer.run_full_analysis(image_path, mask_path, output_directory)
        print("膝关节路径分析完成！")
    except Exception as e:
        print(f"分析过程中出错: {e}")
