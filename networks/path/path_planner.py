import numpy as np
import logging
from .knee_analysis import KneePathAnalysis
#将深度学习分割结果转换为具体的手术路径，在path_planning中被调用
class PathPlanner:
    def __init__(self, config=None):
        self.config = config or {
            'dilation_iter': 3,
            'dist_thresh': 15.0,
            'trim_ratio': 0.25
        }
        self.knee_analyzer = KneePathAnalysis()

    def plan_paths(self, segmentation, image_volume, affine=None):
        """
        基于分割结果进行路径规划
        """
        batch_size = segmentation.shape[0]
        batch_paths = []

        for b in range(batch_size):
            try:
                seg_np = segmentation[b].detach().cpu().numpy()
                image_np = image_volume[b, 0].detach().cpu().numpy()
                mask_data = np.argmax(seg_np, axis=0)

                paths = self.knee_analyzer.analyze_knee_paths(
                    image_np, mask_data, affine
                )
                batch_paths.append(paths)

            except Exception as e:
                logging.warning(f"Volume {b} path planning failed: {e}")
                batch_paths.append(None)

        return batch_paths