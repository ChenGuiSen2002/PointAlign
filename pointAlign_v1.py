import os, time
import numpy as np
import open3d as o3d
import trimesh
from teaserpp_python import _teaserpp as teaserpp_python
import copy
from scipy.spatial.transform import Rotation as R
import cv2




class InHandPoseEstimator:
    def __init__(self, cad_path, camera_intrinsic_matrix, hand_eye_transform):
        """
        初始化位姿估计器
        :param cad_path: 物体CAD模型路径 (.obj,.ply)
        :param camera_intrinsic_matrix: 相机内参 (3x3 numpy array)
        :param hand_eye_transform: 手眼标定矩阵 T_gripper_camera (4x4 numpy array)
        """
        self.cam_K = camera_intrinsic_matrix
        self.T_g_c = hand_eye_transform # Gripper to Camera
        
        # 1. 加载并预处理CAD模型 (Source Point Cloud)
        self.cad_mesh = o3d.io.read_triangle_mesh(cad_path)
        # 采样点云用于配准 (Source)
        self.source_pcd = self.cad_mesh.sample_points_poisson_disk(number_of_points=5000)
        
        # 预计算CAD模型的FPFH特征
        self.voxel_size = 0.005 # 5mm voxel size for downsampling
        self.source_pcd_down, self.source_fpfh = self._preprocess_point_cloud(self.source_pcd, self.voxel_size)

    def _preprocess_point_cloud(self, pcd, voxel_size):
        """
        点云预处理：降采样、估计法线、计算FPFH特征
        """
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 估计法线 (对于FPFH是必须的)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # 计算FPFH特征
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def remove_gripper_points(self, scene_pcd, gripper_urdf_path, joint_angles):
        """
        使用URDF模型剔除场景中的夹爪点云 (Self-Filtering)
        :param scene_pcd: 原始场景点云
        :param gripper_urdf_path: 机器人URDF文件路径
        :param joint_angles: 当前关节角 {'joint_name': angle}
        """
        # 注意：这里使用简化逻辑，实际工程中通常使用 urdfpy 加载模型
        # 并计算每个link在相机坐标系下的mesh，然后检查点是否在mesh内部或表面
        # 这里为了代码可运行，使用简单的空间裁剪示例
        
        print("[Info] Filtering gripper points...")
        # 假设夹爪在相机前方 z < 0.1m 的区域或者特定包围盒内
        # 实际代码应加载 urdfpy 并使用 collision manager
        
        # 简单示例：剔除距离相机过近的点（可能是噪声）和过远的点（背景）
        points = np.asarray(scene_pcd.points)
        mask = (points[:, 2] > 0.1) & (points[:, 2] < 1.0) # 保留 10cm 到 1m 的点
        
        # 模拟剔除：假设我们知道夹爪的大致位置，这里先不做复杂URDF碰撞检测
        # 在真实应用中，需将URDF模型变换到相机系，然后使用 trimesh.ray.contains_points 剔除

        idx = np.flatnonzero(mask).astype(int).tolist()   # 关键：转成 list[int]

        if len(idx) == 0:
            print("[Warn] mask produced 0 points, skip filtering.")
            return scene_pcd
        
        filtered_pcd = scene_pcd.select_by_index(idx)
        return filtered_pcd

    def execute_teaser_registration(self, rgb_image, depth_image):
        """
        执行配准主流程
        :return: T_gripper_object (4x4), Visualization Geometries
        """
        print(" Generating Scene Point Cloud from RGB-D...")
        # Open3D RGBD图像生成
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image), 
            o3d.geometry.Image(depth_image),
            depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False
        )
        K = np.asarray(self.cam_K, dtype=np.float64)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(rgb_image.shape[1]),
            height=int(rgb_image.shape[0]),
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )
        scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        print("depth_m stats:",
            "min", np.nanmin(depth_image),
            "max", np.nanmax(depth_image),
            "nonzero", np.count_nonzero(depth_image),
            "ratio", np.count_nonzero(depth_image)/depth_image.size)

        print("scene_pcd points:", np.asarray(scene_pcd.points).shape[0])

        
        # 剔除夹爪和背景噪声 (Target Point Cloud)
        scene_pcd_filtered = self.remove_gripper_points(scene_pcd, None, None)
        
        # 预处理场景点云
        target_pcd_down, target_fpfh = self._preprocess_point_cloud(scene_pcd_filtered, self.voxel_size)
        
        print(f" Extracting Features: Source({len(self.source_pcd_down.points)}), Target({len(target_pcd_down.points)})")
        self.save_pcd_debug(target_pcd_down, "debug/target_pcd_down")

        # 建立对应关系 (Correspondences)
        # 使用互惠匹配 (Reciprocal Matching) 提高鲁棒性
        print(" Matching Features...")
        scores = np.dot(self.source_fpfh.data.T, target_fpfh.data)
        # 简单的最近邻匹配，实际TEASER++不需要完美匹配，只需要部分内点
        src_indices = np.argmax(scores, axis=1) # 对每个source找最近的target
        dst_indices = np.arange(len(src_indices))
        
        # 准备TEASER++输入数据 (3xN numpy arrays)
        # 注意：这里我们选取部分可信度高的匹配，或者直接输入所有匹配让TEASER剔除
        src_points = np.asarray(self.source_pcd_down.points).T
        dst_points = np.asarray(target_pcd_down.points)[src_indices].T
        
        print(" Running TEASER++ Solver...")
        # TEASER++ 配置
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1.0 # 噪声界限平方
        solver_params.noise_bound = 0.05 # 估计的最大噪声 (5cm)
        solver_params.estimate_scaling = False # 已知尺度（CAD模型与真实物体尺度一致）
        # solver_params.rotation_estimation_algorithm = \
        #     teaserpp_python.RobustRegistrationSolver.rotation_estimation_algorithm.GNC_TLS
        # 先看默认是什么
        print("default rotation_estimation_algorithm =", solver_params.rotation_estimation_algorithm)
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(src_points, dst_points)
        
        solution = solver.getSolution()
        R_cam_obj = solution.rotation
        t_cam_obj = solution.translation
        
        T_cam_obj = np.eye(4)
        T_cam_obj[:3, :3] = R_cam_obj
        T_cam_obj[:3, 3] = t_cam_obj
        
        print("TEASER++ Estimated Transform (Object -> Camera):\n", T_cam_obj)
        
        # 计算物体相对于夹爪的位姿
        # T_gripper_object = T_gripper_camera * T_camera_object
        T_gripper_object = np.dot(self.T_g_c, T_cam_obj)
        
        # 提取旋转矩阵
        R_gripper_object = T_gripper_object[:3, :3]
        print("\n>>> Final Result: Rotation Matrix (Object -> Gripper Center) <<<")
        print(R_gripper_object)
        
        return T_cam_obj, scene_pcd_filtered

    def save_pcd_debug(self, pcd, out_prefix, max_points=200000):
        """
        保存点云到 PLY，并用 matplotlib 生成一个 3D 散点图 PNG（headless 可用）
        """
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

        # 1) 保存 ply
        o3d.io.write_point_cloud(out_prefix + ".ply", pcd)

        # 2) 保存 png（采样避免太慢）
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            return
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.1)
        ax.set_axis_off()
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        plt.savefig(out_prefix + ".png", dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    def visualize_result(self, T_cam_obj, scene_pcd):
        """
        方案A + 方案C（headless 可用）
        A: 保存 scene / obj_transformed / merged 点云 + 保存 T_cam_obj
        C: 用内参把点云投影到图像平面，生成并保存渲染图/叠加图（不需要 OpenGL 窗口）
        """

        # -------------------------
        # 0) 输出目录
        # -------------------------
        ts = int(time.time() * 1000)
        out_dir = os.path.join(os.getcwd(), "vis_outputs", f"run_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        # -------------------------
        # 1) 准备“变换后的CAD点云”
        # -------------------------
        obj_pcd = copy.deepcopy(self.source_pcd)
        obj_pcd.paint_uniform_color([1.0, 0.706, 0.0])  # yellow (RGB, float)
        obj_pcd.transform(T_cam_obj)

        # scene 上色（如果没有颜色则不处理）
        try:
            if len(scene_pcd.colors) == 0:
                scene_pcd.paint_uniform_color([0.0, 0.651, 0.929])  # blue
        except Exception:
            pass

        # -------------------------
        # 2) 方案A：保存点云 + 矩阵
        # -------------------------
        T_txt = os.path.join(out_dir, "T_cam_obj.txt")
        T_npy = os.path.join(out_dir, "T_cam_obj.npy")
        np.savetxt(T_txt, T_cam_obj, fmt="%.8f")
        np.save(T_npy, T_cam_obj)

        scene_ply = os.path.join(out_dir, "scene.ply")
        obj_ply = os.path.join(out_dir, "object_transformed.ply")
        o3d.io.write_point_cloud(scene_ply, scene_pcd, write_ascii=False, compressed=True)
        o3d.io.write_point_cloud(obj_ply, obj_pcd, write_ascii=False, compressed=True)

        # 合并点云（尽量用 numpy 拼接，避免不同 open3d 后端差异）
        scene_pts = np.asarray(scene_pcd.points)
        obj_pts = np.asarray(obj_pcd.points)

        scene_cols = None
        obj_cols = None
        try:
            if len(scene_pcd.colors) == len(scene_pts) and len(scene_pts) > 0:
                scene_cols = np.asarray(scene_pcd.colors)
        except Exception:
            scene_cols = None

        try:
            if len(obj_pcd.colors) == len(obj_pts) and len(obj_pts) > 0:
                obj_cols = np.asarray(obj_pcd.colors)
        except Exception:
            obj_cols = None

        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.vstack([scene_pts, obj_pts]))

        if scene_cols is not None and obj_cols is not None:
            merged.colors = o3d.utility.Vector3dVector(np.vstack([scene_cols, obj_cols]))
        else:
            merged.paint_uniform_color([0.8, 0.8, 0.8])

        merged_ply = os.path.join(out_dir, "merged.ply")
        o3d.io.write_point_cloud(merged_ply, merged, write_ascii=False, compressed=True)

        # -------------------------
        # 3) 方案C：内参投影“渲染”保存图片（不需要窗口/GL）
        # -------------------------
        # 背景图：优先使用你在 execute_teaser_registration 里缓存的 RGB
        # 建议你加：self._last_rgb = rgb_image
        bg = getattr(self, "_last_rgb", None)
        if isinstance(bg, np.ndarray) and bg.ndim == 3:
            H, W = bg.shape[:2]
            base = bg.copy()
            # 约定 base 为 uint8；如果不是，做一次转换
            if base.dtype != np.uint8:
                base = np.clip(base, 0, 255).astype(np.uint8)
        else:
            # 没有背景图就用默认 640x480 黑底
            H, W = 480, 640
            base = np.zeros((H, W, 3), dtype=np.uint8)

        K = np.asarray(self.cam_K, dtype=np.float64)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        def _project_render(points_xyz, colors_rgb_float01=None, bg_img=None):
            """
            points_xyz: (N,3) in camera frame
            colors_rgb_float01: (N,3) float in [0,1] (Open3D default); if None -> depth gray
            bg_img: (H,W,3) uint8 BGR/RGB 都行，只是保存时不做通道交换
            return: rendered uint8 image
            """
            img = (bg_img.copy() if bg_img is not None else np.zeros((H, W, 3), dtype=np.uint8))

            if points_xyz.size == 0:
                return img

            x = points_xyz[:, 0]
            y = points_xyz[:, 1]
            z = points_xyz[:, 2]

            valid = z > 1e-6
            if not np.any(valid):
                return img

            x, y, z = x[valid], y[valid], z[valid]

            u = (fx * x / z + cx).astype(np.int32)
            v = (fy * y / z + cy).astype(np.int32)

            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not np.any(inside):
                return img

            u, v, z = u[inside], v[inside], z[inside]

            flat = v * W + u
            order = np.argsort(z)  # 近的先，做 z-buffer
            flat_s = flat[order]

            # 每个像素只保留最近点
            _, first = np.unique(flat_s, return_index=True)
            keep = order[first]
            flat_keep = flat[keep]

            if colors_rgb_float01 is not None:
                cols = colors_rgb_float01[valid][inside]  # (M,3) float01
                cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
            else:
                # 深度灰度（近亮远暗）
                z_norm = (z - z.min()) / (max(1e-6, (z.max() - z.min())))
                g = np.clip((1.0 - z_norm) * 255.0, 0, 255).astype(np.uint8)
                cols = np.stack([g, g, g], axis=1)

            img.reshape(-1, 3)[flat_keep] = cols[keep]
            return img

        # scene 渲染（尽量用 scene 自带颜色）
        scene_cols = None
        try:
            if len(scene_pcd.colors) == len(scene_pts) and len(scene_pts) > 0:
                scene_cols = np.asarray(scene_pcd.colors)
        except Exception:
            scene_cols = None
        img_scene = _project_render(scene_pts, scene_cols, bg_img=None)

        # object 渲染（黄色）
        obj_cols = np.tile(np.array([[1.0, 0.706, 0.0]], dtype=np.float32), (len(obj_pts), 1))
        img_obj = _project_render(obj_pts, obj_cols, bg_img=None)

        # 叠加到背景
        overlay = base.copy()
        overlay = _project_render(scene_pts, scene_cols, bg_img=overlay)
        overlay = _project_render(obj_pts, obj_cols, bg_img=overlay)

        # 保存图片（不依赖 OpenGL）
        import cv2
        cv2.imwrite(os.path.join(out_dir, "render_scene.png"), img_scene)
        cv2.imwrite(os.path.join(out_dir, "render_object.png"), img_obj)
        cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)

        print(f"[OK] Saved outputs to: {out_dir}")
        print(f"  - {scene_ply}")
        print(f"  - {obj_ply}")
        print(f"  - {merged_ply}")
        print(f"  - {T_txt}")
        print(f"  - overlay.png / render_scene.png / render_object.png")

# ================= 使用示例 =================
if __name__ == "__main__":
    # === 1) 你的文件路径 ===
    color_path = "data/realsense_aligned_color_1768296485768.png"
    depth_m_path = "data/realsense_aligned_aligned_depth_m_1768296485768.npy"
    K_path = "data/realsense_aligned_K_color_1768296485768.txt"
    cad_path = "data/box.ply"  # TODO: 换成你的 CAD (.ply/.obj/.stl 转 ply/obj)

    # === 2) 读取 K ===
    # 你上传的 K 文件就是 3x3 矩阵：realsense_aligned_K_color_*.txt :contentReference[oaicite:0]{index=0}
    K = np.loadtxt(K_path).astype(np.float32)

    # === 3) 手眼标定（必须你自己填真实值） ===
    # 建议统一成 ^G T_C（相机坐标系到夹爪坐标系）或在下面按你的定义改乘法
    T_hand_eye = np.eye(4, dtype=np.float32)
    # TODO: 用你标定结果替换

    # === 4) 读取 RGB（注意 Open3D 建议用 RGB 顺序） ===
    bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # === 5) 读取 Depth（米单位，float32） ===
    depth_m = np.load(depth_m_path).astype(np.float32)  # HxW, meters

    # 尺寸一致性检查（必须）
    assert rgb.shape[0] == depth_m.shape[0] and rgb.shape[1] == depth_m.shape[1], \
        f"RGB size {rgb.shape[:2]} != Depth size {depth_m.shape}"

    # === 初始化并运行 ===
    estimator = InHandPoseEstimator(cad_path, K, T_hand_eye)

    T_est, scene_cloud = estimator.execute_teaser_registration(rgb, depth_m)
    estimator.visualize_result(T_est, scene_cloud)