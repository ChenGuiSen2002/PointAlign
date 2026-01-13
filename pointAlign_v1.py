import numpy as np
import open3d as o3d
import trimesh
import teaserpp_python
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
        
        filtered_pcd = scene_pcd.select_by_index(np.where(mask))
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
            depth_scale=1000.0, depth_trunc=1.0, convert_rgb_to_intensity=False
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=rgb_image.shape[1], height=rgb_image.shape,
            fx=self.cam_K, fy=self.cam_K[1,1], cx=self.cam_K, cy=self.cam_K[1, 2]
        )
        scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 剔除夹爪和背景噪声 (Target Point Cloud)
        scene_pcd_filtered = self.remove_gripper_points(scene_pcd, None, None)
        
        # 预处理场景点云
        target_pcd_down, target_fpfh = self._preprocess_point_cloud(scene_pcd_filtered, self.voxel_size)
        
        print(f" Extracting Features: Source({len(self.source_pcd_down.points)}), Target({len(target_pcd_down.points)})")

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
        solver_params.rotation_estimation_algorithm = \
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
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

    def visualize_result(self, T_cam_obj, scene_pcd):
        """
        可视化配准结果：显示场景点云和变换后的CAD模型
        """
        source_temp = copy.deepcopy(self.source_pcd)
        source_temp.paint_uniform_color([1, 0.706, 0]) # 黄色是物体模型
        source_temp.transform(T_cam_obj)
        
        scene_pcd.paint_uniform_color([0, 0.651, 0.929]) # 蓝色是场景点云
        
        o3d.visualization.draw_geometries([source_temp, scene_pcd], 
                                          window_name="TEASER++ Registration Result",
                                          width=800, height=600)

# ================= 使用示例 =================
if __name__ == "__main__":
    # === 1) 你的文件路径 ===
    color_path = "realsense_aligned_color_1768296485768.png"
    depth_m_path = "realsense_aligned_aligned_depth_m_1768296485768.npy"
    K_path = "realsense_aligned_K_color_1768296485768.txt"
    cad_path = "temp_obj.ply"  # TODO: 换成你的 CAD (.ply/.obj/.stl 转 ply/obj)

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