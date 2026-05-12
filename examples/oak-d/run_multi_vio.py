#!/usr/bin/env python3
"""
多 OAK-D VIO — 主进程可视化 + 内存看门狗 + 坐标系转换。
环境变量覆盖默认值：
  ENABLE_VIZ=1   -> 开启 Rerun 可视化
  ENABLE_HDF5=1  -> 开启轨迹保存
"""

import os, sys, time, signal, gc
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Process, Queue, set_start_method

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation
import cuvslam as vslam
import h5py
import psutil
import rerun as rr
import rerun.blueprint as rrb

# ========== ROS2（可选）==========
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from builtin_interfaces.msg import Time
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ========== 调参配置 ==========
@dataclass
class TuningParams:
    """所有可调节参数集中在此，方便快速调参。"""

    # --- 相机 ---
    fps: int = 30
    resolution: tuple[int, int] = (640, 480)
    warmup_frames: int = 60            # 前 N 帧用于重力初始化，不输出位姿

    # --- IMU 噪声（OAK-D 标定值 × 系数）---
    imu_frequency: int = 200
    gyro_noise_density: float = 6.07e-03
    gyro_random_walk: float = 3.62e-05
    accel_noise_density: float = 3.36e-02
    accel_random_walk: float = 9.83e-04

    # --- 图像边框裁剪（排除鱼眼畸变边缘）---
    border_top: int = 15
    border_bottom: int = 15
    border_left: int = 15
    border_right: int = 15

    # --- 追踪 ---
    odometry_mode: str = "Inertial"    # Multicamera / Inertial / RGBD / Mono
    multicam_mode: str = "Precision"   # Precision / Performance / Moderate
    async_sba: bool = True
    rectified_stereo_camera: bool = False

    # --- SLAM ---
    enable_reading_internals: bool = False
    max_landmarks_distance: float = 30.0
    max_map_size: int = 300
    map_cell_size: float = 0.0          # 0 = 从基线自动计算
    throttling_time_ms: int = 1000

TP = TuningParams()

# ========== 运行配置 ==========
DEFAULT_ENABLE_VIZ = True
DEFAULT_ENABLE_HDF5 = False
ENABLE_VISUALIZATION = os.environ.get("ENABLE_VIZ", str(int(DEFAULT_ENABLE_VIZ))) == "1"
ENABLE_HDF5 = os.environ.get("ENABLE_HDF5", str(int(DEFAULT_ENABLE_HDF5))) == "1"

# ---------- 内存看门狗 ----------
MEMORY_CRITICAL = 0.92       # 系统内存超过 92% 自动终止
MEMORY_CHECK_INTERVAL = 2.0  # 检查间隔（秒）
VIZ_FRAME_SKIP = 3           # 每隔 N 帧发送一帧到可视化队列

# ---------- 坐标系 ----------
# cuVSLAM 原生使用 OpenCV 约定：X-右, Y-下, Z-前
# 设为 "ros" 输出 X-前, Y-左, Z-上（与 OAK 官方约定一致）
POSE_CONVENTION = "ros"

# ---------- 相机安装 ----------
# 安装朝向：key=逻辑相机 ID(cid), value="normal"(正装) 或 "inverted"(倒装,绕Z轴180°)
CAMERA_MOUNTING: dict[int, str] = {29: "inverted", 40: "inverted"}

# MXID（硬件序列号）→ 逻辑相机 ID，确保同一物理相机始终获得相同 cid
# 留空则按枚举顺序自动分配。运行一次查看 MXID 后填入。
CAMERA_ID_MAP: dict[str, int] = {
    "14442C1091AFD5D200": 29,
    "19443010A1AC187E00": 40,
    "19443010D1061C7E00": 41,
    "1944301091C0187E00": 46,
}

# cuVSLAM → ROS 坐标变换矩阵
_R_CUVSLAM_TO_ROS = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
_Q_CUVSLAM_TO_ROS = Rotation.from_matrix(_R_CUVSLAM_TO_ROS).as_quat()  # [x, y, z, w]

CM_TO_METERS = 100

# ========== 辅助函数 ==========
def oak_transform_to_pose(extr):
    arr = np.array(extr)
    rot = arr[:3, :3]; t = arr[:3, 3] / CM_TO_METERS
    return vslam.Pose(rotation=Rotation.from_matrix(rot).as_quat(), translation=t)

def set_cuvslam_camera(params):
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, params["distortion"])
    cam.focal = (params["intrinsics"][0][0], params["intrinsics"][1][1])
    cam.principal = (params["intrinsics"][0][2], params["intrinsics"][1][2])
    cam.size = params["resolution"]
    cam.rig_from_camera = oak_transform_to_pose(params["extrinsics"])
    cam.border_top = TP.border_top; cam.border_bottom = TP.border_bottom
    cam.border_left = TP.border_left; cam.border_right = TP.border_right
    return cam

def get_stereo_calib(calib, res):
    s = {"left": {}, "right": {}}
    for side, sock in zip(["left", "right"], [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]):
        s[side]["resolution"] = res
        s[side]["intrinsics"] = calib.getCameraIntrinsics(sock, res[0], res[1])
        s[side]["extrinsics"] = calib.getCameraExtrinsics(sock, dai.CameraBoardSocket.CAM_A)
        s[side]["distortion"] = calib.getDistortionCoefficients(sock)[:8]
    return s

def get_imu_calib(extr):
    """DepthAI 固件已将 IMU 原始数据旋转到相机坐标系 (X-右,Y-下,Z-前)。
       EEPROM 外参矩阵包含物理芯片旋转——重复应用会双重旋转。
       因此只取平移偏移量，旋转设为单位四元数。"""
    imu = vslam.ImuCalibration()
    arr = np.array(extr)
    t = arr[:3, 3] / CM_TO_METERS
    imu.rig_from_imu = vslam.Pose(rotation=np.array([0., 0., 0., 1.]), translation=t)
    imu.gyroscope_noise_density = TP.gyro_noise_density
    imu.gyroscope_random_walk = TP.gyro_random_walk
    imu.accelerometer_noise_density = TP.accel_noise_density
    imu.accelerometer_random_walk = TP.accel_random_walk
    imu.frequency = TP.imu_frequency
    return imu

def ts_to_ns(ts):
    """将 DepthAI 时间戳转换为纳秒整数。"""
    if ts is None:
        return None
    if hasattr(ts, 'total_seconds'):
        return int(ts.total_seconds() * 1e9)
    if hasattr(ts, 'seconds') and hasattr(ts, 'microseconds'):
        return int(ts.seconds * 1e9 + ts.microseconds * 1000)
    return None

def color_from_id(uid):
    return [(uid * 17) % 256, (uid * 31) % 256, (uid * 47) % 256]

def resolve_camera_id(mxid, fallback):
    if mxid in CAMERA_ID_MAP:
        return CAMERA_ID_MAP[mxid]
    print(f"警告: MXID {mxid} 不在 CAMERA_ID_MAP 中，使用回退 cid={fallback}")
    return fallback

def get_imu_timestamp(pkt):
    """从 IMU 数据包中提取时间戳（纳秒）。"""
    for m in ('getTimestampDevice', 'getTimestamp', 'getTimestampHost'):
        if hasattr(pkt, m):
            ts = getattr(pkt, m)()
            if ts is not None:
                return ts_to_ns(ts)
    for sf in ('acceleroMeter', 'gyroscope'):
        if hasattr(pkt, sf):
            so = getattr(pkt, sf)
            for a in ('tsDevice', 'timestamp', 'timestampDevice'):
                if hasattr(so, a):
                    ts = getattr(so, a)
                    if ts is not None:
                        return ts_to_ns(ts)
    return None

def get_msg_timestamp(msg):
    """从同步帧消息中提取设备时间戳（纳秒）。"""
    for m in ('getTimestampDevice', 'getTimestamp'):
        if hasattr(msg, m):
            ts = getattr(msg, m)()
            if ts is not None:
                return ts_to_ns(ts)
    return None

def transform_pose_to_ros(trans, quat_xyzw):
    """将位姿从 cuVSLAM 约定 (X-右,Y-下,Z-前) 转换到 ROS 约定 (X-前,Y-左,Z-上)。"""
    trans_ros = np.array([trans[2], -trans[0], -trans[1]])
    q_cb = Rotation.from_quat([_Q_CUVSLAM_TO_ROS[0], _Q_CUVSLAM_TO_ROS[1],
                                _Q_CUVSLAM_TO_ROS[2], _Q_CUVSLAM_TO_ROS[3]])
    q_cuv = Rotation.from_quat([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]])
    q_ros = (q_cb * q_cuv * q_cb.inv()).as_quat()
    return trans_ros, np.array([q_ros[0], q_ros[1], q_ros[2], q_ros[3]])

# ---------- 内存监控 ----------
def memory_status_str():
    mem = psutil.virtual_memory()
    return f"{mem.percent:.1f}% ({mem.used // (1024**3)}GB / {mem.total // (1024**3)}GB)"

def check_system_memory():
    return psutil.virtual_memory().percent >= MEMORY_CRITICAL * 100

def process_rss_mb():
    return psutil.Process().memory_info().rss // (1024**2)

# ========== ROS2 发布器 ==========
class CameraPosePublisher(Node):
    def __init__(self, cid):
        super().__init__(f'vio_camera_{cid}')
        self.pub = self.create_publisher(PoseStamped, f'cam_{cid}/pose', 10)
        self.cid = cid

    def publish_pose(self, ts_ns, trans, quat_xyzw):
        msg = PoseStamped()
        sec, nsec = int(ts_ns // 1e9), int(ts_ns % 1e9)
        stamp = Time(); stamp.sec = sec; stamp.nanosec = nsec
        msg.header.stamp = stamp
        msg.header.frame_id = f'cam_{self.cid}_odom'
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        self.pub.publish(msg)

# ========== VIO 子进程 ==========
def vio_process(camera_id, device_id, vis_queue, traj_queue, enable_viz):
    if ROS2_AVAILABLE:
        rclpy.init(args=None)
        ros_node = CameraPosePublisher(camera_id)
    else:
        ros_node = None

    infos = dai.Device.getAllAvailableDevices()
    target = next((d for d in infos if d.deviceId == device_id), None)
    if target is None:
        print(f"[Cam {camera_id}] 设备 {device_id} 未找到。")
        traj_queue.put((camera_id, []))
        return

    device = dai.Device(target)
    mounting = CAMERA_MOUNTING.get(camera_id, "normal")
    #print(f"[Cam {camera_id}] 安装={mounting}, 坐标系={'Z-上(ROS)' if POSE_CONVENTION == 'ros' else 'Z-前(cuVSLAM)'}")
    calib = device.readCalibration()
    stereo = get_stereo_calib(calib, TP.resolution)
    cams = [set_cuvslam_camera(stereo["left"]), set_cuvslam_camera(stereo["right"])]
    rig = vslam.Rig(cams)
    imu_extr = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A)
    rig.imus = [get_imu_calib(imu_extr)]

    odom_mode = getattr(vslam.Tracker.OdometryMode, TP.odometry_mode)
    mc_mode = getattr(vslam.Tracker.MulticameraMode, TP.multicam_mode)
    odom_cfg = vslam.Tracker.OdometryConfig(
        async_sba=TP.async_sba,
        enable_final_landmarks_export=False,
        enable_observations_export=True,
        rectified_stereo_camera=TP.rectified_stereo_camera,
        odometry_mode=odom_mode,
        multicam_mode=mc_mode,
    )
    slam_cfg = vslam.Tracker.SlamConfig(
        enable_reading_internals=TP.enable_reading_internals,
        max_landmarks_distance=TP.max_landmarks_distance,
        max_map_size=TP.max_map_size,
        map_cell_size=TP.map_cell_size,
        throttling_time_ms=TP.throttling_time_ms,
    )
    tracker = vslam.Tracker(rig, odom_cfg, slam_config=slam_cfg)

    # 构建 DepthAI pipeline
    pipeline = dai.Pipeline(device)
    lcam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=TP.fps)
    rcam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=TP.fps)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5 / TP.fps))
    lo = lcam.requestOutput(TP.resolution, type=dai.ImgFrame.Type.GRAY8)
    ro = rcam.requestOutput(TP.resolution, type=dai.ImgFrame.Type.GRAY8)
    lo.link(sync.inputs["left"]); ro.link(sync.inputs["right"])
    sq = sync.out.createOutputQueue()

    imu_node = pipeline.create(dai.node.IMU)
    imu_node.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], TP.imu_frequency)
    imu_node.setBatchReportThreshold(1); imu_node.setMaxBatchReports(10)
    iq = imu_node.out.createOutputQueue()

    pipeline.start()

    frame_id = 0
    last_imu_ts = None; last_sent_ts = None
    imu_buf = []
    traj_full = []

    try:
        while pipeline.isRunning():
            msg = sq.get()
            ts_ns = get_msg_timestamp(msg)
            if ts_ns is None:
                continue

            # 收集 IMU 数据
            for im in iq.tryGetAll():
                for pkt in im.packets:
                    its = get_imu_timestamp(pkt)
                    if its is not None:
                        imu_buf.append((its, pkt.acceleroMeter, pkt.gyroscope))
            imu_buf.sort(key=lambda x: x[0])

            # 按时间顺序喂给 tracker
            while imu_buf and imu_buf[0][0] <= ts_ns:
                its, accel, gyro = imu_buf.pop(0)
                if last_sent_ts is not None and its <= last_sent_ts:
                    continue
                if last_imu_ts is not None and its <= last_imu_ts:
                    continue
                last_imu_ts = its
                meas = vslam.ImuMeasurement()
                meas.timestamp_ns = its
                meas.linear_accelerations = np.array([accel.x, accel.y, accel.z], dtype=np.float32)
                meas.angular_velocities = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)
                tracker.register_imu_measurement(0, meas)
                last_sent_ts = its

            frame_id += 1
            if frame_id <= TP.warmup_frames:
                continue

            left_img = msg["left"].getCvFrame(); right_img = msg["right"].getCvFrame()
            odom_est, slam_est = tracker.track(ts_ns, (left_img, right_img))
            pose = slam_est if slam_est is not None else odom_est.world_from_rig
            if pose is None:
                print(f"[Cam {camera_id}] 追踪失败 frame={frame_id}")
                continue

            trans = pose.translation.copy(); quat = pose.rotation.copy()
            if POSE_CONVENTION == "ros":
                trans, quat = transform_pose_to_ros(trans, quat)
            traj_full.append((ts_ns, trans, quat))

            raw_obs = tracker.get_last_observations(0)
            obs_list = [(o.u, o.v, o.id) for o in raw_obs] if raw_obs else []

            if ros_node:
                ros_node.publish_pose(ts_ns, trans, quat)
                rclpy.spin_once(ros_node, timeout_sec=0)

            if frame_id % 60 == 0:
                print(f"[Cam {camera_id}] pos=({trans[0]:.3f},{trans[1]:.3f},{trans[2]:.3f}) "
                      f"quat=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f})")

            if frame_id % 300 == 0:
                gc.collect()
                print(f"[Cam {camera_id}] RSS={process_rss_mb()}MB")

            if enable_viz and vis_queue is not None and frame_id % VIZ_FRAME_SKIP == 0:
                # 倒置相机：绕 X 轴(前向)旋转 180°，翻转上下和左右
                if mounting == "inverted":
                    vis_trans = np.array([trans[0], -trans[1], -trans[2]])
                    r_x180 = Rotation.from_euler('x', 180, degrees=True)
                    r_orig = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
                    vis_quat = (r_x180 * r_orig).as_quat()
                else:
                    vis_trans, vis_quat = trans, quat
                vis_queue.put({
                    'camera_id': camera_id,
                    'timestamp_ns': ts_ns,
                    'left_image': left_img,
                    'translation': vis_trans,
                    'rotation': vis_quat,
                    'observations': obs_list,
                })
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        device.close()
        if ros_node:
            try:
                ros_node.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass
        traj_queue.put((camera_id, traj_full))
        print(f"[Camera {camera_id}] 结束 {len(traj_full)} 个位姿。")

# ========== 进程管理 ==========
def shutdown_processes(processes, reason=""):
    """优雅关闭所有 VIO 子进程——Ctrl+C 和内存看门狗共用。"""
    if reason:
        print(f"\n!!! {reason} !!!")
    print(f"系统内存: {memory_status_str()}")
    for p in processes:
        if p.is_alive():
            os.kill(p.pid, signal.SIGINT)
    for p in processes:
        p.join(timeout=5)
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

def save_trajectories(traj_queue):
    if not ENABLE_HDF5:
        while not traj_queue.empty():
            traj_queue.get()
        print("HDF5 未启用，轨迹未保存。")
        return
    all_traj = {}
    while not traj_queue.empty():
        cid, data = traj_queue.get()
        all_traj[cid] = data
    if not all_traj:
        print("警告: 未收到任何轨迹数据。")
        return
    with h5py.File("multi_oak_vio.h5", "w") as f:
        for cid, data in all_traj.items():
            if not data:
                continue
            timestamps = np.array([d[0] for d in data])
            positions = np.array([d[1] for d in data])
            quats = np.array([d[2] for d in data])
            grp = f.create_group(f"cam_{cid}")
            grp.create_dataset("timestamps", data=timestamps)
            grp.create_dataset("positions", data=positions)
            grp.create_dataset("quaternions_xyzw", data=quats)
            print(f"已保存 cam_{cid}: {len(timestamps)} 个位姿。")
    print("全部轨迹已保存到 multi_oak_vio.h5")

# ========== 主进程 ==========
def main():
    set_start_method('spawn', force=True)

    enable_viz = ENABLE_VISUALIZATION
    convention_str = "X-前 Y-左 Z-上 (ROS)" if POSE_CONVENTION == "ros" else "X-右 Y-下 Z-前 (cuVSLAM)"
    print(f"可视化: {'ON' if enable_viz else 'OFF'},  "
          f"HDF5: {'ON' if ENABLE_HDF5 else 'OFF'},  "
          f"坐标系: {convention_str}")

    if enable_viz:
        rr.init("multi_oak_vio", spawn=True)
        time.sleep(2)

    # 枚举设备
    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        print("未找到 OAK-D 设备。")
        return

    device_assignments: list[tuple[int, str, str, str]] = []  # (cid, mxid, name, mounting)
    print(f"发现 {len(infos)} 台设备:")
    for i, info in enumerate(infos):
        cid = resolve_camera_id(info.deviceId, i)
        mounting = CAMERA_MOUNTING.get(cid, "normal")
        device_assignments.append((cid, info.deviceId, info.name, mounting))
        mark = "*" if info.deviceId in CAMERA_ID_MAP else " "
        print(f"  [{mark}] cid={cid} mxid={info.deviceId} name={info.name} 安装={mounting}")
    if not CAMERA_ID_MAP:
        print("  提示: 设置 CAMERA_ID_MAP = {mxid: cid, ...} 以固定相机编号。")

    # Rerun 蓝图
    if enable_viz:
        cam_views = [rrb.Spatial2DView(origin=f"cam_{cid}/left", name=f"cam{cid}-left")
                     for cid, _, _, _ in device_assignments]
        blueprint = rrb.Blueprint(rrb.Horizontal(rrb.Vertical(*cam_views), rrb.Spatial3DView(name="3D")))
        rr.send_blueprint(blueprint)

    # 虚拟轨迹偏移（垂直平面：倒置相机上层 Z=+0.5，正装相机下层 Z=-0.5）
    sorted_cids = sorted(cid for cid, _, _, _ in device_assignments)
    normal_cids = [cid for cid, _, _, m in device_assignments if m == "normal"]
    inverted_cids = [cid for cid, _, _, m in device_assignments if m == "inverted"]
    offsets = {}
    for row_cids, z in [(normal_cids, -0.5), (inverted_cids, 0.5)]:
        for col, cid in enumerate(row_cids):
            offsets[cid] = np.array([0.0, (col - (len(row_cids) - 1) / 2) * 1.0, z])

    # 队列 & 启动子进程
    vis_queue = Queue() if enable_viz else None
    traj_queue = Queue()
    processes = []
    vslam.warm_up_gpu()
    for cid, mxid, _, _ in device_assignments:
        p = Process(target=vio_process, args=(cid, mxid, vis_queue, traj_queue, enable_viz))
        p.start()
        processes.append(p)

    # 主循环
    shutdown_reason = ""
    last_mem_check = time.time()
    last_gc = time.time()
    try:
        if enable_viz:
            trajectories = {cid: deque(maxlen=2000) for cid in sorted_cids}
            viz_frame_id = 0
            last_traj_update = {cid: 0 for cid in sorted_cids}

        while any(p.is_alive() for p in processes):
            if enable_viz:
                while not vis_queue.empty():
                    data = vis_queue.get()
                    cid = data['camera_id']
                    trans = data['translation']; quat = data['rotation']
                    left_img = data['left_image']; observations = data['observations']

                    vis_trans = trans + offsets.get(cid, np.zeros(3))
                    trajectories[cid].append(vis_trans)

                    prefix = f"cam_{cid}"
                    rr.set_time_sequence("frame", viz_frame_id)
                    viz_frame_id += 1
                    rr.log(f"{prefix}/left", rr.Image(left_img).compress(jpeg_quality=80))
                    rr.log(f"{prefix}/world/rig", rr.Transform3D(
                        translation=vis_trans, rotation=rr.Quaternion(xyzw=quat)))

                    # 坐标轴
                    qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                    rot = Rotation.from_quat([qx, qy, qz, qw])
                    axis_len = 0.1
                    rr.log(f"{prefix}/world/axes",
                           rr.Arrows3D(origins=[vis_trans] * 3,
                                       vectors=[rot.apply([axis_len, 0, 0]),
                                                rot.apply([0, axis_len, 0]),
                                                rot.apply([0, 0, axis_len])],
                                       colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]))

                    # 轨迹线（每 10 帧更新一次）
                    if len(trajectories[cid]) > 1 and viz_frame_id - last_traj_update[cid] >= 10:
                        rr.log(f"{prefix}/world/trajectory",
                               rr.LineStrips3D([np.array(trajectories[cid])]))
                        last_traj_update[cid] = viz_frame_id

                    # 特征点
                    if observations:
                        pts = [[u, v] for (u, v, _) in observations]
                        cols = [color_from_id(oid) for (_, _, oid) in observations]
                        if pts:
                            rr.log(f"{prefix}/left/observations", rr.Points2D(pts, radii=4, colors=cols))
                time.sleep(0.001)
            else:
                time.sleep(0.1)

            # 内存看门狗
            now = time.time()
            if now - last_mem_check >= MEMORY_CHECK_INTERVAL:
                last_mem_check = now
                if check_system_memory():
                    shutdown_reason = "CRITICAL: 系统内存耗尽，自动终止"
                    shutdown_processes(processes, shutdown_reason)
                    break

            # 定期 GC（无可视化时更频繁）
            if not enable_viz and now - last_gc >= 30:
                gc.collect()
                last_gc = now
                print(f"主进程 RSS={process_rss_mb()}MB, 系统={memory_status_str()}")
    except KeyboardInterrupt:
        print("\nCtrl+C 检测到...")
        shutdown_reason = "用户中断"
        shutdown_processes(processes)

    save_trajectories(traj_queue)
    if shutdown_reason:
        print(f"退出原因: {shutdown_reason}")

if __name__ == "__main__":
    main()
