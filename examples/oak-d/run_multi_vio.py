#!/usr/bin/env python3
"""
Multi OAK-D VIO – Stable main-process visualization.
Use environment vars to override defaults:
  ENABLE_VIZ=0   -> disable visualization
  ENABLE_HDF5=0  -> disable trajectory saving
"""

import os, sys, time, signal, gc
from datetime import timedelta
from multiprocessing import Process, Queue, set_start_method
from collections import deque

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation
import cuvslam as vslam
import h5py
import psutil
import rerun as rr
import rerun.blueprint as rrb

# ---------- ROS2 (optional) ----------
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from builtin_interfaces.msg import Time
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ==========  Configuration ==========
# 默认值：可视化开启，HDF5保存开启
# 可以通过修改下面两行为 False 来改变默认行为
DEFAULT_ENABLE_VIZ = False
DEFAULT_ENABLE_HDF5 = False

# 环境变量覆盖默认值（终端传参方式：ENABLE_VIZ=0 python3 run_multi_vio.py）
ENABLE_VISUALIZATION = os.environ.get("ENABLE_VIZ", str(int(DEFAULT_ENABLE_VIZ))) == "1"
ENABLE_HDF5 = os.environ.get("ENABLE_HDF5", str(int(DEFAULT_ENABLE_HDF5))) == "1"

FPS = 30
#RESOLUTION = (1280, 720)
RESOLUTION = (640, 480)
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6
IMU_JITTER_THRESHOLD_NS = 6 * 1e6
IMU_FREQUENCY = 200
BORDER_TOP, BORDER_BOTTOM, BORDER_LEFT, BORDER_RIGHT = 15, 15, 15, 15
CM_TO_METERS = 100

IMU_GYROSCOPE_NOISE_DENSITY = 6.0673370376614875e-03 * 1
IMU_GYROSCOPE_RANDOM_WALK = 3.6211951458325785e-05 * 1
IMU_ACCELEROMETER_NOISE_DENSITY = 3.3621979208052800e-02 * 1
IMU_ACCELEROMETER_RANDOM_WALK = 9.8256589971851467e-04 * 1

# ----------  memory watchdog ----------
MEMORY_CRITICAL = 0.92       # auto-terminate at 92% system RAM
MEMORY_CHECK_INTERVAL = 2.0  # seconds between memory checks
VIZ_FRAME_SKIP = 3           # send every Nth frame to vis queue

# ----------  coordinate system ----------
# cuVSLAM uses OpenCV convention: X-right, Y-down, Z-forward
# Set to "ros" for X-forward, Y-left, Z-up (matching OAK official convention)
POSE_CONVENTION = "ros"

# Per-camera mounting orientation: "normal" (right-side up) or "inverted" (180° around Z)
# Keyed by logical camera ID (cid). Empty = all normal.
CAMERA_MOUNTING = {41: "inverted", 46:"inverted"}

# Map MXID (hardware serial) → logical camera ID, so the same physical camera
# always gets the same cid regardless of USB port. Run once to see MXIDs, then
# fill in. Empty = auto-assign by enumeration order.
# Example: {"1944301091C0187E00": 0, "19443010A1AC187E00": 1}
CAMERA_ID_MAP = {
    "14442C1091AFD5D200": 29,   # 右下
    "19443010A1AC187E00": 40,   # 左下
    "19443010D1061C7E00": 41,   # 左上
    "1944301091C0187E00": 46,   # 右上
    }


def resolve_camera_id(mxid, fallback):
    """Return the user-assigned logical camera ID for a given MXID, or fallback."""
    if mxid in CAMERA_ID_MAP:
        return CAMERA_ID_MAP[mxid]
    print(f"Warning: MXID {mxid} not in CAMERA_ID_MAP, using fallback cid={fallback}")
    return fallback

# Change-of-basis: cuVSLAM (X-right, Y-down, Z-forward) → ROS (X-forward, Y-left, Z-up)
_R_CUVSLAM_TO_ROS = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
_Q_CUVSLAM_TO_ROS = Rotation.from_matrix(_R_CUVSLAM_TO_ROS).as_quat()  # [x, y, z, w]

# ==========  Helper Functions ==========
def oak_transform_to_pose(extr):
    arr = np.array(extr)
    rot = arr[:3,:3]; t = arr[:3,3] / CM_TO_METERS
    return vslam.Pose(rotation=Rotation.from_matrix(rot).as_quat(), translation=t)

def set_cuvslam_camera(params):
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, params["distortion"])
    cam.focal = (params["intrinsics"][0][0], params["intrinsics"][1][1])
    cam.principal = (params["intrinsics"][0][2], params["intrinsics"][1][2])
    cam.size = params["resolution"]
    cam.rig_from_camera = oak_transform_to_pose(params["extrinsics"])
    cam.border_top = BORDER_TOP; cam.border_bottom = BORDER_BOTTOM
    cam.border_left = BORDER_LEFT; cam.border_right = BORDER_RIGHT
    return cam

def get_stereo_calib(calib, res):
    s = {"left":{},"right":{}}
    for side, sock in zip(["left","right"], [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]):
        s[side]["resolution"] = res
        s[side]["intrinsics"] = calib.getCameraIntrinsics(sock, res[0], res[1])
        s[side]["extrinsics"] = calib.getCameraExtrinsics(sock, dai.CameraBoardSocket.CAM_A)
        s[side]["distortion"] = calib.getDistortionCoefficients(sock)[:8]
    return s

def get_imu_calib(extr):
    # DepthAI firmware pre-rotates IMU readings (ACCELEROMETER_RAW) into the
    # camera coordinate frame (X-right, Y-down, Z-forward). The EEPROM extrinsics
    # matrix contains the physical chip rotation — applying it would double-rotate.
    # So we use identity rotation and keep only the translation offset.
    imu = vslam.ImuCalibration()
    t = np.array(extr)[:3,3] / CM_TO_METERS
    imu.rig_from_imu = vslam.Pose(rotation=np.array([0.,0.,0.,1.]), translation=t)
    imu.gyroscope_noise_density = IMU_GYROSCOPE_NOISE_DENSITY
    imu.gyroscope_random_walk = IMU_GYROSCOPE_RANDOM_WALK
    imu.accelerometer_noise_density = IMU_ACCELEROMETER_NOISE_DENSITY
    imu.accelerometer_random_walk = IMU_ACCELEROMETER_RANDOM_WALK
    imu.frequency = IMU_FREQUENCY
    return imu

def ts_to_ns(ts):
    if ts is None: return None
    if hasattr(ts, 'total_seconds'): return int(ts.total_seconds()*1e9)
    if hasattr(ts, 'seconds') and hasattr(ts, 'microseconds'): return int(ts.seconds*1e9 + ts.microseconds*1000)
    if isinstance(ts, (int,float)): return int(ts*1e9)
    try: return int(ts.total_seconds()*1e9)
    except: return None

def color_from_id(uid): return [(uid*17)%256, (uid*31)%256, (uid*47)%256]

def memory_status_str():
    mem = psutil.virtual_memory()
    return f"{mem.percent:.1f}% ({mem.used // (1024**3)}GB / {mem.total // (1024**3)}GB)"

def check_system_memory():
    return psutil.virtual_memory().percent >= MEMORY_CRITICAL * 100

def process_rss_mb():
    return psutil.Process().memory_info().rss // (1024**2)

def transform_pose_to_ros(trans, quat_xyzw):
    """Convert pose from cuVSLAM (X-right, Y-down, Z-forward) to ROS (X-forward, Y-left, Z-up)."""
    trans_ros = np.array([trans[2], -trans[0], -trans[1]])
    q_cb = Rotation.from_quat([_Q_CUVSLAM_TO_ROS[0], _Q_CUVSLAM_TO_ROS[1],
                                _Q_CUVSLAM_TO_ROS[2], _Q_CUVSLAM_TO_ROS[3]])
    q_cuv = Rotation.from_quat([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]])
    q_ros = (q_cb * q_cuv * q_cb.inv()).as_quat()  # [x, y, z, w]
    return trans_ros, np.array([q_ros[0], q_ros[1], q_ros[2], q_ros[3]])

# ==========  ROS2 Publisher ==========
class CameraPosePublisher(Node):
    def __init__(self, cid):
        super().__init__(f'vio_camera_{cid}')
        self.pub = self.create_publisher(PoseStamped, f'cam_{cid}/pose', 10)
        self.cid = cid
    def publish_pose(self, ts_ns, trans, quat_xyzw):
        msg = PoseStamped()
        sec, nsec = int(ts_ns//1e9), int(ts_ns%1e9)
        stamp = Time(); stamp.sec=sec; stamp.nanosec=nsec
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

# ==========  VIO Process (NO RERUN, sends data via Queue) ==========
def vio_process(camera_id, device_id, num_cameras, vis_queue, traj_queue, enable_viz):
    if ROS2_AVAILABLE:
        rclpy.init(args=None)
        ros_node = CameraPosePublisher(camera_id)
    else:
        ros_node = None

    infos = dai.Device.getAllAvailableDevices()
    target = next((d for d in infos if d.deviceId == device_id), None)
    if target is None:
        print(f"[Cam {camera_id}] Device {device_id} not found.")
        traj_queue.put((camera_id, []))
        return

    device = dai.Device(target)
    mounting = CAMERA_MOUNTING.get(camera_id, "normal")
    print(f"[Cam {camera_id}] mount={mounting}, convention={'Z-up' if POSE_CONVENTION == 'ros' else 'Z-fwd'}")
    calib = device.readCalibration()
    stereo = get_stereo_calib(calib, RESOLUTION)
    cams = [set_cuvslam_camera(stereo["left"]), set_cuvslam_camera(stereo["right"])]
    rig = vslam.Rig(cams)
    imu_extr = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A)
    rig.imus = [get_imu_calib(imu_extr)]

    odom_cfg = vslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_final_landmarks_export=False,
        enable_observations_export=True,
        rectified_stereo_camera=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial
    )
    slam_cfg = vslam.Tracker.SlamConfig(
        enable_reading_internals=False,
        max_landmarks_distance=30.0,
        throttling_time_ms=1000,
    )
    tracker = vslam.Tracker(rig, odom_cfg, slam_config=slam_cfg)

    pipeline = dai.Pipeline(device)
    lcam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=FPS)
    rcam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=FPS)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5/FPS))
    lo = lcam.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    ro = rcam.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    lo.link(sync.inputs["left"]); ro.link(sync.inputs["right"])
    sq = sync.out.createOutputQueue()

    imu_node = pipeline.create(dai.node.IMU)
    imu_node.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], IMU_FREQUENCY)
    imu_node.setBatchReportThreshold(1); imu_node.setMaxBatchReports(10)
    iq = imu_node.out.createOutputQueue()

    pipeline.start()

    frame_id = 0
    prev_cts = None; last_imu_ts = None; last_sent_ts = None
    imu_buf = []; imu_count = 0
    traj_full = []

    try:
        while pipeline.isRunning():
            mg = sq.get()
            lf, rf = mg["left"], mg["right"]
            ts_dev = None
            for m in ('getTimestampDevice','getTimestamp'):
                if hasattr(mg, m):
                    ts_dev = getattr(mg, m)()
                    break
            if ts_dev is None: continue
            ts_ns = ts_to_ns(ts_dev)
            if ts_ns is None: continue

            ims = iq.tryGetAll()
            for im in ims:
                for pkt in im.packets:
                    accel = pkt.acceleroMeter; gyro = pkt.gyroscope
                    ts = None
                    for m in ('getTimestampDevice','getTimestamp','getTimestampHost'):
                        if hasattr(pkt, m): ts = getattr(pkt, m)(); break
                    if ts is None:
                        for sf in ('acceleroMeter','gyroscope'):
                            if hasattr(pkt, sf):
                                so = getattr(pkt, sf)
                                for a in ('tsDevice','timestamp','timestampDevice'):
                                    if hasattr(so, a): ts = getattr(so, a); break
                            if ts is not None: break
                    if ts is None: continue
                    its = ts_to_ns(ts)
                    if its is None: continue
                    imu_buf.append((its, accel, gyro))
            imu_buf.sort(key=lambda x: x[0])

            while imu_buf and imu_buf[0][0] <= ts_ns:
                its, accel, gyro = imu_buf.pop(0)
                if last_sent_ts is not None and its <= last_sent_ts: continue
                if last_imu_ts is not None and (its - last_imu_ts) <= 0: continue
                last_imu_ts = its
                meas = vslam.ImuMeasurement()
                meas.timestamp_ns = its
                meas.linear_accelerations = np.array([accel.x, accel.y, accel.z], dtype=np.float32)
                meas.angular_velocities = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)
                tracker.register_imu_measurement(0, meas)
                imu_count += 1
                last_sent_ts = its

            prev_cts = ts_ns; imu_count = 0; frame_id += 1

            if frame_id <= WARMUP_FRAMES: continue

            left_img = lf.getCvFrame(); right_img = rf.getCvFrame()
            odom_est, slam_est = tracker.track(ts_ns, (left_img, right_img))
            pose = slam_est if slam_est is not None else odom_est.world_from_rig
            if pose is None:
                print(f"[Cam {camera_id}] Tracking failed at {frame_id}")
                continue

            trans = pose.translation.copy(); quat = pose.rotation.copy()
            if POSE_CONVENTION == "ros":
                trans, quat = transform_pose_to_ros(trans, quat)
            traj_full.append((ts_ns, trans, quat))

            # 序列化特征点
            raw_obs = tracker.get_last_observations(0)
            obs_list = [(o.u, o.v, o.id) for o in raw_obs] if raw_obs else []

            if ros_node:
                ros_node.publish_pose(ts_ns, trans, quat)
                rclpy.spin_once(ros_node, timeout_sec=0)

            # 定时打印位姿（每 60 帧打印一次）
            if frame_id % 60 == 0:
                print(f"[Cam {camera_id}] pos: ({trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}) "
                      f"quat: ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

            # 定期 GC + 内存日志（每 300 帧）
            if frame_id % 300 == 0:
                gc.collect()
                print(f"[Cam {camera_id}] RSS: {process_rss_mb()}MB")

            # 仅在启用可视化且队列有效时发送数据（跳帧发送，降低队列压力）
            if enable_viz and vis_queue is not None and frame_id % VIZ_FRAME_SKIP == 0:
                vis_queue.put({
                    'camera_id': camera_id,
                    'timestamp_ns': ts_ns,
                    'left_image': left_img,
                    'translation': trans,
                    'rotation': quat,
                    'observations': obs_list
                })
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        device.close()
        if ros_node:
            try:
                ros_node.destroy_node()
            except:
                pass
            try:
                rclpy.shutdown()
            except:
                pass
        traj_queue.put((camera_id, traj_full))
        print(f"[Camera {camera_id}] Finished. {len(traj_full)} poses.")


def shutdown_processes(processes, reason=""):
    """Gracefully shutdown all VIO processes — shared by Ctrl+C and memory watchdog."""
    if reason:
        print(f"\n!!! {reason} !!!")
    print(f"System memory: {memory_status_str()}")
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
            _, _ = traj_queue.get()
        print("HDF5 writing disabled. Trajectories not saved.")
        return
    all_traj = {}
    while not traj_queue.empty():
        cid, data = traj_queue.get()
        all_traj[cid] = data
    if not all_traj:
        print("Warning: No trajectory data received, nothing saved.")
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
            print(f"Saved cam_{cid}: {len(timestamps)} poses.")
    print("All trajectories saved to multi_oak_vio.h5")


# ==========  Main Process ==========
def main():
    set_start_method('spawn', force=True)

    # ---------- 可视化开关 ----------
    enable_viz = ENABLE_VISUALIZATION
    print(f"Visualization: {'ON' if enable_viz else 'OFF'}")
    print(f"HDF5 saving: {'ON' if ENABLE_HDF5 else 'OFF'}")
    print(f"Pose convention: {'Z-up (ROS)' if POSE_CONVENTION == 'ros' else 'Z-fwd (cuVSLAM/OpenCV)'}")
    print(f"Camera mounting: {CAMERA_MOUNTING if CAMERA_MOUNTING else 'all normal'}")

    if enable_viz:
        rr.init("multi_oak_vio", spawn=True)
        time.sleep(2)

    # ---------- 获取所有 OAK-D 设备 ----------
    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        print("No OAK-D devices found.")
        return

    # Resolve stable logical camera IDs from MXID map
    convention_str = "X-fwd Y-left Z-up (ROS)" if POSE_CONVENTION == "ros" else "X-right Y-down Z-fwd (cuVSLAM)"
    device_assignments = []  # (cid, mxid, name, mounting)
    print(f"Found {len(infos)} device(s):")
    for i, info in enumerate(infos):
        cid = resolve_camera_id(info.deviceId, i)
        mounting = CAMERA_MOUNTING.get(cid, "normal")
        device_assignments.append((cid, info.deviceId, info.name, mounting))
        map_mark = "*" if info.deviceId in CAMERA_ID_MAP else " "
        print(f"  [{map_mark}] cid={cid} mxid={info.deviceId} name={info.name} mount={mounting}")

    if not CAMERA_ID_MAP:
        print("  Hint: set CAMERA_ID_MAP = {mxid: cid, ...} for stable IDs across USB ports.")
    print(f"  Frame convention: {convention_str}")

    num_cameras = len(device_assignments)

    # ---------- 蓝图 ----------
    if enable_viz:
        cam_views = []
        for cid, _, _, _ in device_assignments:
            cam_views.append(rrb.Spatial2DView(origin=f"cam_{cid}/left", name=f"cam{cid}-left"))
        left_col = rrb.Vertical(*cam_views)
        right_3d = rrb.Spatial3DView(name="3D")
        blueprint = rrb.Blueprint(rrb.Horizontal(left_col, right_3d))
        rr.send_blueprint(blueprint)

    # ---------- 轨迹偏移（虚拟起点）----------
    offsets = {}
    sorted_cids = sorted(cid for cid, _, _, _ in device_assignments)
    if num_cameras == 1:
        offsets[sorted_cids[0]] = np.zeros(3)
    elif num_cameras == 2:
        offsets[sorted_cids[0]] = np.array([-0.5, 0.0, 0.0])
        offsets[sorted_cids[1]] = np.array([ 0.5, 0.0, 0.0])
    elif num_cameras == 3:
        offsets[sorted_cids[0]] = np.array([-0.5, -0.5, 0])
        offsets[sorted_cids[1]] = np.array([ 0.5, -0.5, 0])
        offsets[sorted_cids[2]] = np.array([ 0.0,  0.5, 0])
    else:
        offsets[sorted_cids[0]] = np.array([-0.5, -0.5, 0])
        offsets[sorted_cids[1]] = np.array([ 0.5, -0.5, 0])
        offsets[sorted_cids[2]] = np.array([-0.5,  0.5, 0])
        offsets[sorted_cids[3]] = np.array([ 0.5,  0.5, 0])

    # ---------- 队列 ----------
    vis_queue = Queue() if enable_viz else None
    traj_queue = Queue()
    processes = []

    # ---------- 启动子进程 ----------
    try:
        vslam.warm_up_gpu()
        print("GPU warmup done.")
    except Exception as e:
        print(f"GPU warmup skipped ({e})")
    for cid, mxid, _, mounting in device_assignments:
        p = Process(target=vio_process,
                    args=(cid, mxid, num_cameras,
                          vis_queue, traj_queue, enable_viz))
        p.start()
        processes.append(p)

    # ---------- 主循环 ----------
    shutdown_reason = ""
    try:
        if enable_viz:
            from collections import deque
            max_traj_len = 2000
            trajectories = {cid: deque(maxlen=max_traj_len) for cid in sorted_cids}
            viz_frame_id = 0
            last_traj_update = {cid: 0 for cid in sorted_cids}
            last_mem_check = time.time()

            while any(p.is_alive() for p in processes):
                while not vis_queue.empty():
                    data = vis_queue.get()
                    cid = data['camera_id']
                    ts_ns = data['timestamp_ns']
                    left_img = data['left_image']
                    trans = data['translation']
                    quat = data['rotation']
                    observations = data['observations']

                    offset = offsets.get(cid, np.zeros(3))
                    vis_trans = trans + offset
                    trajectories[cid].append(vis_trans)

                    rr.set_time_sequence("frame", viz_frame_id)
                    viz_frame_id += 1

                    prefix = f"cam_{cid}"
                    # 图像
                    rr.log(f"{prefix}/left", rr.Image(left_img).compress(jpeg_quality=80))
                    # 位姿
                    rr.log(f"{prefix}/world/rig", rr.Transform3D(
                        translation=vis_trans,
                        rotation=rr.Quaternion(xyzw=quat)
                    ))
                    # 坐标轴
                    qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                    rot = Rotation.from_quat([qx, qy, qz, qw])
                    axis_len = 0.1
                    axis_x = rot.apply([axis_len, 0, 0])
                    axis_y = rot.apply([0, axis_len, 0])
                    axis_z = rot.apply([0, 0, axis_len])
                    rr.log(f"{prefix}/world/axes",
                           rr.Arrows3D(origins=[vis_trans]*3,
                                       vectors=[axis_x, axis_y, axis_z],
                                       colors=[(255,0,0), (0,255,0), (0,0,255)]))
                    # 轨迹
                    if len(trajectories[cid]) > 1 and viz_frame_id - last_traj_update[cid] >= 10:
                        traj_np = np.array(trajectories[cid])
                        rr.log(f"{prefix}/world/trajectory", rr.LineStrips3D([traj_np]))
                        last_traj_update[cid] = viz_frame_id
                    # 特征点
                    if observations:
                        pts = [[u, v] for (u, v, _) in observations]
                        cols = [color_from_id(oid) for (_, _, oid) in observations]
                        if pts:
                            rr.log(f"{prefix}/left/observations",
                                   rr.Points2D(pts, radii=4, colors=cols))
                time.sleep(0.001)

                # ---------- 内存看门狗 ----------
                now = time.time()
                if now - last_mem_check >= MEMORY_CHECK_INTERVAL:
                    last_mem_check = now
                    if check_system_memory():
                        shutdown_reason = "CRITICAL: System memory exhausted, auto-terminating"
                        shutdown_processes(processes, shutdown_reason)
                        break
        else:
            print("Visualization disabled. Waiting for VIO processes...")
            last_mem_check = time.time()
            last_gc = time.time()
            while any(p.is_alive() for p in processes):
                time.sleep(0.1)

                now = time.time()
                if now - last_mem_check >= MEMORY_CHECK_INTERVAL:
                    last_mem_check = now
                    if check_system_memory():
                        shutdown_reason = "CRITICAL: System memory exhausted, auto-terminating"
                        shutdown_processes(processes, shutdown_reason)
                        break
                # 定期 GC（每 30 秒）
                if now - last_gc >= 30:
                    gc.collect()
                    last_gc = now
                    print(f"Main RSS: {process_rss_mb()}MB, system: {memory_status_str()}")
    except KeyboardInterrupt:
        print("\nCtrl+C detected...")
        shutdown_reason = "user interrupt"
        shutdown_processes(processes)

    # ---------- 保存 HDF5 ----------
    save_trajectories(traj_queue)
    if shutdown_reason:
        print(f"Exit reason: {shutdown_reason}")


if __name__ == "__main__":
    main()