# CLAUDE.md — cuVSLAM

NVIDIA CUDA-accelerated Visual SLAM library (v15.0.0). C++ engine with Python bindings (nanobind).

## Build

```bash
# C++ build
mkdir build && cd build && cmake .. && make -j$(nproc)

# Python package (requires C++ build first)
CUVSLAM_BUILD_DIR=<build_dir> pip install -e python/
```

## Architecture

- `libs/` — 25 C++ sub-libraries: `odometry/`, `slam/`, `sof/` (frontend), `cuda_modules/` (GPU kernels), `cuvslam/` (public API), `pipelines/`, `imu/`, `map/`
- `python/` — nanobind bindings (`cuvslam2.cpp`), high-level `Tracker` wrapper (`tracker.py`)
- `examples/` — dataset runners (EuRoC, KITTI, TUM) and live camera runners (RealSense, OAK-D, ZED, Orbbec)
- `tools/` — CLI utilities

## Python API — Key Types

```python
import cuvslam as vslam

# Core types: Pose, Camera, Rig, ImuCalibration, ImuMeasurement, Distortion
# Tracker wraps Odometry + Slam
tracker = vslam.Tracker(rig, odom_config, slam_config)
pose_est, slam_pose = tracker.track(timestamp_ns, (left_img, right_img))
tracker.register_imu_measurement(0, imu_meas)
```

Coordinate system: right-hand, X-right, Y-down, Z-forward.

## Memory-Sensitive Configuration (Python binding gotchas)

**`SlamConfig.enable_reading_internals` defaults to `True` in Python** (`python/cuvslam2.cpp:633`) even though C++ default is `False`. This allocates 7+ extra data structures per SLAM instance. Only set `True` if calling `get_pose_graph()`, `get_slam_landmarks()`, etc.

When SLAM is enabled, `Tracker.__init__` forces `enable_observations_export=True` and `enable_landmarks_export=True` (`python/tracker.py:66-68`).

Key memory knobs for `SlamConfig`:
- `max_map_size` (300) — pose graph node cap; 0 = unlimited
- `max_landmarks_distance` (100.0) — meters; reduce for indoor scenes
- `map_cell_size` (0.0 = auto from baseline) — larger = fewer spatial index cells
- `enable_reading_internals` (True in Python!) — disable unless needed

`OdometryConfig.enable_final_landmarks_export` accumulates all landmarks across all frames — only enable if calling `get_final_landmarks()`.

## Testing

```bash
# Python tests
python3 -m unittest discover -v -s python/test --locals

# C++ tests
cd build && GTEST_FILTER=-*SpeedUp* ctest --output-on-failure
```

Pre-commit: `clang-format`, license headers, trailing whitespace, no-commit-to-branch (master).

## User's Setup (this workspace)

- OAK-D cameras via USB 3.0 hub, running `examples/oak-d/run_multi_vio.py`
- 16GB RAM system — memory is tight with 4 cameras
- `examples/oak-d/monitor.py` — psutil-based CPU/GPU/memory monitor for benchmarking
- ROS2 optional (`rclpy`), Rerun for visualization
- Viz and HDF5 saving controlled via `ENABLE_VIZ` / `ENABLE_HDF5` env vars

## Code Style

- No comments for obvious code; short comments only for non-obvious WHY
- Edit existing files, don't create new ones unless needed
- No backwards-compatibility shims or dead code
