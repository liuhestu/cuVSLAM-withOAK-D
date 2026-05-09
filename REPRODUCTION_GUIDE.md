# cuVSLAM 复现指南 / Reproduction Guide

本文档系统性地说明复现 cuVSLAM 项目所需的 **硬件要求**、**软件环境** 以及 **实现步骤**。

---

## 目录

- [1. 硬件要求](#1-硬件要求)
  - [1.1 GPU](#11-gpu)
  - [1.2 CPU / 平台](#12-cpu--平台)
  - [1.3 相机与传感器（实时场景）](#13-相机与传感器实时场景)
  - [1.4 其他外设](#14-其他外设)
- [2. 软件环境](#2-软件环境)
  - [2.1 操作系统](#21-操作系统)
  - [2.2 NVIDIA 驱动与 CUDA](#22-nvidia-驱动与-cuda)
  - [2.3 编译工具链](#23-编译工具链)
  - [2.4 Python 环境](#24-python-环境)
  - [2.5 Docker 环境（可选）](#25-docker-环境可选)
- [3. 实现步骤](#3-实现步骤)
  - [方式 A：预编译 Wheel 安装（最快）](#方式-a预编译-wheel-安装最快)
  - [方式 B：从源码编译 C++ 库](#方式-b从源码编译-c-库)
  - [方式 C：从源码安装 PyCuVSLAM](#方式-c从源码安装-pycuvslam)
  - [方式 D：Docker 容器运行](#方式-ddocker-容器运行)
  - [方式 E：Jetson（ARM）远程编译](#方式-ejetsonarm远程编译)
- [4. 运行示例](#4-运行示例)
  - [4.1 公开数据集示例](#41-公开数据集示例)
  - [4.2 实时相机示例](#42-实时相机示例)
- [5. 常见问题与注意事项](#5-常见问题与注意事项)

---

## 1. 硬件要求

### 1.1 GPU

| 项目 | 说明 |
|------|------|
| **GPU 品牌** | 必须是 **NVIDIA GPU**（支持 CUDA） |
| **CUDA 计算能力** | 需要支持 CUDA 12 或 CUDA 13 的 GPU |
| **显存** | 无严格最低要求，但建议 ≥ 4 GB（多相机/高分辨率场景建议 ≥ 8 GB） |
| **桌面/工作站** | GeForce RTX 30/40/50 系列、RTX A 系列、Quadro 等均可 |
| **嵌入式平台** | NVIDIA Jetson Orin / Jetson Thor（JetPack 6.1/6.2/7.0/7.1） |

### 1.2 CPU / 平台

| 架构 | 典型平台 | 说明 |
|------|----------|------|
| **x86_64** | 普通台式机 / 笔记本 / 服务器 | 主要开发和测试平台 |
| **aarch64** | NVIDIA Jetson Orin / Thor | 嵌入式部署场景 |

### 1.3 相机与传感器（实时场景）

> **注意**：如果仅使用公开数据集（EuRoC、KITTI、TUM 等）复现，则 **不需要** 额外相机硬件。

以下相机已在示例中验证支持：

| 相机 | 支持的模式 | 备注 |
|------|-----------|------|
| **Intel RealSense** (D435i 等) | 单目深度、双目、双目惯性、多相机 | 完整 Docker 支持，有多相机组装指南 |
| **Stereolabs ZED** (ZED 2/2i/Mini 等) | 单目深度、双目 | 支持 SVO2 录制与离线回放 |
| **Luxonis OAK-D** | 双目 | 支持畸变图像输入 |
| **Orbbec** | 单目深度、双目 | — |

**相机要求**：
- 分辨率建议 **VGA (640×480) 或更高**
- 帧率建议 **30 FPS**（适合正常运动速度，快速运动需提高帧率）
- 需要准确的 **内参标定 (Intrinsic Calibration)**
- 多相机设置需要准确的 **外参标定 (Extrinsic Calibration)**
- 多相机需要 **硬件同步**

### 1.4 其他外设

| 外设 | 用途 | 是否必须 |
|------|------|----------|
| IMU（惯性测量单元） | 双目惯性里程计 (Stereo-Inertial Odometry) | 仅在使用惯性模式时需要（RealSense D435i 自带） |
| USB 3.0 接口 | 连接深度相机 | 实时相机场景需要 |
| 显示器 + X11 | Rerun 可视化界面 | 可选，用于可视化结果 |

---

## 2. 软件环境

### 2.1 操作系统

| 操作系统 | 测试状态 |
|----------|----------|
| **Ubuntu 22.04 LTS** (x86_64 / aarch64) | ✅ 已验证 |
| **Ubuntu 24.04 LTS** (x86_64 / aarch64) | ✅ 已验证 |
| Ubuntu 22+ 其他版本 | 应该可以工作 |

### 2.2 NVIDIA 驱动与 CUDA

| 组件 | 版本要求 |
|------|----------|
| **CUDA Toolkit** | **CUDA 12** 或 **CUDA 13** |
| **NVIDIA 驱动**（CUDA 12.6 对应） | ≥ **560** |
| **NVIDIA 驱动**（CUDA 13.0 对应） | ≥ **580** |
| **JetPack**（Jetson 平台） | 6.1 / 6.2 / 7.0 / 7.1 |

验证驱动版本：
```bash
nvidia-smi
```

验证 CUDA 版本：
```bash
nvcc --version
```

### 2.3 编译工具链

从源码编译需安装以下工具：

```bash
sudo apt update && sudo apt install -y g++ cmake git git-lfs python3-dev
```

| 工具 | 最低版本 |
|------|----------|
| `g++` | 系统默认即可 |
| `cmake` | ≥ 3.19 |
| `git` + `git-lfs` | 克隆仓库需要（含 LFS 大文件） |
| `python3-dev` | Python 绑定所需 |

### 2.4 Python 环境

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.9（源码编译）；预编译 Wheel：Ubuntu 22.04 → 3.10，Ubuntu 24.04 → 3.12+ |
| pip | 最新版本 |

示例运行需要的 Python 依赖（`examples/requirements.txt`）：
```
numpy==2.2.4
pillow==11.1.0
pyyaml==6.0.2
rerun-sdk==0.22.1
scipy==1.14.1
```

### 2.5 Docker 环境（可选）

如果使用 Docker 方式，需要：

| 组件 | 说明 |
|------|------|
| Docker Engine | 最新版本 |
| NVIDIA Container Toolkit | `nvidia-docker2` 或 `nvidia-container-toolkit` |
| NVIDIA 驱动 | 满足上述版本要求 |

---

## 3. 实现步骤

### 方式 A：预编译 Wheel 安装（最快）

适用场景：快速上手运行 Python 示例与数据集。

```bash
# 1. 安装 CUDA Toolkit（若未安装）
#    参考：https://developer.nvidia.com/cuda/toolkit

# 2. 克隆仓库（需要 git-lfs）
git clone https://github.com/nvidia-isaac/cuVSLAM.git
cd cuVSLAM

# 3. 创建 Python 虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 4. 从 GitHub Releases 下载 Wheel
#    https://github.com/nvidia-isaac/cuVSLAM/releases
#    选择匹配的 CUDA 版本 (cu12/cu13)、Python 版本和架构 (x86_64/aarch64)

# 5. 安装 Wheel
pip install cuvslam-*.whl

# 6. 安装示例依赖
pip install -r examples/requirements.txt

# 7. 运行示例（以 EuRoC 单目为例）
python3 examples/euroc/run_mono.py
```

### 方式 B：从源码编译 C++ 库

适用场景：需要修改 C++ 代码、使用 C++ API 或构建工具。

```bash
# 1. 安装系统依赖
sudo apt update && sudo apt install -y g++ cmake git git-lfs python3-dev

# 2. 克隆仓库
git clone https://github.com/nvidia-isaac/cuVSLAM.git
cd cuVSLAM

# 3. 方法一：手动 CMake 编译
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. 方法二：使用构建脚本
export CUVSLAM_SRC_DIR=$(pwd)
export CUVSLAM_DST_DIR=$(pwd)/build
./build_release.sh
```

### 方式 C：从源码安装 PyCuVSLAM

适用场景：需要同时修改 C++ 和 Python 代码。

```bash
# 前提：已完成方式 B 的 C++ 编译

# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装 PyCuVSLAM（指定构建目录）
CUVSLAM_BUILD_DIR=<path-to-cuvslam-build> pip install python/

# 3. 安装示例依赖
pip install -r examples/requirements.txt

# 4. 验证安装
python3 -c "import cuvslam; print('PyCuVSLAM installed successfully')"
```

> **注意**：由于 scikit-build-core 的限制，重新编译 libcuvslam 后需要重新安装 PyCuVSLAM 绑定。

### 方式 D：Docker 容器运行

适用场景：快速搭建完整环境，特别适合 RealSense 相机用户。

```bash
# 1. 克隆仓库
git clone https://github.com/nvidia-isaac/cuVSLAM.git
cd cuVSLAM

# 2. 构建 Docker 镜像

# Ubuntu 22.04 + CUDA 12：
docker build -f docker/Dockerfile.realsense-cu12 -t pycuvslam:realsense-cu12 .

# 或 Ubuntu 24.04 + CUDA 13：
docker build -f docker/Dockerfile.realsense-cu13 -t pycuvslam:realsense-cu13 .

# 3. 运行容器

# CUDA 12（默认）：
./docker/run_docker.sh

# CUDA 13：
./docker/run_docker.sh 24

# 4. 在容器内运行示例（如 RealSense 双目）
python3 examples/realsense/run_stereo.py
```

### 方式 E：Jetson（ARM）远程编译

适用场景：在 Jetson Orin/Thor 嵌入式平台上部署。

```bash
# 在 x86 主机上操作：

# 1. 将源码复制到 Jetson
./copy_to_remote.sh <jetson-host>

# 2. SSH 到 Jetson 并编译
ssh <jetson-host> 'export CUVSLAM_SRC_DIR=~/cuvslam/src CUVSLAM_DST_DIR=~/cuvslam/build && ~/cuvslam/src/build_release.sh'

# 3. 将编译产物复制回主机
./copy_from_remote.sh <jetson-host>
```

---

## 4. 运行示例

### 4.1 公开数据集示例

以下数据集可以在无实体相机的情况下复现：

| 数据集 | 支持模式 | 示例路径 | 说明 |
|--------|---------|----------|------|
| **EuRoC MAV** | 单目 VO、双目 VIO | `examples/euroc/` | 常用基准数据集 |
| **KITTI** | 双目 VO、SLAM | `examples/kitti/` | 自动驾驶场景 |
| **TUM RGB-D** | 单目深度 VO | `examples/tum/` | 室内 RGB-D 场景 |
| **TartanAir / Tartan Ground** | 多相机 VO | `examples/multicamera_edex/` | 仿真多相机 |
| **R2B Galileo** | 多相机 VO | `examples/multicamera_edex/` | 机器人场景 |

每个数据集目录下均有详细的 `README.md`，说明数据下载和运行方式。

**快速运行 EuRoC 示例**：
```bash
# 按照 examples/euroc/README.md 下载数据集
# 然后运行：
python3 examples/euroc/run_mono.py
```

### 4.2 实时相机示例

| 相机 | 示例 | 命令 |
|------|------|------|
| RealSense | 双目 VO | `python3 examples/realsense/run_stereo.py` |
| RealSense | 单目深度 VO | `python3 examples/realsense/run_mono_depth.py` |
| RealSense | 双目惯性 VO | `python3 examples/realsense/run_stereo_imu.py` |
| RealSense | 多相机 VO | `python3 examples/realsense/run_multicamera.py` |
| ZED | 双目 VO | `python3 examples/zed/live/run_stereo.py` |
| OAK-D | 双目 VO | `python3 examples/oak-d/run_stereo.py` |
| Orbbec | 双目 VO | `python3 examples/orbbec/run_stereo.py` |

---

## 5. 常见问题与注意事项

### 性能相关

1. **避免硬件过载**：运行 cuVSLAM 时关闭不必要的可视化或图像保存操作，防止丢帧。
2. **标定精度**：内参和外参标定的精度直接影响追踪质量，务必使用精确标定。
3. **时间同步**：多相机需要硬件同步，确保各相机图像同时采集。
4. **帧率**：通常 30 FPS 足以应对人类速度的运动；快速运动需提高帧率。
5. **分辨率**：建议 VGA 及以上。cuVSLAM 利用 CUDA 加速，可高效处理高分辨率图像。
6. **运动模糊**：应使用较短曝光时间减少运动模糊。

### 环境相关

7. **git-lfs**：仓库包含大文件，克隆时必须先安装 `git-lfs`。
8. **scikit-build-core**：从源码安装 PyCuVSLAM 后，若重新编译了 libcuvslam，需重新 `pip install python/`。
9. **Jetson CUDA 版本**：在较旧的 JetPack（如 6.0 / CUDA 12.2）上使用 Docker 时，可能需要设置 `NVIDIA_DISABLE_REQUIRE=1`。

### 更多故障排除

详见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)，包含标定检查、EDEX 格式说明、Tracker 工具使用等详细指导。
