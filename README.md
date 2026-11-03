# LF-3DGS

单光场相机上的粒子重建与测速：用可微三维高斯溅射做逐帧拟合（Stage 1），再把同一套表示摊还成前馈网络（Stage 2）。检测用冻结的匈牙利匹配，两帧之间用匈牙利追踪。

本仓库是这套方案的可运行核心，不包含训练权重、合成数据集或合作方原始图像。

## 方案

```text
子孔径图像
    │
    ├─ Stage 1  逐帧优化 3D Gaussian（GaussianSplatRenderer + optimise_sample）
    │
    └─ Stage 2  LF3DGSNetV3 前馈检测（代价体、峰值提取、top-k）
            │
            ▼
      匈牙利匹配（半径 2 体素，NMS 1.5，α = 0.05）
            │
            ▼
      两帧匈牙利追踪 → 位移 / 速度
```

| 模块 | 文件 | 作用 |
|---|---|---|
| 相机 | `lf_3dgs/camera.py` | 解析光场投影，物理坐标与体素坐标互转 |
| Stage 1 渲染 | `lf_3dgs/renderer.py` | 可微高斯溅射 |
| Stage 1 优化 | `lf_3dgs/optimizer.py` | 单帧 `optimise_sample` |
| Stage 2 | `lf_3dgs/model_v3.py` | `LF3DGSNetV3` |
| 检测评估 | `lf_3dgs/nmi_sprint/evaluator.py` | `hungarian_match`，半径 2 体素 |
| 追踪 | `lf_3dgs/piv/piv_tracking.py` | `hungarian_tracking` |

Stage 2 的正式工作点是：在 `lightfield_dataset9` 上训练的 `p7_margin`，官方 test F1 0.870；同一批五十对流场上 RelErr_XY 25.8%。权重和数据不在本仓库。

## 环境

需要 Python 3.10+、PyTorch 2、NumPy、SciPy、PyYAML。CPU 即可完成下面的检查；正式训练和逐帧优化使用 GPU。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke check

在仓库根目录执行。不下载数据、不加载权重，大约几十秒。

```bash
python scripts/smoke_check.py
```

通过时最后一行是 `SMOKE_OK`，进程退出码为 0。它依次检查：

1. 解析投影给出有限的子孔径坐标，物理坐标能转成体素坐标。
2. Stage 1 渲染器对 4 个高斯、2 个视角输出非负有限图像，并且位置参数能反传。
3. Stage 2 在缩小的光场 `(1, 4, 4, 32, 32)` 上前向一次，抽出 8 个有限的物理中心。缩小只为让 CPU 检查跑完；正式网络的输出体是 `(64, 128, 128)`，输入是 186×232 的子孔径图。
4. 匈牙利匹配在半径内 F1 = 1，平移出半径后 TP = 0。
5. 两帧追踪找回已知位移。

## 正式尺寸

| 项 | 值 |
|---|---|
| 微透镜阵列 | 186 × 232，节距 0.099 mm |
| 每透镜角分辨率 | 22 × 22 |
| 重建体 | 64 × 128 × 128 |
| 物理范围 | x ±10 mm，y ±8 mm，z ±5 mm |
| 检测 | 匈牙利半径 2 体素，NMS 1.5 体素，α = 0.05 |
| Stage 2 top-k | 1500（p7） |

逐帧优化示例（需要一份含 `sai_stack` 的 `.npz` 和 GPU）：

```bash
python -m lf_3dgs.optimizer --sample path/to/sample.npz --output output/opt --device cuda
```

## 不包含

- 训练好的 `p7_margin` 权重
- `lightfield_dataset9` 及后续合成集
- 合作方水槽、气流原始 TIFF
- 论文稿与实验日志
