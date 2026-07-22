# SOP

## 1. 环境准备

进入项目目录并激活已经配置好的 conda 环境：

```powershell
cd D:\workplace\Mission3\Project
conda activate project
```

确认依赖可用：

```powershell
python -c "import cv2, pandas, sklearn, supervision; from ultralytics import YOLO; print('env ok')"
```

如需补装依赖：

```powershell
python -m pip install -r requirements.txt
```

`supervision` 建议使用本地源码可编辑安装：

```powershell
cd D:\workplace\Mission3\Project\supervision
python -m pip install -e .
cd D:\workplace\Mission3\Project
```

YOLO 权重统一放在：

```text
Project/checkpoints/yolov8x.pt
```

## 2. 准备视频和标定文件

测试视频：

```text
Project/data/vehicles.mp4
```

标定文件：

```text
Project/data/vehicles/vehicles.calibration.json
```

如果需要重新标定道路四边形：

```powershell
python source_calibration_tool/calibrate_source.py --source-video-path data\vehicles.mp4 --display-width 1280 --padding 300
```

## 3. 第一步：分析单个视频

`video_analyzer.py` 负责车辆检测、ID 分配、透视变换、速度/加速度估计、默认弹窗预览和原始轨迹 CSV 输出。

默认弹窗运行：

```powershell
python video_analyzer.py --source-video-path data\vehicles.mp4 --calibration-path data\vehicles\vehicles.calibration.json
```

不弹窗，只生成 CSV：

```powershell
python video_analyzer.py --source-video-path data\vehicles.mp4 --calibration-path data\vehicles\vehicles.calibration.json --no-show
```

同时保存标注视频：

```powershell
python video_analyzer.py --source-video-path data\vehicles.mp4 --calibration-path data\vehicles\vehicles.calibration.json --save-annotated-video
```

如果已知停止线在道路坐标中的 y 值，例如 `249`：

```powershell
python video_analyzer.py --source-video-path data\vehicles.mp4 --calibration-path data\vehicles\vehicles.calibration.json --stopline-road-y-m 249
```

输出目录会自动包含日期时间戳和原视频文件名：

```text
outputs/
└── 20260722_153012_vehicles/
    ├── vehicles_vehicle_tracks.csv
    ├── metadata.json
    └── vehicles_annotated.mp4
```

## 4. 第二步：清洗数据并生成数据集

`make_dataset.py` 读取一个或多个 `*_vehicle_tracks.csv`，完成数据清洗、特征工程和弱标签生成。

它会处理：

- 数值字段转换；
- 低置信度检测过滤；
- 过短车辆轨迹过滤；
- 异常速度和异常加速度过滤；
- 轨迹年龄、速度变化、检测框尺寸等特征生成；
- 基于物理公式生成 `required_braking_distance_m`；
- 可选生成 `latest_brake_line_m`、`brake_boundary_crossed` 和 `risk_level_weak`。

从全部 `outputs` 自动查找原始 CSV：

```powershell
python make_dataset.py --input outputs --stopline-road-y-m 249
```

指定某一个分析结果目录：

```powershell
python make_dataset.py --input outputs\20260722_153012_vehicles --stopline-road-y-m 249
```

指定某一个 CSV：

```powershell
python make_dataset.py --input outputs\20260722_153012_vehicles\vehicles_vehicle_tracks.csv --stopline-road-y-m 249
```

输出目录示例：

```text
datasets/
└── 20260722_154210_dataset/
    ├── dataset.csv
    ├── dataset_metadata.json
    └── feature_columns.json
```

## 5. 第三步：训练机器学习模型

`train.py` 读取 `make_dataset.py` 生成的 `dataset.csv`，默认训练随机森林回归模型，目标是预测：

```text
required_braking_distance_m
```

训练命令：

```powershell
python train.py --dataset-path datasets\20260722_154210_dataset\dataset.csv
```

指定目标列：

```powershell
python train.py --dataset-path datasets\20260722_154210_dataset\dataset.csv --target-column required_braking_distance_m
```

输出目录示例：

```text
models/
└── 20260722_154530_brake_model/
    ├── brake_distance_model.joblib
    ├── training_metadata.json
    └── validation_predictions.csv
```

`training_metadata.json` 中包含训练集/验证集行数、特征列、目标列和 MAE、RMSE、R2 等指标。

## 6. 推荐完整流程

```powershell
cd D:\workplace\Mission3\Project
conda activate project

python video_analyzer.py --source-video-path data\vehicles.mp4 --calibration-path data\vehicles\vehicles.calibration.json --stopline-road-y-m 249 --save-annotated-video

python make_dataset.py --input outputs --stopline-road-y-m 249

python train.py --dataset-path datasets\最新生成的数据集目录\dataset.csv
```

最后一行中的 `最新生成的数据集目录` 替换为实际生成的目录名。
