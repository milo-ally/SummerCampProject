# 路段动态风险实时感知系统

本项目实现《融合前端机器视觉与机理耦合时序网络的路段动态风险实时预判研究》的工程主线：

```text
视频输入 -> 区域标定 -> YOLO 检测 -> ByteTrack 跟踪 -> 透视变换测量
       -> 机理瞬时风险 P_A(t) -> 时序数据集 -> RNN/LSTM/Transformer/Mamba 风险预测
```

核心车辆状态向量为：

```text
o_i(t) = [x_i(t), y_i(t), v_ix(t), v_iy(t), a_ix(t), a_iy(t), theta_i(t), c_i]
```

其中位置单位为 `m`，速度单位为 `m/s`，加速度单位为 `m/s^2`，航向角单位为 `rad`。

## 1. 环境准备

```powershell
cd D:\workplace\Mission3\Project
conda activate project
python -m pip install -r requirements.txt
```

确认核心依赖：

```powershell
python -c `
  "import cv2, numpy, pandas, supervision, torch; from ultralytics import YOLO; print('env ok')"
```

YOLO 权重放在：

```text
checkpoints/yolov8x.pt
```

如果没有测试视频，可下载：

```text
https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

建议保存为：

```text
data/vehicles.mp4
```

## 2. 标定研究区域

启动标注软件：

```powershell
python labeling.py
```

标注软件支持：

- 选择视频并自动抽取有效帧；
- 后台解析并显示进度百分比；
- 画布自动适配窗口，支持 `Fit Frame` 和 `Ctrl + 鼠标滚轮` 缩放；
- 新增、删除、命名多个研究区域；
- 每个区域独立设置真实宽度和长度；
- 拖动四边形控制点完成道路区域标定；
- 鼠标拉线标注参考距离；
- 导出兼容 `video_analyzer.py` 的标定 JSON。

标定文件示例：

```json
{
  "source": [[0, 0], [100, 0], [100, 100], [0, 100]],
  "target": [[0, 0], [25, 0], [25, 125], [0, 125]],
  "regions": [
    {
      "region_id": "region_1",
      "name": "region_1",
      "source": [[0, 0], [100, 0], [100, 100], [0, 100]],
      "target": [[0, 0], [25, 0], [25, 125], [0, 125]],
      "target_width_m": 25.0,
      "target_length_m": 125.0
    }
  ],
  "distance_annotations": []
}
```

`regions` 可包含多个感兴趣区域。每个区域都有独立透视变换矩阵，因此可适配鸟瞰、街角监控、斜向道路等不同视角。顶层 `source` 和 `target` 是第一个区域的兼容字段。

## 3. 分析视频并计算机理风险

默认弹窗运行：

```powershell
python video_analyzer.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json
```

无弹窗批处理：

```powershell
python video_analyzer.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --no-show
```

保存标注视频：

```powershell
python video_analyzer.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --save-annotated-video
```

智能交通监控大屏模式：
```powershell
python video_analyzer.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --dashboard
```

保存大屏版视频：
```powershell
python video_analyzer.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --dashboard `
  --save-annotated-video
```

大屏会在同一画面中实时显示检测视频、车辆总数、有效风险车辆数、最大区域风险、平均速度、车流密度、处理 FPS、区域风险排行和底部滚动趋势。可通过 `--dashboard-width`、`--dashboard-height` 调整画布尺寸，默认是 `1600x900`。

关键风险参数：

```powershell
python video_analyzer.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --risk-alpha 1.5 `
  --risk-beta 1.0 `
  --risk-horizon-seconds 10 `
  --lateral-longitudinal-gate-m 12
```

参数含义：

- `--risk-alpha`：纵向追尾 TTC 风险时间衰减尺度，单位 `s`；
- `--risk-beta`：侧向擦碰 LTTC 风险时间衰减尺度，单位 `s`；
- `--risk-horizon-seconds`：超过该时间的 TTC/LTTC 不计为即时冲突；
- `--lateral-longitudinal-gate-m`：只有两车纵向距离小于该阈值时才计算侧向擦碰风险；
- `--anchor CENTER`：默认使用检测框中心点代表车辆位置，减小框底抖动影响。

当前风险概率使用带时域截断的指数衰减映射，避免原始 Sigmoid 形式对所有有限 TTC 都给出 0.5 以上概率造成虚高。

分析输出目录示例：

```text
outputs/
└── 20260722_153012_vehicles/
    ├── vehicles_vehicle_tracks.csv
    ├── vehicles_frame_risk_timeseries.csv
    ├── vehicles_pairwise_risk.csv
    ├── metadata.json
    ├── plots/
    │   ├── vehicle_kinematics_by_id.png
    │   ├── vehicle_motion_components_by_id.png
    │   └── region_risk_timeseries.png
    └── vehicles_annotated.mp4
```

## 4. 输出字段

`vehicles_vehicle_tracks.csv` 前 8 列严格对应状态向量：

```text
车辆横向位置x_i(t)（m）
车辆纵向位置y_i(t)（m）
车辆横向速度v_ix(t)（m/s）
车辆纵向速度v_iy(t)（m/s）
车辆横向加速度a_ix(t)（m/s^2）
车辆纵向加速度a_iy(t)（m/s^2）
车辆航向角theta_i(t)（rad）
车型编码c_i
```

后续辅助列保留区域、检测框、置信度、视频帧、车辆 ID 和车型等信息。

`vehicles_frame_risk_timeseries.csv` 每帧每区域输出一行，核心字段为：

```text
区域车辆数N(t)
有效风险车辆数
研究区域面积S_A（m^2）
车流密度rho(t)（veh/m^2）
区域瞬时事故概率P_A(t)
最大单车风险P_i(t)
最大车辆对风险P_ij(t)
最大风险类型
```

`vehicles_pairwise_risk.csv` 记录非零风险车辆对：

```text
TTC_ij(t)（s）
LTTC_ij(t)（s）
纵向追尾风险P_long_ij(t)
侧向擦碰风险P_lat_ij(t)
车辆对综合碰撞概率P_ij(t)
```

视频预览会实时显示车辆状态向量和每个区域的：

```text
region_1 risk P_A(t) = ...%
N(t) = ...
rho(t) = ... veh/m^2
max pair = i->j ...%
```

分析结束后会自动生成车辆运动学可视化：

- `vehicle_kinematics_by_id.png`：按车辆 ID 绘制轨迹、速度模长、加速度模长和航向角；
- `vehicle_motion_components_by_id.png`：按车辆 ID 绘制 `v_ix/v_iy` 与 `a_ix/a_iy` 分量；
- `region_risk_timeseries.png`：绘制各区域 `P_A(t)`、车流密度和最大车辆对风险。

如果不需要保存图片，可添加：

```powershell
python video_analyzer.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --no-plots
```

## 5. 构建时序数据集

`make_dataset.py` 读取 `*_frame_risk_timeseries.csv`，按 `video_id + region_id` 独立切分滑动窗口。

```powershell
python make_dataset.py `
  --input outputs `
  --window-size 60 `
  --horizon-size 60 `
  --risk-threshold 0.7
```

也可以传入具体的时间戳目录：

```powershell
python make_dataset.py `
  --input outputs\20260722_230505_vehicles `
  --window-size 60 `
  --horizon-size 60 `
  --risk-threshold 0.7
```

默认输入特征：

```text
区域瞬时事故概率P_A(t)
车流密度rho(t)（veh/m^2）
区域车辆数N(t)
有效风险车辆数
最大单车风险P_i(t)
最大车辆对风险P_ij(t)
risk_type_long
risk_type_lat
```

弱监督标签定义为：

```text
Y(t) = 未来 horizon_size 帧内 max(P_A) 是否超过 risk_threshold
```

输出目录示例：

```text
datasets/
└── 20260722_163000_risk_sequence/
    ├── train.npz
    ├── val.npz
    ├── test.npz
    ├── feature_columns.json
    ├── sequence_index.csv
    └── dataset_metadata.json
```

## 6. 训练时序模型

`train.py` 使用 PyTorch 训练未来风险预测模型，支持：

```text
rnn
gru
lstm
transformer
mamba
mtpnet
```

示例：

```powershell
python train.py `
  --dataset-dir datasets\20260722_163000_risk_sequence `
  --model mtpnet `
  --epochs 30

python train.py `
  --dataset-dir datasets\20260722_163000_risk_sequence `
  --model lstm `
  --epochs 30

python train.py `
  --dataset-dir datasets\20260722_163000_risk_sequence `
  --model transformer `
  --epochs 30

python train.py `
  --dataset-dir datasets\20260722_163000_risk_sequence `
  --model mamba `
  --epochs 30
```

训练输出：

```text
trained_models/
└── 20260722_170000_lstm_risk/
    ├── best_model.pt
    ├── training_history.json
    ├── plots/
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   ├── precision_recall_curve.png
    │   └── probability_histogram.png
    └── training_metadata.json
```

`best_model.pt` 保存模型权重、模型结构参数、特征列、标准化均值和标准差。实时部署时必须使用相同特征顺序和标准化参数构造滑动窗口。

训练可视化包括 loss、accuracy、precision、recall、F1、AUC、ROC、PR、混淆矩阵和测试集预测概率分布。

## 6.1 Chapter 4 experiment runner

`experiments/run_chapter4_experiments.py` generates the experiment artifacts required by Chapter 4, including baseline comparison, MTPNet ablation tables, temporal model latency, and templates for sensitivity, qualitative cases, system latency and low-intrusion deployment records.

Quick structural check without training:

```powershell
python experiments\run_chapter4_experiments.py `
  --dataset-dir datasets\20260722_233646_risk_sequence `
  --skip-training `
  --skip-latency
```

Full baseline and ablation run:

```powershell
python experiments\run_chapter4_experiments.py `
  --dataset-dir datasets\20260722_233646_risk_sequence `
  --epochs 30 `
  --models rnn gru lstm transformer mamba mtpnet
```

The script writes CSV tables under `experiments/chapter4/<timestamp>_chapter4/tables/`.

## 7. 端到端推理 Demo

`demo.py` 将完整串联：

```text
视频 -> YOLO 检测 -> ByteTrack 跟踪 -> 透视变换 -> 机理风险 P_A(t)
     -> 滑动窗口 -> 时序模型 -> 未来风险概率
```

运行示例：

```powershell
python demo.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --temporal-model-path trained_models\20260723_000625_mtpnet_risk\best_model.pt
```

无弹窗批处理：

```powershell
python demo.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --temporal-model-path trained_models\20260723_000625_mtpnet_risk\best_model.pt `
  --no-show
```

保存端到端标注视频：

```powershell
python demo.py 
  --source-video-path data\vehicles.mp4 
  --calibration-path data\vehicles\vehicles.calibration.json 
  --temporal-model-path trained_models\20260723_000625_mtpnet_risk\best_model.pt 
  --save-video
```

`demo.py` 默认使用智能交通监控大屏布局，右侧区域状态会同时显示当前机理风险和时序模型预测的未来风险。保存视频时会直接输出大屏版 MP4。

如果需要恢复为旧版仅在视频上叠加标注和预测块的画面：
```powershell
python demo.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --temporal-model-path trained_models\20260723_000625_mtpnet_risk\best_model.pt `
  --no-dashboard
```

大屏尺寸同样可用 `--dashboard-width` 和 `--dashboard-height` 调整，例如：
```powershell
python demo.py `
  --source-video-path data\vehicles.mp4 `
  --calibration-path data\vehicles\vehicles.calibration.json `
  --temporal-model-path trained_models\20260723_000625_mtpnet_risk\best_model.pt `
  --dashboard-width 1920 `
  --dashboard-height 1080
```

常用参数：

- `--window-size`：时序模型输入窗口长度；默认优先读取训练目录中的 `training_metadata.json`；
- `--threshold`：未来高风险报警阈值；默认读取模型 checkpoint 中的训练阈值；
- `--device auto|cpu|cuda`：推理设备；
- `--display-width`：预览窗口宽度；
- `--risk-alpha`、`--risk-beta`、`--risk-horizon-seconds`、
  `--lateral-longitudinal-gate-m`：端到端推理时沿用的机理风险参数。

输出目录示例：

```text
demo_outputs/
└── 20260722_180000_vehicles_demo/
    ├── vehicles_demo_predictions.csv
    ├── vehicles_demo.mp4
    └── demo_metadata.json
```

其中 `vehicles_demo_predictions.csv` 每帧每区域输出：

```text
instant_risk_P_A_t
future_risk_probability
alert
window_ready
```

画面中同时显示当前机理风险 `P_A(t)` 和时序模型预测的未来风险概率 `future Y`。

## 8. 项目结构

```text
.
├── labeling.py
├── video_analyzer.py
├── make_dataset.py
├── train.py
├── demo.py
├── models/
│   ├── __init__.py
│   ├── rnn.py
│   ├── transformer.py
│   ├── mamba.py
│   └── mtpnet.py
├── requirements.txt
├── data/
├── checkpoints/
├── outputs/
├── demo_outputs/
├── datasets/
└── trained_models/
```

## 9. 注意事项

- `checkpoints/`、`outputs/`、`demo_outputs/`、`datasets/`、`trained_models/` 和视频文件默认不纳入 Git；
- 标定 JSON 可以提交，便于复现实验；
- 当前训练标签是基于未来机理风险阈值构造的弱监督标签；
- 后续如果有真实事故、近碰、急刹或人工高危标注，应替换弱标签，升级为真实事故概率预测；
- OpenCV 预览窗口不可靠渲染中文，因此画面上使用 `region_id`，CSV 和 JSON 仍可保留中文语义字段。

## Chapter 4 table script usage

Use the `project` conda environment when generating Chapter 4 experiment tables,
because the latency table needs PyTorch and the local model definitions:

```powershell
conda run -n project python experiments\run_chapter4_experiments.py `
  --dataset-dir datasets\20260724_174006_risk_sequence `
  --output-root experiments\chapter4 `
  --epochs 50 `
  --models rnn gru lstm transformer mamba mtpnet
```

Quick structural check without model training:

```powershell
conda run -n project python experiments\run_chapter4_experiments.py `
  --dataset-dir datasets\20260724_174006_risk_sequence `
  --skip-training `
  --skip-latency
```

After the script finishes, tables are saved under:

```text
experiments/chapter4/<timestamp>_chapter4/tables/
```

For the current experiment directory, use:

```text
experiments/chapter4/20260724_174209_chapter4/tables/
```

Expected table files:

```text
table_4_1_dataset_summary.csv
table_4_2_model_settings.csv
table_4_3_baseline_comparison.csv
table_4_4_ablation_study.csv
table_4_5_sensitivity.csv
table_4_6_qualitative_cases.csv
table_4_7_temporal_model_latency.csv
table_4_8_system_latency.csv
table_4_9_low_intrusion_deployment.csv
```

If the run is interrupted, inspect the latest output directory:

```powershell
Get-ChildItem experiments\chapter4
Get-ChildItem experiments\chapter4\<timestamp>_chapter4\tables
```

Status notes:

- `completed`: real test metrics were written from `training_metadata.json`.
- `checkpoint_only_no_test_metadata`: `best_model.pt` exists, but final evaluation metadata was not written.
- `dataset_ready_training_not_completed`: the ablation dataset exists, but the corresponding model still needs training.
- `table_4_8_system_latency.csv` needs per-module timing records from `video_analyzer.py`; use `table_4_7_temporal_model_latency.csv` for temporal-model-only inference latency.
