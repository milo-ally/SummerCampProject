# SOURCE Calibration Tool

这个工具用于从视频画面中点选 `SOURCE` 的 4 个角点，并输出可复制到分析脚本的坐标。

默认会按视频文件名创建独立文件夹，并把 `source` 写入对应 JSON 文件。`target` 需要根据人工实测结果手动修改。

运行：

```bash
python source_calibration_tool/calibrate_source.py --source-video-path data/vehicles.mp4 --display-width 1280 --padding 300
```

默认输出：

```text
data/vehicles/vehicles.calibration.json
```

也可以显式指定输出路径：

```bash
python source_calibration_tool/calibrate_source.py --source-video-path data/vehicles.mp4 --display-width 1280 --padding 300 --output-json-path data/vehicles/vehicles.calibration.json
```

操作：

- 鼠标左键依次点击 4 个点
- 点选顺序要和 `TARGET` 一致：左上、右上、右下、左下
- 如果下边两个点在画面外，仍然先点右下，再点左下
- 移动鼠标时会显示从上一个点到当前鼠标位置的红色预览线
- 按住 `Shift` 点击，可以把当前点和上一个点拉成水平直线
- 按 `Ctrl+Z` 或 `z` 撤回上一个点
- 按 `r` 重新点选
- 按 `q` 或 `ESC` 退出

重要说明：

- 输出坐标和显示器分辨率无关
- 输出坐标基于视频原始帧分辨率，比如 `3840 x 2160`
- 工具窗口可以缩小显示，比如 `1280` 宽
- 鼠标点击的是缩放后的显示坐标，工具会自动换算回原始视频坐标
- `SOURCE` 的点可以在视频帧外，因此工具会在画面四周加黑色边距
- 点在画面外时，输出坐标可能是负数，或大于视频原始宽高，这是正常的

输出示例：

```python
SOURCE = np.array(
    [
        [1252, 787],
        [2298, 803],
        [5039, 2159],
        [-550, 2259]
    ]
)
```
