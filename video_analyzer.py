import argparse
import csv
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINTS_DIR / "yolov8x.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR.mkdir(exist_ok=True)
ANCHOR_CHOICES = sv.Position.list()
SYSTEM_SHORT_NAME = "VRPS"
SYSTEM_FULL_NAME = "Vision-based Real-time Perception & Prediction System"
TEMPORAL_NET_NAME = "MTPNet"
TEMPORAL_NET_FULL_NAME = "Mechanism Temporal Prediction Net"

COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_TYPE_BY_COCO_ID = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CSV_FIELDS = [
    "车辆横向位置x_i(t)（m）",
    "车辆纵向位置y_i(t)（m）",
    "车辆横向速度v_ix(t)（m/s）",
    "车辆纵向速度v_iy(t)（m/s）",
    "车辆横向加速度a_ix(t)（m/s^2）",
    "车辆纵向加速度a_iy(t)（m/s^2）",
    "车辆航向角theta_i(t)（rad）",
    "车型编码c_i",
    "region_id",
    "region_name",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px2",
    "confidence",
    "image_anchor_x",
    "image_anchor_y",
    "video_id",
    "frame_id",
    "timestamp_s",
    "vehicle_id",
    "vehicle_type",
    "theta_i_t_deg",
]

STATE_VECTOR_FIELDS = CSV_FIELDS[:8]

FRAME_RISK_FIELDS = [
    "video_id",
    "frame_id",
    "timestamp_s",
    "region_id",
    "region_name",
    "区域车辆数N(t)",
    "有效风险车辆数",
    "研究区域面积S_A（m^2）",
    "车流密度rho(t)（veh/m^2）",
    "区域瞬时事故概率P_A(t)",
    "最大单车风险P_i(t)",
    "最大风险车辆id",
    "最大车辆对风险P_ij(t)",
    "最大风险前车id",
    "最大风险类型",
]

PAIRWISE_RISK_FIELDS = [
    "video_id",
    "frame_id",
    "timestamp_s",
    "region_id",
    "region_name",
    "车辆i_id",
    "车辆j_id",
    "TTC_ij(t)（s）",
    "LTTC_ij(t)（s）",
    "纵向追尾风险P_long_ij(t)",
    "侧向擦碰风险P_lat_ij(t)",
    "车辆对综合碰撞概率P_ij(t)",
    "主导风险类型",
]


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.M = cv2.getPerspectiveTransform(self.source, self.target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.M)
        return transformed_points.reshape(-1, 2)


@dataclass
class CalibrationRegion:
    region_id: str
    name: str
    source: np.ndarray
    target: np.ndarray
    transformer: ViewTransformer

    @property
    def source_polygon(self) -> np.ndarray:
        return self.source.astype(np.int32)

    def contains_point(self, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        return cv2.pointPolygonTest(self.source_polygon, (x, y), False) >= 0

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        return self.transformer.transform_points(np.array([point], dtype=np.float32))[0]

    @property
    def target_area_m2(self) -> float:
        return abs(float(cv2.contourArea(self.target.astype(np.float32))))


@dataclass
class VehicleState:
    region: CalibrationRegion
    vehicle_id: int
    x: float
    y: float
    vx: float | None
    vy: float | None
    ax: float | None
    ay: float | None
    theta: float | None
    class_id: int


@dataclass
class PairwiseRisk:
    follower_id: int
    leader_id: int
    ttc_s: float | None
    lttc_s: float | None
    long_probability: float
    lateral_probability: float
    probability: float
    risk_type: str


@dataclass
class RegionRisk:
    region: CalibrationRegion
    vehicle_count: int
    valid_vehicle_count: int
    density: float
    probability: float
    max_vehicle_probability: float
    max_vehicle_id: int | None
    max_pair: PairwiseRisk | None
    pairwise_risks: list[PairwiseRisk]


@dataclass
class DashboardState:
    maxlen: int = 180
    risk_history: deque[float] | None = None
    vehicle_history: deque[int] | None = None
    density_history: deque[float] | None = None

    def __post_init__(self) -> None:
        self.risk_history = deque(maxlen=self.maxlen)
        self.vehicle_history = deque(maxlen=self.maxlen)
        self.density_history = deque(maxlen=self.maxlen)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VRPS video analyzer: export perception, mechanism risk and trajectory CSV."
    )
    parser.add_argument(
        "--source-video-path",
        required=True,
        help="Path to the source video file.",
        type=str,
    )
    parser.add_argument(
        "--calibration-path",
        default="data/vehicles/vehicles.calibration.json",
        help="Path to calibration JSON containing source and target points.",
        type=str,
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for timestamped analysis output folders.",
        type=str,
    )
    parser.add_argument(
        "--anchor",
        default="CENTER",
        choices=ANCHOR_CHOICES,
        help="Detection anchor used for vehicle position and motion estimation.",
        type=str,
    )
    parser.add_argument(
        "--speed-window-seconds",
        default=1.0,
        help="Time window used to estimate speed.",
        type=float,
    )
    parser.add_argument(
        "--speed-smoothing-seconds",
        default=0.5,
        help="Time window used to smooth velocity before acceleration estimation.",
        type=float,
    )
    parser.add_argument(
        "--display-width",
        default=960,
        help="Preview window width in pixels.",
        type=int,
    )
    parser.add_argument(
        "--vehicle-label-mode",
        default="speed",
        choices=["none", "speed", "id", "full"],
        help="Vehicle label style in the preview: none, speed, id, or full.",
        type=str,
    )
    parser.add_argument(
        "--show-state-labels",
        action="store_true",
        help="Show detailed per-vehicle state blocks on the video frame.",
    )
    parser.add_argument(
        "--plot-max-vehicles",
        default=8,
        help="Maximum number of vehicle IDs shown in kinematic plots.",
        type=int,
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable analysis PNG visualizations.",
    )
    parser.add_argument(
        "--risk-alpha",
        default=1.5,
        help="Time-decay scale alpha in seconds for TTC-based rear-end risk.",
        type=float,
    )
    parser.add_argument(
        "--risk-beta",
        default=1.0,
        help="Time-decay scale beta in seconds for LTTC-based lateral risk.",
        type=float,
    )
    parser.add_argument(
        "--risk-horizon-seconds",
        default=10.0,
        help="Ignore pairwise conflicts whose TTC/LTTC is beyond this horizon.",
        type=float,
    )
    parser.add_argument(
        "--same-direction-degrees",
        default=30.0,
        help="Maximum heading difference treated as same-direction travel.",
        type=float,
    )
    parser.add_argument(
        "--min-risk-speed-mps",
        default=0.5,
        help="Vehicles slower than this are ignored in pairwise motion risk.",
        type=float,
    )
    parser.add_argument(
        "--lateral-longitudinal-gate-m",
        default=12.0,
        help="Lateral risk is considered only when vehicles are within this longitudinal distance.",
        type=float,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the annotated preview window while processing.",
    )
    parser.add_argument(
        "--save-annotated-video",
        action="store_true",
        help="Save an annotated MP4 next to the CSV.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Render the VRPS dashboard instead of only the annotated frame.",
    )
    parser.add_argument(
        "--dashboard-width",
        default=1600,
        help="Dashboard canvas width in pixels.",
        type=int,
    )
    parser.add_argument(
        "--dashboard-height",
        default=900,
        help="Dashboard canvas height in pixels.",
        type=int,
    )
    return parser.parse_args()


def resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    project_relative_path = PROJECT_ROOT / path
    if project_relative_path.exists():
        return project_relative_path

    return path


def make_calibration_region(
    region_id: str,
    name: str,
    source_data: list[list[float]],
    target_data: list[list[float]],
) -> CalibrationRegion:
    source = np.array(source_data, dtype=np.float32)
    target = np.array(target_data, dtype=np.float32)

    if source.shape != (4, 2) or target.shape != (4, 2):
        raise ValueError("Calibration source and target must both be 4x2 point arrays.")

    return CalibrationRegion(
        region_id=region_id,
        name=name,
        source=source,
        target=target,
        transformer=ViewTransformer(source=source, target=target),
    )


def load_calibration(calibration_path: Path) -> list[CalibrationRegion]:
    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    if "regions" in calibration and calibration["regions"]:
        return [
            make_calibration_region(
                region_id=str(region.get("region_id", f"region_{index + 1}")),
                name=str(region.get("name", f"region_{index + 1}")),
                source_data=region["source"],
                target_data=region["target"],
            )
            for index, region in enumerate(calibration["regions"])
        ]

    return [
        make_calibration_region(
            region_id="region_1",
            name="region_1",
            source_data=calibration["source"],
            target_data=calibration["target"],
        )
    ]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "video"


def create_output_dir(source_video_path: Path, output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = safe_name(source_video_path.stem)
    output_dir = output_root / f"{timestamp}_{video_stem}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_label_float(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "..."
    return f"{value:.{digits}f}"


def build_metadata(
    source_video_path: Path,
    calibration_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    video_info: sv.VideoInfo,
    regions: list[CalibrationRegion],
) -> None:
    metadata = {
        "system": {
            "short_name": SYSTEM_SHORT_NAME,
            "full_name": SYSTEM_FULL_NAME,
        },
        "temporal_prediction_network": {
            "short_name": TEMPORAL_NET_NAME,
            "full_name": TEMPORAL_NET_FULL_NAME,
        },
        "source_video_path": str(source_video_path),
        "source_video_name": source_video_path.name,
        "calibration_path": str(calibration_path),
        "fps": video_info.fps,
        "resolution_wh": video_info.resolution_wh,
        "speed_window_seconds": args.speed_window_seconds,
        "speed_smoothing_seconds": args.speed_smoothing_seconds,
        "anchor": args.anchor,
        "risk_model": {
            "alpha": args.risk_alpha,
            "beta": args.risk_beta,
            "risk_horizon_seconds": args.risk_horizon_seconds,
            "same_direction_degrees": args.same_direction_degrees,
            "min_risk_speed_mps": args.min_risk_speed_mps,
            "lateral_longitudinal_gate_m": args.lateral_longitudinal_gate_m,
            "notes": (
                "TTC/LTTC beyond the risk horizon are treated as no imminent "
                "conflict. Finite conflict times are mapped by a normalized "
                "exponential decay to avoid a 0.5 lower bound for all finite TTC."
            ),
        },
        "regions": [
            {
                "region_id": region.region_id,
                "name": region.name,
                "source": region.source.tolist(),
                "target": region.target.tolist(),
                "target_area_m2": region.target_area_m2,
            }
            for region in regions
        ],
        "state_vector_columns": STATE_VECTOR_FIELDS,
        "road_coordinate_system": {
            "车辆横向位置x_i(t)（m）": "lateral axis in the calibrated road plane",
            "车辆纵向位置y_i(t)（m）": "longitudinal axis in the calibrated road plane",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def estimate_velocity_mps(
    history: deque[tuple[float, float, float]],
) -> tuple[float, float] | None:
    if len(history) < 2:
        return None

    start_time, start_x, start_y = history[0]
    end_time, end_x, end_y = history[-1]
    elapsed = end_time - start_time
    if elapsed <= 0:
        return None

    return (end_x - start_x) / elapsed, (end_y - start_y) / elapsed


def vector_magnitude(x: float | None, y: float | None) -> float | None:
    if x is None or y is None:
        return None
    return float(np.hypot(x, y))


def heading_angle_rad(velocity_x: float | None, velocity_y: float | None) -> float | None:
    speed = vector_magnitude(velocity_x, velocity_y)
    if speed is None or speed <= 1e-6:
        return None
    return float(np.arctan2(velocity_y, velocity_x))


def finite_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def conflict_probability(
    conflict_time_s: float | None,
    time_scale_s: float,
    risk_horizon_s: float,
) -> float:
    if (
        conflict_time_s is None
        or not np.isfinite(conflict_time_s)
        or conflict_time_s <= 0
        or conflict_time_s > risk_horizon_s
        or time_scale_s <= 0
    ):
        return 0.0

    raw_probability = float(np.exp(-conflict_time_s / time_scale_s))
    horizon_probability = float(np.exp(-risk_horizon_s / time_scale_s))
    normalized_probability = (raw_probability - horizon_probability) / (
        1.0 - horizon_probability
    )
    return float(np.clip(normalized_probability, 0.0, 1.0))


def is_valid_motion_state(vehicle: VehicleState, min_speed_mps: float) -> bool:
    speed = vector_magnitude(vehicle.vx, vehicle.vy)
    return (
        vehicle.vx is not None
        and vehicle.vy is not None
        and vehicle.theta is not None
        and speed is not None
        and speed >= min_speed_mps
    )


def calculate_pairwise_risk(
    ego: VehicleState,
    other: VehicleState,
    alpha: float,
    beta: float,
    risk_horizon_s: float,
    same_direction_cosine: float,
    lateral_longitudinal_gate_m: float,
) -> PairwiseRisk:
    ego_position = np.array([ego.x, ego.y], dtype=np.float64)
    other_position = np.array([other.x, other.y], dtype=np.float64)
    ego_velocity = np.array([ego.vx, ego.vy], dtype=np.float64)
    other_velocity = np.array([other.vx, other.vy], dtype=np.float64)

    ego_speed = float(np.linalg.norm(ego_velocity))
    other_speed = float(np.linalg.norm(other_velocity))
    ego_direction = ego_velocity / ego_speed
    other_direction = other_velocity / other_speed
    direction_cosine = float(np.dot(ego_direction, other_direction))

    relative_position = other_position - ego_position
    relative_velocity = ego_velocity - other_velocity
    longitudinal_gap = float(np.dot(relative_position, ego_direction))
    closing_speed = float(np.dot(relative_velocity, ego_direction))

    ttc_s = None
    if (
        direction_cosine >= same_direction_cosine
        and longitudinal_gap > 0
        and closing_speed > 0
    ):
        ttc_s = longitudinal_gap / closing_speed

    lateral_direction = np.array([-ego_direction[1], ego_direction[0]], dtype=np.float64)
    lateral_gap = float(np.dot(ego_position - other_position, lateral_direction))
    lateral_closing_speed = float(np.dot(ego_velocity - other_velocity, lateral_direction))

    lttc_s = None
    if (
        abs(longitudinal_gap) <= lateral_longitudinal_gate_m
        and lateral_gap * lateral_closing_speed < 0
    ):
        lttc_s = -lateral_gap / lateral_closing_speed

    long_probability = conflict_probability(ttc_s, alpha, risk_horizon_s)
    lateral_probability = conflict_probability(lttc_s, beta, risk_horizon_s)
    probability = max(long_probability, lateral_probability)
    if probability <= 0:
        risk_type = "none"
    elif long_probability >= lateral_probability:
        risk_type = "long"
    else:
        risk_type = "lat"

    return PairwiseRisk(
        follower_id=ego.vehicle_id,
        leader_id=other.vehicle_id,
        ttc_s=ttc_s,
        lttc_s=lttc_s,
        long_probability=long_probability,
        lateral_probability=lateral_probability,
        probability=probability,
        risk_type=risk_type,
    )


def calculate_region_risks(
    regions: list[CalibrationRegion],
    vehicle_states: list[VehicleState],
    alpha: float,
    beta: float,
    risk_horizon_s: float,
    same_direction_degrees: float,
    min_speed_mps: float,
    lateral_longitudinal_gate_m: float,
) -> list[RegionRisk]:
    states_by_region = defaultdict(list)
    for vehicle_state in vehicle_states:
        states_by_region[vehicle_state.region.region_id].append(vehicle_state)

    same_direction_cosine = float(np.cos(np.deg2rad(same_direction_degrees)))
    region_risks = []
    for region in regions:
        region_states = states_by_region[region.region_id]
        valid_states = [
            state
            for state in region_states
            if is_valid_motion_state(state, min_speed_mps=min_speed_mps)
        ]
        vehicle_max_probability = {state.vehicle_id: 0.0 for state in valid_states}
        pairwise_risks = []

        for ego in valid_states:
            for other in valid_states:
                if ego.vehicle_id == other.vehicle_id:
                    continue
                pairwise_risk = calculate_pairwise_risk(
                    ego=ego,
                    other=other,
                    alpha=alpha,
                    beta=beta,
                    risk_horizon_s=risk_horizon_s,
                    same_direction_cosine=same_direction_cosine,
                    lateral_longitudinal_gate_m=lateral_longitudinal_gate_m,
                )
                pairwise_risks.append(pairwise_risk)
                vehicle_max_probability[ego.vehicle_id] = max(
                    vehicle_max_probability[ego.vehicle_id],
                    pairwise_risk.probability,
                )

        if len(valid_states) >= 2:
            safety_probability = 1.0
            for probability in vehicle_max_probability.values():
                safety_probability *= 1.0 - probability
            area_probability = 1.0 - safety_probability
        else:
            area_probability = 0.0

        if vehicle_max_probability:
            max_vehicle_id, max_vehicle_probability = max(
                vehicle_max_probability.items(),
                key=lambda item: item[1],
            )
        else:
            max_vehicle_id = None
            max_vehicle_probability = 0.0

        max_pair = None
        if pairwise_risks:
            max_pair = max(pairwise_risks, key=lambda item: item.probability)
            if max_pair.probability <= 0:
                max_pair = None

        area_m2 = region.target_area_m2
        density = len(region_states) / area_m2 if area_m2 > 0 else 0.0
        region_risks.append(
            RegionRisk(
                region=region,
                vehicle_count=len(region_states),
                valid_vehicle_count=len(valid_states),
                density=density,
                probability=float(np.clip(area_probability, 0.0, 1.0)),
                max_vehicle_probability=max_vehicle_probability,
                max_vehicle_id=max_vehicle_id,
                max_pair=max_pair,
                pairwise_risks=pairwise_risks,
            )
        )

    return region_risks


def print_progress(
    frame_id: int,
    total_frames: int | None,
    rows_written: int,
    detections_count: int,
    start_time: float,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    processed_frames = frame_id + 1
    processing_fps = processed_frames / elapsed
    if total_frames and total_frames > 0:
        percent = processed_frames * 100 / total_frames
        progress = f"{processed_frames}/{total_frames} ({percent:5.1f}%)"
    else:
        progress = f"{processed_frames} frames"

    print(
        "\r"
        f"Processing {progress} | "
        f"frame={frame_id} | "
        f"vehicles={detections_count} | "
        f"rows={rows_written} | "
        f"fps={processing_fps:.2f}",
        end="",
        flush=True,
    )


def assign_points_to_regions(
    image_points: np.ndarray,
    regions: list[CalibrationRegion],
) -> tuple[np.ndarray, list[CalibrationRegion], np.ndarray]:
    keep_mask = []
    assigned_regions = []
    road_points = []

    for point in image_points:
        assigned_region = None
        for region in regions:
            if region.contains_point(point):
                assigned_region = region
                break

        keep_mask.append(assigned_region is not None)
        if assigned_region is not None:
            assigned_regions.append(assigned_region)
            road_points.append(assigned_region.transform_point(point))

    if road_points:
        road_points_array = np.array(road_points, dtype=np.float32)
    else:
        road_points_array = np.empty((0, 2), dtype=np.float32)
    return np.array(keep_mask, dtype=bool), assigned_regions, road_points_array


def make_empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty(0, dtype=np.float32),
        class_id=np.empty(0, dtype=int),
        tracker_id=np.empty(0, dtype=int),
    )


def calculate_annotation_style(
    resolution_wh: tuple[int, int],
    display_width: int,
) -> tuple[int, float]:
    source_width, source_height = resolution_wh
    if source_width <= 0 or source_height <= 0 or display_width <= 0:
        return (
            sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh),
            sv.calculate_optimal_text_scale(resolution_wh=resolution_wh),
        )

    display_scale = display_width / source_width
    preview_resolution_wh = (
        display_width,
        max(1, int(round(source_height * display_scale))),
    )
    preview_thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=preview_resolution_wh
    )
    preview_text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=preview_resolution_wh
    )
    drawing_thickness = max(1, int(round(preview_thickness / display_scale)))
    drawing_text_scale = max(0.1, preview_text_scale / display_scale)
    return drawing_thickness, drawing_text_scale


def resize_frame_to_window(
    frame: np.ndarray,
    window_name: str,
    fallback_width: int,
) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return frame

    target_width = max(1, int(fallback_width))
    target_height = max(1, int(round(frame_height * target_width / frame_width)))
    try:
        _, _, window_width, window_height = cv2.getWindowImageRect(window_name)
        if window_width > 1 and window_height > 1:
            target_width, target_height = window_width, window_height
    except cv2.error:
        pass

    scale = min(target_width / frame_width, target_height / frame_height)
    resized_width = max(1, int(round(frame_width * scale)))
    resized_height = max(1, int(round(frame_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

    if resized_width == target_width and resized_height == target_height:
        return resized

    canvas = np.full((target_height, target_width, 3), (42, 42, 42), dtype=frame.dtype)
    paste_x = (target_width - resized_width) // 2
    paste_y = (target_height - resized_height) // 2
    canvas[paste_y : paste_y + resized_height, paste_x : paste_x + resized_width] = resized
    return canvas


def interpolate_bgr(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    clipped_ratio = float(np.clip(ratio, 0.0, 1.0))
    return tuple(
        int(round(start_value + (end_value - start_value) * clipped_ratio))
        for start_value, end_value in zip(start, end)
    )


def risk_color_bgr(probability: float) -> tuple[int, int, int]:
    clipped_probability = float(np.clip(probability, 0.0, 1.0))
    green = (84, 214, 176)
    yellow = (74, 214, 242)
    red = (66, 88, 245)
    if clipped_probability <= 0.5:
        return interpolate_bgr(green, yellow, clipped_probability / 0.5)
    return interpolate_bgr(yellow, red, (clipped_probability - 0.5) / 0.5)


def draw_region_polygons(
    frame: np.ndarray,
    source_polygons: list[np.ndarray],
    region_risks: list[RegionRisk],
    thickness: int,
) -> np.ndarray:
    annotated_frame = frame.copy()
    outline_thickness = max(2, thickness * 2)
    shadow_thickness = outline_thickness + max(2, thickness)

    if region_risks:
        polygon_items = [
            (region_risk.region.source_polygon, risk_color_bgr(region_risk.probability))
            for region_risk in region_risks
        ]
    else:
        polygon_items = [(source_polygon, (66, 88, 245)) for source_polygon in source_polygons]

    for polygon, color in polygon_items:
        points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            annotated_frame,
            [points],
            isClosed=True,
            color=(20, 20, 20),
            thickness=shadow_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            annotated_frame,
            [points],
            isClosed=True,
            color=color,
            thickness=outline_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated_frame


def draw_state_label_blocks(
    frame: np.ndarray,
    boxes: np.ndarray,
    state_lines: list[list[str]],
    text_scale: float,
    thickness: int,
) -> np.ndarray:
    annotated_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.28, text_scale * 0.55)
    line_height = max(12, int(28 * font_scale))
    padding = max(3, int(round(5 * font_scale)))

    for bbox, lines in zip(boxes, state_lines):
        x1, y1, x2, _ = [int(round(value)) for value in bbox]
        max_text_width = 0
        text_height = 0
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line,
                font,
                font_scale,
                thickness,
            )
            max_text_width = max(max_text_width, text_width)

        panel_width = max_text_width + padding * 2
        panel_height = line_height * len(lines) + padding * 2
        panel_x1 = max(0, min(x1, annotated_frame.shape[1] - panel_width - 1))
        panel_y1 = max(0, y1 - panel_height - 4)
        if panel_y1 <= 0:
            panel_y1 = min(
                annotated_frame.shape[0] - panel_height - 1,
                int(round(y1)) + 4,
            )
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (26, 37, 47),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        cv2.rectangle(
            annotated_frame,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (112, 224, 218),
            1,
        )

        text_y = panel_y1 + padding + text_height
        for line in lines:
            cv2.putText(
                annotated_frame,
                line,
                (panel_x1 + padding, text_y),
                font,
                font_scale,
                (232, 246, 246),
                thickness,
                cv2.LINE_AA,
            )
            text_y += line_height

        cv2.line(
            annotated_frame,
            (panel_x1, panel_y2),
            (x1, int(round(y1))),
            (112, 224, 218),
            1,
        )

    return annotated_frame


def draw_compact_box_labels(
    frame: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    text_scale: float,
    thickness: int,
) -> np.ndarray:
    annotated_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.20, text_scale * 0.30)
    text_thickness = max(1, int(round(thickness * 0.45)))
    padding = max(2, int(round(4 * font_scale)))
    line_gap = max(1, int(round(3 * font_scale)))

    for bbox, label in zip(boxes, labels):
        lines = [line for line in label.splitlines() if line]
        if not lines:
            continue

        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        max_text_width = 0
        line_heights = []
        baselines = []
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line,
                font,
                font_scale,
                text_thickness,
            )
            max_text_width = max(max_text_width, text_width)
            line_heights.append(text_height)
            baselines.append(baseline)

        panel_width = max_text_width + padding * 2
        panel_height = (
            sum(line_heights)
            + sum(baselines)
            + line_gap * max(0, len(lines) - 1)
            + padding * 2
        )
        panel_x1 = max(0, min(x1, annotated_frame.shape[1] - panel_width - 1))
        panel_y1 = max(0, min(y1, annotated_frame.shape[0] - panel_height - 1))
        if panel_height > max(8, y2 - y1) and y1 - panel_height - 2 >= 0:
            panel_y1 = y1 - panel_height - 2
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (22, 31, 40),
            -1,
        )
        cv2.addWeighted(overlay, 0.58, annotated_frame, 0.42, 0, annotated_frame)

        text_y = panel_y1 + padding
        for line, text_height, baseline in zip(lines, line_heights, baselines):
            text_y += text_height
            cv2.putText(
                annotated_frame,
                line,
                (panel_x1 + padding, text_y),
                font,
                font_scale,
                (230, 244, 244),
                text_thickness,
                cv2.LINE_AA,
            )
            text_y += baseline + line_gap

    return annotated_frame


def draw_region_risk_blocks(
    frame: np.ndarray,
    region_risks: list[RegionRisk],
    text_scale: float,
    thickness: int,
) -> np.ndarray:
    annotated_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.32, text_scale * 0.65)
    line_height = max(14, int(30 * font_scale))
    padding = max(4, int(round(7 * font_scale)))

    if not region_risks:
        return annotated_frame

    rows = []
    for region_risk in region_risks:
        risk_percent = region_risk.probability * 100
        rows.append(
            (
                risk_percent,
                f"{region_risk.region.region_id}  "
                f"P={risk_percent:.1f}%  "
                f"N={region_risk.vehicle_count}  "
                f"rho={region_risk.density:.4f}",
            )
        )

    header = "VRPS mechanism risk"
    max_text_width = cv2.getTextSize(header, font, font_scale, thickness)[0][0]
    text_height = 0
    for _, line in rows:
        (text_width, text_height), _ = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )
        max_text_width = max(max_text_width, text_width)

    margin = max(6, padding)
    panel_x1 = margin
    panel_y1 = margin
    panel_width = max_text_width + padding * 2
    panel_height = line_height * (len(rows) + 1) + padding * 2
    panel_x2 = min(annotated_frame.shape[1] - 1, panel_x1 + panel_width)
    panel_y2 = min(annotated_frame.shape[0] - 1, panel_y1 + panel_height)

    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 29, 38), -1)
    cv2.addWeighted(overlay, 0.52, annotated_frame, 0.48, 0, annotated_frame)
    cv2.rectangle(
        annotated_frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (92, 190, 202),
        1,
    )

    text_y = panel_y1 + padding + text_height
    cv2.putText(
        annotated_frame,
        header,
        (panel_x1 + padding, text_y),
        font,
        font_scale,
        (215, 240, 242),
        thickness,
        cv2.LINE_AA,
    )
    text_y += line_height
    for risk_percent, line in rows:
        if risk_percent >= 70:
            color = (88, 104, 255)
        elif risk_percent >= 40:
            color = (74, 214, 242)
        else:
            color = (104, 225, 182)
        cv2.putText(
            annotated_frame,
            line,
            (panel_x1 + padding, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_height

    return annotated_frame


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    title: str | None = None,
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (17, 25, 34), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (56, 86, 102), 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y1), (92, 224, 218), 2, cv2.LINE_AA)
    if title:
        draw_text(frame, title, (x1 + 14, y1 + 30), 0.62, (208, 240, 242), 1)


def draw_metric_card(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    value: str,
    accent: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), (22, 32, 43), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 88, 104), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 5, y + height), accent, -1)
    draw_text(frame, label.upper(), (x + 16, y + 24), 0.42, (146, 176, 184), 1)
    draw_text(frame, value, (x + 16, y + height - 17), 0.82, (232, 247, 247), 2)


def draw_sparkline(
    frame: np.ndarray,
    values: list[float],
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    value_max: float | None = None,
) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (14, 22, 30), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (48, 76, 92), 1, cv2.LINE_AA)
    for ratio in (0.25, 0.5, 0.75):
        y = int(round(y2 - (y2 - y1) * ratio))
        cv2.line(frame, (x1 + 1, y), (x2 - 1, y), (30, 45, 55), 1, cv2.LINE_AA)
    if len(values) < 2:
        return
    max_value = value_max if value_max is not None else max(values)
    max_value = max(float(max_value), 1e-6)
    points = []
    for index, value in enumerate(values):
        x = int(round(x1 + index * (x2 - x1) / max(1, len(values) - 1)))
        y = int(round(y2 - np.clip(value / max_value, 0.0, 1.0) * (y2 - y1)))
        points.append((x, y))
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def render_traffic_dashboard(
    annotated_frame: np.ndarray,
    region_risks: list[RegionRisk],
    vehicle_states: list[VehicleState],
    frame_id: int,
    timestamp_s: float,
    source_name: str,
    processing_fps: float,
    dashboard_state: DashboardState,
    predictions: dict[str, float | None] | None = None,
    prediction_threshold: float | None = None,
    output_size: tuple[int, int] = (1600, 900),
) -> np.ndarray:
    width, height = output_size
    width = max(1100, int(width))
    height = max(700, int(height))
    canvas = np.full((height, width, 3), (9, 15, 22), dtype=np.uint8)

    margin = 18
    header_h = 70
    footer_h = 170
    gap = 16
    right_w = max(360, int(width * 0.29))
    video_x1 = margin
    video_y1 = header_h + margin
    video_x2 = width - right_w - gap - margin
    video_y2 = height - footer_h - margin
    right_x1 = video_x2 + gap
    right_x2 = width - margin
    content_h = video_y2 - video_y1

    draw_text(canvas, SYSTEM_SHORT_NAME, (margin, 44), 0.98, (226, 248, 248), 2)
    draw_text(
        canvas,
        f"{SYSTEM_FULL_NAME}  |  {source_name}  |  frame {frame_id}  |  t={timestamp_s:7.2f}s",
        (margin + 130, 43),
        0.55,
        (146, 180, 190),
        1,
    )
    status_color = (84, 214, 176)
    max_risk = max((risk.probability for risk in region_risks), default=0.0)
    if max_risk >= 0.7:
        status_text = "VRPS ALERT"
        status_color = (66, 88, 245)
    elif max_risk >= 0.4:
        status_text = "VRPS WATCH"
        status_color = (74, 214, 242)
    else:
        status_text = "VRPS NORMAL"
    draw_text(canvas, status_text, (width - 245, 44), 0.72, status_color, 2)

    draw_panel(canvas, video_x1, video_y1, video_x2, video_y2, "VISION PERCEPTION")
    inner_x1, inner_y1 = video_x1 + 10, video_y1 + 42
    inner_x2, inner_y2 = video_x2 - 10, video_y2 - 10
    slot_w, slot_h = inner_x2 - inner_x1, inner_y2 - inner_y1
    frame_h, frame_w = annotated_frame.shape[:2]
    scale = min(slot_w / frame_w, slot_h / frame_h)
    resized_w = max(1, int(round(frame_w * scale)))
    resized_h = max(1, int(round(frame_h * scale)))
    resized = cv2.resize(annotated_frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    paste_x = inner_x1 + (slot_w - resized_w) // 2
    paste_y = inner_y1 + (slot_h - resized_h) // 2
    canvas[paste_y : paste_y + resized_h, paste_x : paste_x + resized_w] = resized
    cv2.rectangle(canvas, (paste_x, paste_y), (paste_x + resized_w, paste_y + resized_h), (92, 136, 148), 1)

    total_vehicles = sum(risk.vehicle_count for risk in region_risks)
    valid_vehicles = sum(risk.valid_vehicle_count for risk in region_risks)
    total_density = sum(risk.density for risk in region_risks)
    speeds = [
        vector_magnitude(vehicle.vx, vehicle.vy)
        for vehicle in vehicle_states
        if vector_magnitude(vehicle.vx, vehicle.vy) is not None
    ]
    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    dashboard_state.risk_history.append(max_risk)
    dashboard_state.vehicle_history.append(total_vehicles)
    dashboard_state.density_history.append(total_density)

    card_w = (right_x2 - right_x1 - 12) // 2
    card_h = 78
    metrics = [
        ("perceived veh", str(total_vehicles), (92, 224, 218)),
        ("risk agents", str(valid_vehicles), risk_color_bgr(max_risk)),
        ("mech risk", f"{max_risk * 100:.1f}%", risk_color_bgr(max_risk)),
        ("avg speed", f"{avg_speed:.1f} m/s", (84, 214, 176)),
        ("density", f"{total_density:.4f}", (74, 214, 242)),
        ("vrps fps", f"{processing_fps:.1f}", (150, 150, 230)),
    ]
    for index, (label, value, accent) in enumerate(metrics):
        col = index % 2
        row = index // 2
        draw_metric_card(
            canvas,
            right_x1 + col * (card_w + 12),
            video_y1 + row * (card_h + 12),
            card_w,
            card_h,
            label,
            value,
            accent,
        )

    table_y1 = video_y1 + 3 * (card_h + 12) + 10
    table_title = f"REGION STATE | {TEMPORAL_NET_NAME}"
    draw_panel(canvas, right_x1, table_y1, right_x2, video_y2, table_title)
    sorted_risks = sorted(region_risks, key=lambda item: item.probability, reverse=True)
    row_y = table_y1 + 58
    row_h = max(46, min(62, (video_y2 - row_y - 10) // max(1, len(sorted_risks))))
    for region_risk in sorted_risks[: max(1, (video_y2 - row_y - 10) // row_h)]:
        color = risk_color_bgr(region_risk.probability)
        cv2.rectangle(canvas, (right_x1 + 12, row_y), (right_x2 - 12, row_y + row_h - 8), (21, 32, 43), -1)
        cv2.rectangle(canvas, (right_x1 + 12, row_y), (right_x1 + 18, row_y + row_h - 8), color, -1)
        draw_text(canvas, region_risk.region.region_id, (right_x1 + 28, row_y + 23), 0.52, (230, 238, 238), 1)
        draw_text(
            canvas,
            f"P {region_risk.probability * 100:5.1f}%   N {region_risk.vehicle_count:2d}   rho {region_risk.density:.4f}",
            (right_x1 + 28, row_y + 47),
            0.43,
            (158, 188, 196),
            1,
        )
        if predictions is not None:
            prediction = predictions.get(region_risk.region.region_id)
            if prediction is None:
                pred_text = "MTP warming"
                pred_color = (120, 148, 158)
            else:
                pred_text = f"MTP {prediction * 100:.1f}%"
                pred_color = risk_color_bgr(prediction)
                if prediction_threshold is not None and prediction >= prediction_threshold:
                    pred_text += " ALERT"
            draw_text(canvas, pred_text, (right_x2 - 165, row_y + 23), 0.42, pred_color, 1)
        row_y += row_h

    footer_y1 = height - footer_h
    draw_panel(canvas, margin, footer_y1, width - margin, height - margin, "VRPS REAL-TIME TREND")
    chart_gap = 18
    chart_w = (width - margin * 2 - chart_gap * 2 - 30) // 3
    chart_y1 = footer_y1 + 50
    chart_y2 = height - margin - 18
    chart_x = margin + 15
    draw_text(canvas, "mechanism risk", (chart_x, chart_y1 - 12), 0.44, (165, 190, 198), 1)
    draw_sparkline(canvas, list(dashboard_state.risk_history), (chart_x, chart_y1, chart_x + chart_w, chart_y2), (74, 214, 242), 1.0)
    chart_x += chart_w + chart_gap
    draw_text(canvas, "perceived vehicles", (chart_x, chart_y1 - 12), 0.44, (165, 190, 198), 1)
    vehicle_max = max(5.0, float(max(dashboard_state.vehicle_history or [0])))
    draw_sparkline(canvas, list(dashboard_state.vehicle_history), (chart_x, chart_y1, chart_x + chart_w, chart_y2), (84, 214, 176), vehicle_max)
    chart_x += chart_w + chart_gap
    draw_text(canvas, "traffic density", (chart_x, chart_y1 - 12), 0.44, (165, 190, 198), 1)
    density_max = max(0.001, float(max(dashboard_state.density_history or [0.0])))
    draw_sparkline(canvas, list(dashboard_state.density_history), (chart_x, chart_y1, chart_x + chart_w, chart_y2), (150, 150, 230), density_max)

    return canvas


def read_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def select_vehicle_ids_for_plots(
    tracks: pd.DataFrame,
    max_vehicles: int,
) -> list[int]:
    if tracks.empty or max_vehicles <= 0:
        return []
    counts = tracks.groupby("vehicle_id")["frame_id"].count().sort_values(ascending=False)
    return [int(vehicle_id) for vehicle_id in counts.head(max_vehicles).index]


def plot_vehicle_kinematics(
    track_csv_path: Path,
    output_dir: Path,
    max_vehicles: int,
) -> list[Path]:
    tracks = pd.read_csv(track_csv_path, encoding="utf-8-sig")
    if tracks.empty:
        return []

    for column in [
        "timestamp_s",
        "vehicle_id",
        "车辆横向位置x_i(t)（m）",
        "车辆纵向位置y_i(t)（m）",
        "车辆横向速度v_ix(t)（m/s）",
        "车辆纵向速度v_iy(t)（m/s）",
        "车辆横向加速度a_ix(t)（m/s^2）",
        "车辆纵向加速度a_iy(t)（m/s^2）",
        "车辆航向角theta_i(t)（rad）",
    ]:
        if column in tracks.columns:
            tracks[column] = pd.to_numeric(tracks[column], errors="coerce")

    vehicle_ids = select_vehicle_ids_for_plots(tracks, max_vehicles)
    if not vehicle_ids:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    selected = tracks[tracks["vehicle_id"].isin(vehicle_ids)].copy()
    selected = selected.sort_values(["vehicle_id", "timestamp_s"])

    x_col = "车辆横向位置x_i(t)（m）"
    y_col = "车辆纵向位置y_i(t)（m）"
    vx_col = "车辆横向速度v_ix(t)（m/s）"
    vy_col = "车辆纵向速度v_iy(t)（m/s）"
    ax_col = "车辆横向加速度a_ix(t)（m/s^2）"
    ay_col = "车辆纵向加速度a_iy(t)（m/s^2）"
    theta_col = "车辆航向角theta_i(t)（rad）"

    selected["speed_mps"] = np.hypot(selected[vx_col], selected[vy_col])
    selected["acceleration_mps2"] = np.hypot(selected[ax_col], selected[ay_col])

    saved_paths = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for vehicle_id, group in selected.groupby("vehicle_id"):
        label = f"id {int(vehicle_id)}"
        axes[0, 0].plot(group[x_col], group[y_col], marker=".", linewidth=1.2, label=label)
        axes[0, 1].plot(group["timestamp_s"], group["speed_mps"], linewidth=1.2, label=label)
        axes[1, 0].plot(group["timestamp_s"], group["acceleration_mps2"], linewidth=1.2, label=label)
        axes[1, 1].plot(group["timestamp_s"], group[theta_col], linewidth=1.2, label=label)

    axes[0, 0].set_title("VRPS vehicle trajectories")
    axes[0, 0].set_xlabel("x_i(t) / m")
    axes[0, 0].set_ylabel("y_i(t) / m")
    axes[0, 0].axis("equal")
    axes[0, 1].set_title("VRPS speed magnitude")
    axes[0, 1].set_xlabel("time / s")
    axes[0, 1].set_ylabel("speed / m/s")
    axes[1, 0].set_title("VRPS acceleration magnitude")
    axes[1, 0].set_xlabel("time / s")
    axes[1, 0].set_ylabel("acceleration / m/s^2")
    axes[1, 1].set_title("VRPS heading")
    axes[1, 1].set_xlabel("time / s")
    axes[1, 1].set_ylabel("theta / rad")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    path = plots_dir / "vehicle_kinematics_by_id.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved_paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for vehicle_id, group in selected.groupby("vehicle_id"):
        label = f"id {int(vehicle_id)}"
        axes[0].plot(group["timestamp_s"], group[vx_col], linewidth=1.1, label=f"{label} vx")
        axes[0].plot(group["timestamp_s"], group[vy_col], linewidth=1.1, linestyle="--", label=f"{label} vy")
        axes[1].plot(group["timestamp_s"], group[ax_col], linewidth=1.1, label=f"{label} ax")
        axes[1].plot(group["timestamp_s"], group[ay_col], linewidth=1.1, linestyle="--", label=f"{label} ay")
    axes[0].set_title("VRPS velocity components")
    axes[0].set_ylabel("velocity / m/s")
    axes[1].set_title("VRPS acceleration components")
    axes[1].set_xlabel("time / s")
    axes[1].set_ylabel("acceleration / m/s^2")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    path = plots_dir / "vehicle_motion_components_by_id.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


def plot_region_risk_timeseries(
    frame_risk_csv_path: Path,
    output_dir: Path,
) -> list[Path]:
    frame_risk = pd.read_csv(frame_risk_csv_path, encoding="utf-8-sig")
    if frame_risk.empty:
        return []

    for column in [
        "timestamp_s",
        "区域瞬时事故概率P_A(t)",
        "车流密度rho(t)（veh/m^2）",
        "区域车辆数N(t)",
        "最大车辆对风险P_ij(t)",
    ]:
        if column in frame_risk.columns:
            frame_risk[column] = pd.to_numeric(frame_risk[column], errors="coerce").fillna(0.0)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for region_id, group in frame_risk.groupby("region_id"):
        group = group.sort_values("timestamp_s")
        axes[0].plot(group["timestamp_s"], group["区域瞬时事故概率P_A(t)"], label=region_id)
        axes[1].plot(group["timestamp_s"], group["车流密度rho(t)（veh/m^2）"], label=region_id)
        axes[2].plot(group["timestamp_s"], group["最大车辆对风险P_ij(t)"], label=region_id)

    axes[0].set_title("VRPS mechanism risk")
    axes[0].set_ylabel("P_A(t)")
    axes[1].set_title("VRPS traffic density")
    axes[1].set_ylabel("vehicles / m^2")
    axes[2].set_title("VRPS max pairwise risk")
    axes[2].set_xlabel("time / s")
    axes[2].set_ylabel("max P_ij(t)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    path = plots_dir / "region_risk_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def make_annotated_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    state_label_blocks: list[list[str]],
    source_polygons: list[np.ndarray],
    region_risks: list[RegionRisk],
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    trace_annotator: sv.TraceAnnotator,
    text_scale: float,
    thickness: int,
    show_state_labels: bool,
    vehicle_label_mode: str,
) -> np.ndarray:
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = draw_region_polygons(
        frame=annotated_frame,
        source_polygons=source_polygons,
        region_risks=region_risks,
        thickness=thickness,
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    if labels and vehicle_label_mode == "speed":
        annotated_frame = draw_compact_box_labels(
            frame=annotated_frame,
            boxes=detections.xyxy,
            labels=labels,
            text_scale=text_scale,
            thickness=thickness,
        )
    elif labels:
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )
    if show_state_labels:
        annotated_frame = draw_state_label_blocks(
            frame=annotated_frame,
            boxes=detections.xyxy,
            state_lines=state_label_blocks,
            text_scale=text_scale,
            thickness=thickness,
        )
    return draw_region_risk_blocks(
        frame=annotated_frame,
        region_risks=region_risks,
        text_scale=text_scale,
        thickness=thickness,
    )


def main() -> None:
    args = parse_arguments()

    source_video_path = resolve_existing_path(args.source_video_path)
    calibration_path = resolve_existing_path(args.calibration_path)
    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {calibration_path}")

    output_dir = create_output_dir(source_video_path, Path(args.output_root))
    csv_path = output_dir / f"{safe_name(source_video_path.stem)}_vehicle_tracks.csv"
    frame_risk_path = output_dir / f"{safe_name(source_video_path.stem)}_frame_risk_timeseries.csv"
    pairwise_risk_path = output_dir / f"{safe_name(source_video_path.stem)}_pairwise_risk.csv"

    regions = load_calibration(calibration_path)
    video_info = sv.VideoInfo.from_video_path(str(source_video_path))
    build_metadata(source_video_path, calibration_path, output_dir, args, video_info, regions)

    model = YOLO(str(MODEL_PATH))
    anchor = sv.Position(args.anchor)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness, text_scale = calculate_annotation_style(
        resolution_wh=video_info.resolution_wh,
        display_width=args.display_width,
    )
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=anchor,
        color_lookup=sv.ColorLookup.TRACK,
    )
    source_polygons = [region.source_polygon for region in regions]
    dashboard_state = DashboardState(maxlen=max(60, int(round(video_info.fps * 12))))
    output_resolution_wh = (
        (args.dashboard_width, args.dashboard_height)
        if args.dashboard
        else video_info.resolution_wh
    )

    speed_window_frames = max(2, round(video_info.fps * args.speed_window_seconds))
    smoothing_frames = max(2, round(video_info.fps * args.speed_smoothing_seconds))
    coordinate_history = defaultdict(lambda: deque(maxlen=speed_window_frames))
    velocity_history = defaultdict(lambda: deque(maxlen=smoothing_frames))
    previous_smoothed_velocity = {}

    video_writer = None
    if args.save_annotated_video:
        annotated_video_path = output_dir / f"{safe_name(source_video_path.stem)}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(annotated_video_path),
            fourcc,
            video_info.fps,
            output_resolution_wh,
        )

    preview_window_name = "annotated_frame"
    if not args.no_show:
        cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
        preview_height = max(
            1,
            int(round(output_resolution_wh[1] * args.display_width / output_resolution_wh[0])),
        )
        cv2.resizeWindow(preview_window_name, args.display_width, preview_height)

    frame_generator = sv.get_video_frames_generator(str(source_video_path))
    rows_written = 0
    total_frames = getattr(video_info, "total_frames", None)
    start_time = time.perf_counter()

    with (
        csv_path.open("w", newline="", encoding="utf-8-sig") as track_file,
        frame_risk_path.open("w", newline="", encoding="utf-8-sig") as frame_risk_file,
        pairwise_risk_path.open("w", newline="", encoding="utf-8-sig") as pairwise_risk_file,
    ):
        writer = csv.DictWriter(track_file, fieldnames=CSV_FIELDS)
        frame_risk_writer = csv.DictWriter(
            frame_risk_file,
            fieldnames=FRAME_RISK_FIELDS,
        )
        pairwise_risk_writer = csv.DictWriter(
            pairwise_risk_file,
            fieldnames=PAIRWISE_RISK_FIELDS,
        )
        writer.writeheader()
        frame_risk_writer.writeheader()
        pairwise_risk_writer.writeheader()

        for frame_id, frame in enumerate(frame_generator):
            timestamp_s = frame_id / video_info.fps
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) > 0:
                vehicle_mask = np.isin(detections.class_id, list(COCO_VEHICLE_CLASS_IDS))
                detections = detections[vehicle_mask]

            if len(detections) > 0:
                detections = byte_track.update_with_detections(detections=detections)

            if detections.tracker_id is None:
                detections = make_empty_detections()

            image_points = detections.get_anchors_coordinates(anchor=anchor)
            keep_mask, assigned_regions, road_points = assign_points_to_regions(
                image_points=image_points,
                regions=regions,
            )
            if len(detections) > 0:
                detections = detections[keep_mask]
                image_points = image_points[keep_mask]

            labels = []
            state_label_blocks = []
            vehicle_states = []
            for index, (bbox, confidence, class_id, tracker_id, region) in enumerate(
                zip(
                    detections.xyxy,
                    detections.confidence,
                    detections.class_id,
                    detections.tracker_id,
                    assigned_regions,
                )
            ):
                image_x, image_y = image_points[index]
                road_x, road_y = road_points[index]
                vehicle_id = int(tracker_id)
                class_id = int(class_id)
                vehicle_type = VEHICLE_TYPE_BY_COCO_ID.get(class_id, "other_vehicle")
                bbox_width = max(0.0, float(bbox[2] - bbox[0]))
                bbox_height = max(0.0, float(bbox[3] - bbox[1]))
                bbox_area = bbox_width * bbox_height

                history_key = (region.region_id, vehicle_id)
                coordinate_history[history_key].append(
                    (timestamp_s, float(road_x), float(road_y))
                )
                velocity = estimate_velocity_mps(coordinate_history[history_key])
                velocity_x_mps = None
                velocity_y_mps = None
                smoothed_velocity_x_mps = None
                smoothed_velocity_y_mps = None
                acceleration_x_mps2 = None
                acceleration_y_mps2 = None
                heading_rad = None

                if velocity is not None:
                    velocity_x_mps, velocity_y_mps = velocity
                    velocity_history[history_key].append(
                        (timestamp_s, velocity_x_mps, velocity_y_mps)
                    )
                    smoothed_velocity_x_mps = float(
                        np.mean([vx for _, vx, _ in velocity_history[history_key]])
                    )
                    smoothed_velocity_y_mps = float(
                        np.mean([vy for _, _, vy in velocity_history[history_key]])
                    )
                    heading_rad = heading_angle_rad(
                        smoothed_velocity_x_mps,
                        smoothed_velocity_y_mps,
                    )
                    previous = previous_smoothed_velocity.get(history_key)
                    if previous is not None:
                        previous_time, previous_vx, previous_vy = previous
                        elapsed = timestamp_s - previous_time
                        if elapsed > 0:
                            acceleration_x_mps2 = (
                                smoothed_velocity_x_mps - previous_vx
                            ) / elapsed
                            acceleration_y_mps2 = (
                                smoothed_velocity_y_mps - previous_vy
                            ) / elapsed
                    previous_smoothed_velocity[history_key] = (
                        timestamp_s,
                        smoothed_velocity_x_mps,
                        smoothed_velocity_y_mps,
                    )

                heading_deg = None if heading_rad is None else np.degrees(heading_rad)
                vehicle_states.append(
                    VehicleState(
                        region=region,
                        vehicle_id=vehicle_id,
                        x=float(road_x),
                        y=float(road_y),
                        vx=smoothed_velocity_x_mps,
                        vy=smoothed_velocity_y_mps,
                        ax=acceleration_x_mps2,
                        ay=acceleration_y_mps2,
                        theta=heading_rad,
                        class_id=class_id,
                    )
                )
                if args.vehicle_label_mode == "id":
                    labels.append(f"#{vehicle_id}")
                elif args.vehicle_label_mode == "speed":
                    labels.append(
                        f"{vehicle_type}\n"
                        f"vx={format_label_float(smoothed_velocity_x_mps)} m/s\n"
                        f"vy={format_label_float(smoothed_velocity_y_mps)} m/s"
                    )
                elif args.vehicle_label_mode == "full":
                    labels.append(f"#{vehicle_id} {vehicle_type} {region.region_id}")
                state_label_blocks.append(
                    [
                        f"region = {region.region_id}",
                        f"x_i(t) = {format_label_float(float(road_x))} m",
                        f"y_i(t) = {format_label_float(float(road_y))} m",
                        f"v_ix(t) = {format_label_float(smoothed_velocity_x_mps)} m/s",
                        f"v_iy(t) = {format_label_float(smoothed_velocity_y_mps)} m/s",
                        f"a_ix(t) = {format_label_float(acceleration_x_mps2)} m/s^2",
                        f"a_iy(t) = {format_label_float(acceleration_y_mps2)} m/s^2",
                        f"theta_i(t) = {format_label_float(heading_rad, 2)} rad",
                        f"c_i = {class_id}",
                    ]
                )

                writer.writerow(
                    {
                        "车辆横向位置x_i(t)（m）": format_float(float(road_x), 3),
                        "车辆纵向位置y_i(t)（m）": format_float(float(road_y), 3),
                        "车辆横向速度v_ix(t)（m/s）": format_float(
                            smoothed_velocity_x_mps, 3
                        ),
                        "车辆纵向速度v_iy(t)（m/s）": format_float(
                            smoothed_velocity_y_mps, 3
                        ),
                        "车辆横向加速度a_ix(t)（m/s^2）": format_float(
                            acceleration_x_mps2, 3
                        ),
                        "车辆纵向加速度a_iy(t)（m/s^2）": format_float(
                            acceleration_y_mps2, 3
                        ),
                        "车辆航向角theta_i(t)（rad）": format_float(heading_rad, 6),
                        "车型编码c_i": class_id,
                        "region_id": region.region_id,
                        "region_name": region.name,
                        "bbox_x1": format_float(float(bbox[0]), 2),
                        "bbox_y1": format_float(float(bbox[1]), 2),
                        "bbox_x2": format_float(float(bbox[2]), 2),
                        "bbox_y2": format_float(float(bbox[3]), 2),
                        "bbox_width_px": format_float(bbox_width, 2),
                        "bbox_height_px": format_float(bbox_height, 2),
                        "bbox_area_px2": format_float(bbox_area, 2),
                        "confidence": format_float(float(confidence), 4),
                        "image_anchor_x": format_float(float(image_x), 2),
                        "image_anchor_y": format_float(float(image_y), 2),
                        "video_id": source_video_path.stem,
                        "frame_id": frame_id,
                        "timestamp_s": format_float(timestamp_s, 3),
                        "vehicle_id": vehicle_id,
                        "vehicle_type": vehicle_type,
                        "theta_i_t_deg": format_float(heading_deg, 3),
                    }
                )
                rows_written += 1

            region_risks = calculate_region_risks(
                regions=regions,
                vehicle_states=vehicle_states,
                alpha=args.risk_alpha,
                beta=args.risk_beta,
                risk_horizon_s=args.risk_horizon_seconds,
                same_direction_degrees=args.same_direction_degrees,
                min_speed_mps=args.min_risk_speed_mps,
                lateral_longitudinal_gate_m=args.lateral_longitudinal_gate_m,
            )
            for region_risk in region_risks:
                max_pair = region_risk.max_pair
                frame_risk_writer.writerow(
                    {
                        "video_id": source_video_path.stem,
                        "frame_id": frame_id,
                        "timestamp_s": format_float(timestamp_s, 3),
                        "region_id": region_risk.region.region_id,
                        "region_name": region_risk.region.name,
                        "区域车辆数N(t)": region_risk.vehicle_count,
                        "有效风险车辆数": region_risk.valid_vehicle_count,
                        "研究区域面积S_A（m^2）": format_float(
                            region_risk.region.target_area_m2,
                            3,
                        ),
                        "车流密度rho(t)（veh/m^2）": format_float(
                            region_risk.density,
                            6,
                        ),
                        "区域瞬时事故概率P_A(t)": format_float(
                            region_risk.probability,
                            6,
                        ),
                        "最大单车风险P_i(t)": format_float(
                            region_risk.max_vehicle_probability,
                            6,
                        ),
                        "最大风险车辆id": ""
                        if region_risk.max_vehicle_id is None
                        else region_risk.max_vehicle_id,
                        "最大车辆对风险P_ij(t)": ""
                        if max_pair is None
                        else format_float(max_pair.probability, 6),
                        "最大风险前车id": ""
                        if max_pair is None
                        else max_pair.leader_id,
                        "最大风险类型": "" if max_pair is None else max_pair.risk_type,
                    }
                )
                for pairwise_risk in region_risk.pairwise_risks:
                    if pairwise_risk.probability <= 0:
                        continue
                    pairwise_risk_writer.writerow(
                        {
                            "video_id": source_video_path.stem,
                            "frame_id": frame_id,
                            "timestamp_s": format_float(timestamp_s, 3),
                            "region_id": region_risk.region.region_id,
                            "region_name": region_risk.region.name,
                            "车辆i_id": pairwise_risk.follower_id,
                            "车辆j_id": pairwise_risk.leader_id,
                            "TTC_ij(t)（s）": finite_seconds(pairwise_risk.ttc_s),
                            "LTTC_ij(t)（s）": finite_seconds(pairwise_risk.lttc_s),
                            "纵向追尾风险P_long_ij(t)": format_float(
                                pairwise_risk.long_probability,
                                6,
                            ),
                            "侧向擦碰风险P_lat_ij(t)": format_float(
                                pairwise_risk.lateral_probability,
                                6,
                            ),
                            "车辆对综合碰撞概率P_ij(t)": format_float(
                                pairwise_risk.probability,
                                6,
                            ),
                            "主导风险类型": pairwise_risk.risk_type,
                        }
                    )

            detections_count = len(detections)
            print_progress(
                frame_id=frame_id,
                total_frames=total_frames,
                rows_written=rows_written,
                detections_count=detections_count,
                start_time=start_time,
            )

            show_preview = not args.no_show
            if show_preview or video_writer is not None:
                annotated_frame = make_annotated_frame(
                    frame=frame,
                    detections=detections,
                    labels=labels,
                    state_label_blocks=state_label_blocks,
                    source_polygons=source_polygons,
                    region_risks=region_risks,
                    box_annotator=box_annotator,
                    label_annotator=label_annotator,
                    trace_annotator=trace_annotator,
                    text_scale=text_scale,
                    thickness=thickness,
                    show_state_labels=args.show_state_labels,
                    vehicle_label_mode=args.vehicle_label_mode,
                )
                if args.dashboard:
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    annotated_frame = render_traffic_dashboard(
                        annotated_frame=annotated_frame,
                        region_risks=region_risks,
                        vehicle_states=vehicle_states,
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        source_name=source_video_path.stem,
                        processing_fps=(frame_id + 1) / elapsed,
                        dashboard_state=dashboard_state,
                        output_size=output_resolution_wh,
                    )

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                if show_preview:
                    display_frame = resize_frame_to_window(
                        annotated_frame,
                        preview_window_name,
                        args.display_width,
                    )
                    cv2.imshow(preview_window_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    if video_writer is not None:
        video_writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    plot_paths = []
    if not args.no_plots:
        plot_paths.extend(
            plot_vehicle_kinematics(
                track_csv_path=csv_path,
                output_dir=output_dir,
                max_vehicles=args.plot_max_vehicles,
            )
        )
        plot_paths.extend(
            plot_region_risk_timeseries(
                frame_risk_csv_path=frame_risk_path,
                output_dir=output_dir,
            )
        )

    print()
    print(f"CSV saved to: {csv_path}")
    print(f"Frame risk CSV saved to: {frame_risk_path}")
    print(f"Pairwise risk CSV saved to: {pairwise_risk_path}")
    if plot_paths:
        print(f"Plots saved to: {output_dir / 'plots'}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
