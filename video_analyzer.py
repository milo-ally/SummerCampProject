import argparse
import csv
import json
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINTS_DIR / "yolov8x.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR.mkdir(exist_ok=True)
ANCHOR_CHOICES = sv.Position.list()

COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_TYPE_BY_COCO_ID = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CSV_FIELDS = [
    "video_id",
    "frame_id",
    "timestamp_s",
    "vehicle_id",
    "vehicle_type",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "image_anchor_x",
    "image_anchor_y",
    "road_x_m",
    "road_y_m",
    "distance_to_stopline_m",
    "speed_mps",
    "speed_kmh",
    "smoothed_speed_mps",
    "acceleration_mps2",
    "speed_window_s",
    "red_light_phase",
    "in_observation_zone",
    "braking_state",
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one traffic video and export vehicle trajectory CSV."
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
        default="BOTTOM_CENTER",
        choices=ANCHOR_CHOICES,
        help="Detection anchor used for speed estimation.",
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
        help="Time window used to smooth speed before acceleration estimation.",
        type=float,
    )
    parser.add_argument(
        "--stopline-road-y-m",
        default=None,
        help="Optional stop line y-coordinate in transformed road meters.",
        type=float,
    )
    parser.add_argument(
        "--red-light-phase",
        action="store_true",
        help="Mark every exported row as red-light phase when analyzing a red-light clip.",
    )
    parser.add_argument(
        "--display-width",
        default=960,
        help="Preview window width in pixels.",
        type=int,
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
    return parser.parse_args()


def resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    project_relative_path = PROJECT_ROOT / path
    if project_relative_path.exists():
        return project_relative_path

    return path


def load_calibration(calibration_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    source = np.array(calibration["source"], dtype=np.float32)
    target = np.array(calibration["target"], dtype=np.float32)

    if source.shape != (4, 2) or target.shape != (4, 2):
        raise ValueError("Calibration source and target must both be 4x2 point arrays.")

    return source, target


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


def build_metadata(
    source_video_path: Path,
    calibration_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    video_info: sv.VideoInfo,
) -> None:
    metadata = {
        "source_video_path": str(source_video_path),
        "source_video_name": source_video_path.name,
        "calibration_path": str(calibration_path),
        "fps": video_info.fps,
        "resolution_wh": video_info.resolution_wh,
        "speed_window_seconds": args.speed_window_seconds,
        "speed_smoothing_seconds": args.speed_smoothing_seconds,
        "anchor": args.anchor,
        "stopline_road_y_m": args.stopline_road_y_m,
        "red_light_phase": args.red_light_phase,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def estimate_speed_mps(history: deque[tuple[float, float, float]]) -> float | None:
    if len(history) < 2:
        return None

    start_time, _, start_y = history[0]
    end_time, _, end_y = history[-1]
    elapsed = end_time - start_time
    if elapsed <= 0:
        return None

    return abs(end_y - start_y) / elapsed


def classify_braking_state(acceleration_mps2: float | None) -> str:
    if acceleration_mps2 is None:
        return "unknown"
    if acceleration_mps2 < -1.0:
        return "decelerating"
    if acceleration_mps2 > 1.0:
        return "accelerating"
    return "steady"


def make_annotated_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    source_polygon: np.ndarray,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    trace_annotator: sv.TraceAnnotator,
) -> np.ndarray:
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = sv.draw_polygon(
        annotated_frame,
        polygon=source_polygon,
        color=sv.Color.RED,
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    return label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
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

    source, target = load_calibration(calibration_path)
    video_info = sv.VideoInfo.from_video_path(str(source_video_path))
    build_metadata(source_video_path, calibration_path, output_dir, args, video_info)

    model = YOLO(str(MODEL_PATH))
    anchor = sv.Position(args.anchor)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh
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
    source_polygon = source.astype(np.int32)
    polygon_zone = sv.PolygonZone(polygon=source_polygon)
    view_transformer = ViewTransformer(source=source, target=target)

    speed_window_frames = max(2, round(video_info.fps * args.speed_window_seconds))
    smoothing_frames = max(2, round(video_info.fps * args.speed_smoothing_seconds))
    coordinate_history = defaultdict(lambda: deque(maxlen=speed_window_frames))
    speed_history = defaultdict(lambda: deque(maxlen=smoothing_frames))
    previous_smoothed_speed = {}

    video_writer = None
    if args.save_annotated_video:
        annotated_video_path = output_dir / f"{safe_name(source_video_path.stem)}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(annotated_video_path),
            fourcc,
            video_info.fps,
            video_info.resolution_wh,
        )

    frame_generator = sv.get_video_frames_generator(str(source_video_path))
    rows_written = 0

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for frame_id, frame in enumerate(frame_generator):
            timestamp_s = frame_id / video_info.fps
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) > 0:
                vehicle_mask = np.isin(detections.class_id, list(COCO_VEHICLE_CLASS_IDS))
                detections = detections[vehicle_mask]
                detections = detections[polygon_zone.trigger(detections)]
                detections = byte_track.update_with_detections(detections=detections)

            image_points = detections.get_anchors_coordinates(anchor=anchor)
            road_points = view_transformer.transform_points(points=image_points)

            labels = []
            for index, (bbox, confidence, class_id, tracker_id) in enumerate(
                zip(
                    detections.xyxy,
                    detections.confidence,
                    detections.class_id,
                    detections.tracker_id,
                )
            ):
                image_x, image_y = image_points[index]
                road_x, road_y = road_points[index]
                vehicle_id = int(tracker_id)
                vehicle_type = VEHICLE_TYPE_BY_COCO_ID.get(int(class_id), "other_vehicle")

                coordinate_history[vehicle_id].append(
                    (timestamp_s, float(road_x), float(road_y))
                )
                speed_mps = estimate_speed_mps(coordinate_history[vehicle_id])
                smoothed_speed_mps = None
                acceleration_mps2 = None

                if speed_mps is not None:
                    speed_history[vehicle_id].append((timestamp_s, speed_mps))
                    smoothed_speed_mps = float(
                        np.mean([speed for _, speed in speed_history[vehicle_id]])
                    )
                    previous = previous_smoothed_speed.get(vehicle_id)
                    if previous is not None:
                        previous_time, previous_speed = previous
                        elapsed = timestamp_s - previous_time
                        if elapsed > 0:
                            acceleration_mps2 = (
                                smoothed_speed_mps - previous_speed
                            ) / elapsed
                    previous_smoothed_speed[vehicle_id] = (
                        timestamp_s,
                        smoothed_speed_mps,
                    )

                distance_to_stopline_m = None
                if args.stopline_road_y_m is not None:
                    distance_to_stopline_m = abs(float(road_y) - args.stopline_road_y_m)

                braking_state = classify_braking_state(acceleration_mps2)
                speed_kmh = None if speed_mps is None else speed_mps * 3.6
                label_speed = "..." if speed_kmh is None else f"{speed_kmh:.0f} km/h"
                labels.append(f"#{vehicle_id} {vehicle_type} {label_speed}")

                writer.writerow(
                    {
                        "video_id": source_video_path.stem,
                        "frame_id": frame_id,
                        "timestamp_s": format_float(timestamp_s, 3),
                        "vehicle_id": vehicle_id,
                        "vehicle_type": vehicle_type,
                        "bbox_x1": format_float(float(bbox[0]), 2),
                        "bbox_y1": format_float(float(bbox[1]), 2),
                        "bbox_x2": format_float(float(bbox[2]), 2),
                        "bbox_y2": format_float(float(bbox[3]), 2),
                        "confidence": format_float(float(confidence), 4),
                        "image_anchor_x": format_float(float(image_x), 2),
                        "image_anchor_y": format_float(float(image_y), 2),
                        "road_x_m": format_float(float(road_x), 3),
                        "road_y_m": format_float(float(road_y), 3),
                        "distance_to_stopline_m": format_float(
                            distance_to_stopline_m, 3
                        ),
                        "speed_mps": format_float(speed_mps, 3),
                        "speed_kmh": format_float(speed_kmh, 3),
                        "smoothed_speed_mps": format_float(smoothed_speed_mps, 3),
                        "acceleration_mps2": format_float(acceleration_mps2, 3),
                        "speed_window_s": format_float(args.speed_window_seconds, 2),
                        "red_light_phase": str(args.red_light_phase).lower(),
                        "in_observation_zone": "true",
                        "braking_state": braking_state,
                    }
                )
                rows_written += 1

            show_preview = not args.no_show
            if show_preview or video_writer is not None:
                annotated_frame = make_annotated_frame(
                    frame=frame,
                    detections=detections,
                    labels=labels,
                    source_polygon=source_polygon,
                    box_annotator=box_annotator,
                    label_annotator=label_annotator,
                    trace_annotator=trace_annotator,
                )

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                if show_preview:
                    height, width = annotated_frame.shape[:2]
                    display_height = int(height * args.display_width / width)
                    display_frame = cv2.resize(
                        annotated_frame,
                        (args.display_width, display_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow("annotated_frame", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if frame_id % 100 == 0:
                print(f"Processed frame {frame_id}, rows written: {rows_written}")

    if video_writer is not None:
        video_writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print(f"CSV saved to: {csv_path}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
