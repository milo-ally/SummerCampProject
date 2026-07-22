import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs"

NUMERIC_COLUMNS = [
    "frame_id",
    "timestamp_s",
    "vehicle_id",
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
]

COMFORT_DECEL_MPS2 = {
    "car": 3.0,
    "motorcycle": 3.2,
    "bus": 2.0,
    "truck": 1.8,
    "other_vehicle": 2.5,
}

FEATURE_COLUMNS = [
    "vehicle_type",
    "timestamp_s",
    "track_age_s",
    "track_age_frames",
    "road_x_m",
    "road_y_m",
    "distance_to_stopline_m",
    "speed_mps",
    "speed_kmh",
    "smoothed_speed_mps",
    "acceleration_mps2",
    "delta_speed_mps",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_ratio",
    "confidence",
    "red_light_phase",
    "in_observation_zone",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw vehicle track CSV files and build an ML dataset."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_INPUT_ROOT)],
        help="Raw CSV files or directories containing *_vehicle_tracks.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dataset output directory. Defaults to datasets/<timestamp>_dataset.",
    )
    parser.add_argument(
        "--stopline-road-y-m",
        default=None,
        type=float,
        help="Optional stop line y-coordinate used when raw CSV has no distance column.",
    )
    parser.add_argument(
        "--min-track-points",
        default=8,
        type=int,
        help="Drop vehicle tracks shorter than this many rows.",
    )
    parser.add_argument(
        "--min-confidence",
        default=0.25,
        type=float,
        help="Drop detections below this confidence.",
    )
    parser.add_argument(
        "--max-speed-mps",
        default=45.0,
        type=float,
        help="Drop physically implausible speeds above this value.",
    )
    parser.add_argument(
        "--max-abs-acceleration-mps2",
        default=12.0,
        type=float,
        help="Drop physically implausible absolute accelerations above this value.",
    )
    parser.add_argument(
        "--reaction-time-s",
        default=1.0,
        type=float,
        help="Reaction time used for weak physical labels.",
    )
    parser.add_argument(
        "--buffer-distance-m",
        default=3.0,
        type=float,
        help="Safety buffer used for weak physical labels.",
    )
    parser.add_argument(
        "--warning-margin-m",
        default=8.0,
        type=float,
        help="Distance margin for weak Warning labels before the brake boundary.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "dataset"


def resolve_input_paths(paths: list[str]) -> list[Path]:
    csv_paths = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            project_path = PROJECT_ROOT / path
            path = project_path if project_path.exists() else path

        if path.is_file():
            csv_paths.append(path)
        elif path.is_dir():
            csv_paths.extend(sorted(path.rglob("*_vehicle_tracks.csv")))
            csv_paths.extend(sorted(path.rglob("*.csv")))
        else:
            raise FileNotFoundError(f"Input path not found: {path}")

    unique_paths = sorted({path.resolve() for path in csv_paths})
    if not unique_paths:
        raise FileNotFoundError("No CSV files found in the input paths.")
    return unique_paths


def create_output_dir(output_dir_text: str | None) -> Path:
    if output_dir_text:
        output_dir = Path(output_dir_text)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_dataset"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def read_raw_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        frame["source_csv"] = str(csv_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def normalize_booleans(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .map({"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0})
        .fillna(0)
        .astype(int)
    )


def clean_raw_tracks(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["red_light_phase", "in_observation_zone"]:
        if column in df.columns:
            df[column] = normalize_booleans(df[column])
        else:
            df[column] = 0

    if "vehicle_type" not in df.columns:
        df["vehicle_type"] = "other_vehicle"
    df["vehicle_type"] = df["vehicle_type"].fillna("other_vehicle").astype(str)

    required = [
        "video_id",
        "frame_id",
        "timestamp_s",
        "vehicle_id",
        "vehicle_type",
        "road_x_m",
        "road_y_m",
        "speed_mps",
        "smoothed_speed_mps",
        "acceleration_mps2",
    ]
    df = df.dropna(subset=[column for column in required if column in df.columns])

    if "confidence" in df.columns:
        df = df[df["confidence"] >= args.min_confidence]
    df = df[(df["speed_mps"] >= 0) & (df["speed_mps"] <= args.max_speed_mps)]
    df = df[df["acceleration_mps2"].abs() <= args.max_abs_acceleration_mps2]

    if "distance_to_stopline_m" not in df.columns:
        df["distance_to_stopline_m"] = np.nan
    if args.stopline_road_y_m is not None:
        missing_distance = df["distance_to_stopline_m"].isna()
        df.loc[missing_distance, "distance_to_stopline_m"] = (
            df.loc[missing_distance, "road_y_m"] - args.stopline_road_y_m
        ).abs()

    df = df.sort_values(["video_id", "vehicle_id", "timestamp_s", "frame_id"])
    track_sizes = df.groupby(["video_id", "vehicle_id"])["frame_id"].transform("count")
    df = df[track_sizes >= args.min_track_points].copy()
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(["video_id", "vehicle_id"], sort=False)

    df["track_age_s"] = grouped["timestamp_s"].transform(lambda s: s - s.min())
    df["track_age_frames"] = grouped.cumcount()
    df["delta_speed_mps"] = grouped["smoothed_speed_mps"].diff().fillna(0.0)

    df["bbox_width"] = (df["bbox_x2"] - df["bbox_x1"]).clip(lower=0)
    df["bbox_height"] = (df["bbox_y2"] - df["bbox_y1"]).clip(lower=0)
    df["bbox_area"] = df["bbox_width"] * df["bbox_height"]
    df["bbox_aspect_ratio"] = df["bbox_width"] / df["bbox_height"].replace(0, np.nan)
    df["bbox_aspect_ratio"] = df["bbox_aspect_ratio"].replace([np.inf, -np.inf], np.nan)
    df["bbox_aspect_ratio"] = df["bbox_aspect_ratio"].fillna(0.0)
    return df


def add_weak_brake_labels(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()
    comfort_decel = df["vehicle_type"].map(COMFORT_DECEL_MPS2).fillna(
        COMFORT_DECEL_MPS2["other_vehicle"]
    )
    speed = df["smoothed_speed_mps"].fillna(df["speed_mps"]).clip(lower=0)
    df["comfort_deceleration_mps2"] = comfort_decel
    df["required_braking_distance_m"] = (
        speed.pow(2) / (2 * comfort_decel)
        + speed * args.reaction_time_s
        + args.buffer_distance_m
    )

    df["latest_brake_line_m"] = np.nan
    if args.stopline_road_y_m is not None:
        df["latest_brake_line_m"] = (
            args.stopline_road_y_m - df["required_braking_distance_m"]
        )

    distance = df["distance_to_stopline_m"]
    crossed = distance.notna() & (distance <= df["required_braking_distance_m"])
    warning = distance.notna() & (
        distance <= df["required_braking_distance_m"] + args.warning_margin_m
    )
    decelerating = df["acceleration_mps2"] < -1.0

    df["brake_boundary_crossed"] = crossed.astype(int)
    df["risk_level_weak"] = "Normal"
    df.loc[warning & ~crossed & ~decelerating, "risk_level_weak"] = "Warning"
    df.loc[crossed & ~decelerating, "risk_level_weak"] = "High Risk"
    return df


def write_outputs(
    dataset: pd.DataFrame,
    raw_count: int,
    csv_paths: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    dataset_path = output_dir / "dataset.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    feature_columns_path = output_dir / "feature_columns.json"
    with feature_columns_path.open("w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, ensure_ascii=False, indent=2)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csvs": [str(path) for path in csv_paths],
        "raw_rows": raw_count,
        "dataset_rows": int(len(dataset)),
        "vehicle_tracks": int(
            dataset.groupby(["video_id", "vehicle_id"]).ngroups
            if not dataset.empty
            else 0
        ),
        "feature_columns": FEATURE_COLUMNS,
        "default_target_column": "required_braking_distance_m",
        "args": vars(args),
    }
    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_arguments()
    csv_paths = resolve_input_paths(args.input)
    output_dir = create_output_dir(args.output_dir)

    raw_tracks = read_raw_csvs(csv_paths)
    cleaned = clean_raw_tracks(raw_tracks, args)
    dataset = add_features(cleaned)
    dataset = add_weak_brake_labels(dataset, args)
    dataset = dataset.dropna(subset=["required_braking_distance_m"]).reset_index(
        drop=True
    )

    write_outputs(dataset, len(raw_tracks), csv_paths, output_dir, args)
    print(f"Dataset saved to: {output_dir / 'dataset.csv'}")
    print(f"Rows: {len(dataset)}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
