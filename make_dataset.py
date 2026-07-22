import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets"

DEFAULT_FEATURE_COLUMNS = [
    "区域瞬时事故概率P_A(t)",
    "车流密度rho(t)（veh/m^2）",
    "区域车辆数N(t)",
    "有效风险车辆数",
    "最大单车风险P_i(t)",
    "最大车辆对风险P_ij(t)",
    "risk_type_long",
    "risk_type_lat",
]

NUMERIC_COLUMNS = [
    "frame_id",
    "timestamp_s",
    "区域车辆数N(t)",
    "有效风险车辆数",
    "研究区域面积S_A（m^2）",
    "车流密度rho(t)（veh/m^2）",
    "区域瞬时事故概率P_A(t)",
    "最大单车风险P_i(t)",
    "最大车辆对风险P_ij(t)",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sliding-window sequence datasets from frame risk CSV files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_INPUT_ROOT)],
        help="CSV files or directories containing *_frame_risk_timeseries.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dataset output directory. Defaults to datasets/<timestamp>_risk_sequence.",
    )
    parser.add_argument(
        "--window-size",
        default=60,
        type=int,
        help="Number of past frames in each input sequence.",
    )
    parser.add_argument(
        "--horizon-size",
        default=60,
        type=int,
        help="Number of future frames used to build the prediction label.",
    )
    parser.add_argument(
        "--risk-threshold",
        default=0.7,
        type=float,
        help="Weak label threshold for future max P_A(t).",
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="Sliding-window stride in frames.",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.15,
        type=float,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        default=0.15,
        type=float,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed used when shuffling sequences.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep chronological order before splitting.",
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
            csv_paths.extend(sorted(path.rglob("*_frame_risk_timeseries.csv")))
        else:
            raise FileNotFoundError(f"Input path not found: {path}")

    unique_paths = sorted({path.resolve() for path in csv_paths})
    if not unique_paths:
        raise FileNotFoundError("No *_frame_risk_timeseries.csv files found.")
    return unique_paths


def create_output_dir(output_dir_text: str | None) -> Path:
    if output_dir_text:
        output_dir = Path(output_dir_text)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_risk_sequence"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def read_frame_risk_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        frame["source_csv"] = str(csv_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_frame_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    required = [
        "video_id",
        "frame_id",
        "timestamp_s",
        "region_id",
        "区域瞬时事故概率P_A(t)",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required frame risk columns: {missing}")

    df["region_name"] = df.get("region_name", df["region_id"]).fillna(df["region_id"])
    df["最大风险类型"] = df.get("最大风险类型", "").fillna("").astype(str)
    df["risk_type_long"] = (df["最大风险类型"] == "long").astype(float)
    df["risk_type_lat"] = (df["最大风险类型"] == "lat").astype(float)

    for column in DEFAULT_FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df = df.dropna(subset=required)
    df = df.sort_values(["source_csv", "video_id", "region_id", "frame_id"])
    return df.reset_index(drop=True)


def split_indices(
    sample_count: int,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(sample_count)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_count = int(round(sample_count * test_ratio))
    val_count = int(round(sample_count * val_ratio))
    train_count = sample_count - val_count - test_count
    if train_count <= 0:
        raise ValueError("Not enough sequences for the requested split ratios.")

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return train_idx, val_idx, test_idx


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    horizon_size: int,
    risk_threshold: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences = []
    labels = []
    rows = []

    group_columns = ["source_csv", "video_id", "region_id"]
    for (source_csv, video_id, region_id), group in df.groupby(group_columns, sort=False):
        group = group.sort_values("frame_id").reset_index(drop=True)
        features = group[feature_columns].to_numpy(dtype=np.float32)
        risk_values = group["区域瞬时事故概率P_A(t)"].to_numpy(dtype=np.float32)
        max_start = len(group) - window_size - horizon_size + 1
        if max_start <= 0:
            continue

        for start in range(0, max_start, stride):
            end = start + window_size
            future_end = end + horizon_size
            future_max_risk = float(np.max(risk_values[end:future_end]))
            label = float(future_max_risk >= risk_threshold)
            sequences.append(features[start:end])
            labels.append(label)
            rows.append(
                {
                    "sample_id": len(rows),
                    "source_csv": source_csv,
                    "video_id": video_id,
                    "region_id": region_id,
                    "start_frame": int(group.loc[start, "frame_id"]),
                    "end_frame": int(group.loc[end - 1, "frame_id"]),
                    "label_start_frame": int(group.loc[end, "frame_id"]),
                    "label_end_frame": int(group.loc[future_end - 1, "frame_id"]),
                    "end_timestamp_s": float(group.loc[end - 1, "timestamp_s"]),
                    "future_max_risk": future_max_risk,
                    "label": label,
                }
            )

    if not sequences:
        raise ValueError(
            "No sequences were created. Try reducing --window-size or --horizon-size."
        )

    return (
        np.stack(sequences).astype(np.float32),
        np.array(labels, dtype=np.float32),
        pd.DataFrame(rows),
    )


def standardize_by_train(
    X: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = X[train_idx].reshape(-1, X.shape[-1])
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((X - mean) / std).astype(np.float32), mean, std


def save_split(
    output_dir: Path,
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    index_df: pd.DataFrame,
    indices: np.ndarray,
) -> None:
    np.savez_compressed(
        output_dir / f"{split_name}.npz",
        X=X[indices],
        y=y[indices],
        sample_id=index_df.iloc[indices]["sample_id"].to_numpy(dtype=np.int64),
    )


def main() -> None:
    args = parse_arguments()
    if args.window_size <= 0 or args.horizon_size <= 0 or args.stride <= 0:
        raise ValueError("window-size, horizon-size and stride must be positive.")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Split ratios must be non-negative and sum to less than 1.")

    csv_paths = resolve_input_paths(args.input)
    output_dir = create_output_dir(args.output_dir)

    raw_df = read_frame_risk_csvs(csv_paths)
    frame_risk = clean_frame_risk(raw_df)
    feature_columns = [column for column in DEFAULT_FEATURE_COLUMNS if column in frame_risk.columns]
    X, y, index_df = build_sequences(
        df=frame_risk,
        feature_columns=feature_columns,
        window_size=args.window_size,
        horizon_size=args.horizon_size,
        risk_threshold=args.risk_threshold,
        stride=args.stride,
    )

    train_idx, val_idx, test_idx = split_indices(
        sample_count=len(y),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        shuffle=not args.no_shuffle,
    )
    X, mean, std = standardize_by_train(X, train_idx)

    save_split(output_dir, "train", X, y, index_df, train_idx)
    save_split(output_dir, "val", X, y, index_df, val_idx)
    save_split(output_dir, "test", X, y, index_df, test_idx)
    index_df.to_csv(output_dir / "sequence_index.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, ensure_ascii=False, indent=2)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csvs": [str(path) for path in csv_paths],
        "raw_rows": int(len(raw_df)),
        "frame_risk_rows": int(len(frame_risk)),
        "samples": int(len(y)),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum()),
        "window_size": args.window_size,
        "horizon_size": args.horizon_size,
        "risk_threshold": args.risk_threshold,
        "feature_columns": feature_columns,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "args": vars(args),
    }
    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Sequence dataset saved to: {output_dir}")
    print(f"Samples: {len(y)} | positives: {int(y.sum())} | features: {len(feature_columns)}")
    print(f"Train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")


if __name__ == "__main__":
    main()
