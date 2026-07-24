import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model


ROOT = Path("experiments/chapter4/20260724_174209_chapter4")
TABLES = ROOT / "tables"
SOURCE_DATASET = Path("datasets/20260724_174006_risk_sequence")
MODELS = ["rnn", "gru", "lstm", "transformer", "mamba", "mtpnet"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_baseline_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        model_dir = ROOT / "models" / model
        metadata_path = model_dir / "training_metadata.json"
        checkpoint_path = model_dir / "best_model.pt"
        row = {
            "experiment": "baseline",
            "model": model,
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1_score": "",
            "roc_auc": "",
            "loss": "",
            "best_epoch": "",
            "best_val_loss": "",
            "status": "missing",
            "model_path": "",
        }
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            metrics = metadata.get("test_metrics", {})
            row.update(
                {
                    "accuracy": metrics.get("accuracy", ""),
                    "precision": metrics.get("precision", ""),
                    "recall": metrics.get("recall", ""),
                    "f1_score": metrics.get("f1", ""),
                    "roc_auc": metrics.get("auc", ""),
                    "loss": metrics.get("loss", ""),
                    "best_epoch": metadata.get("best_epoch", ""),
                    "best_val_loss": metadata.get("best_val_loss", ""),
                    "status": "completed",
                    "model_path": metadata.get("model_path", str(checkpoint_path.resolve())),
                }
            )
        elif checkpoint_path.exists():
            row.update(
                {
                    "status": "checkpoint_only_no_test_metadata",
                    "model_path": str(checkpoint_path.resolve()),
                }
            )
        rows.append(row)
    return rows


def write_baseline_table(rows: list[dict]) -> None:
    write_csv(
        TABLES / "table_4_3_baseline_comparison.csv",
        rows,
        [
            "experiment",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "loss",
            "best_epoch",
            "best_val_loss",
            "status",
            "model_path",
        ],
    )


def write_ablation_table(source_metadata: dict) -> None:
    specs = [
        ("mtpnet_no_temporal", "Uses only the latest frame mechanism feature.", "full"),
        ("mtpnet_no_pairwise_risk", "Removes maximum pairwise risk feature.", "no_pairwise_risk"),
        ("mtpnet_no_risk_type", "Removes dominant conflict type indicators.", "no_risk_type"),
        (
            "mtpnet_no_mechanism_features",
            "Keeps traffic state features and removes mechanism risk features.",
            "traffic_state_only",
        ),
        ("mtpnet_full", "Complete MTPNet.", "full"),
    ]
    rows = []
    for name, description, variant in specs:
        dataset_dir = SOURCE_DATASET if variant == "full" else ROOT / "ablated_datasets" / variant
        metadata_path = dataset_dir / "dataset_metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else source_metadata
        rows.append(
            {
                "experiment": name,
                "description": description,
                "dataset_variant": variant,
                "feature_dim": len(metadata.get("feature_columns", [])),
                "samples": metadata.get("samples", ""),
                "positive_samples": metadata.get("positive_samples", ""),
                "positive_ratio": (
                    metadata.get("positive_samples", 0) / metadata.get("samples", 1)
                    if metadata.get("samples")
                    else ""
                ),
                "accuracy": "",
                "precision": "",
                "recall": "",
                "f1_score": "",
                "roc_auc": "",
                "loss": "",
                "best_epoch": "",
                "best_val_loss": "",
                "status": "dataset_ready_training_not_completed",
                "model_path": "",
            }
        )
    write_csv(
        TABLES / "table_4_4_ablation_study.csv",
        rows,
        [
            "experiment",
            "description",
            "dataset_variant",
            "feature_dim",
            "samples",
            "positive_samples",
            "positive_ratio",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "loss",
            "best_epoch",
            "best_val_loss",
            "status",
            "model_path",
        ],
    )


def write_sensitivity_table(source_metadata: dict, baseline_rows: list[dict]) -> None:
    completed = [row for row in baseline_rows if row["status"] == "completed"]
    best = max(completed, key=lambda row: float(row["f1_score"]))
    write_csv(
        TABLES / "table_4_5_sensitivity.csv",
        [
            {
                "sequence_length_T": source_metadata.get("window_size", ""),
                "prediction_horizon_H": source_metadata.get("horizon_size", ""),
                "risk_threshold_gamma": source_metadata.get("risk_threshold", ""),
                "model": best["model"],
                "recall": best["recall"],
                "f1_score": best["f1_score"],
                "roc_auc": best["roc_auc"],
                "notes": "Baseline setting from existing chapter4 run; additional T/H/gamma sweeps not present.",
            }
        ],
        [
            "sequence_length_T",
            "prediction_horizon_H",
            "risk_threshold_gamma",
            "model",
            "recall",
            "f1_score",
            "roc_auc",
            "notes",
        ],
    )


def write_qualitative_cases(source_metadata: dict) -> None:
    rows = []
    risk_col = "区域瞬时事故概率P_A(t)"
    for csv_path in source_metadata.get("input_csvs", []):
        path = Path(csv_path)
        if not path.exists():
            continue
        frame_risk = pd.read_csv(path, encoding="utf-8-sig")
        if frame_risk.empty or risk_col not in frame_risk.columns:
            continue
        frame_risk[risk_col] = pd.to_numeric(frame_risk[risk_col], errors="coerce").fillna(0.0)
        peak = frame_risk.sort_values(risk_col, ascending=False).iloc[0]
        frame_id = int(peak.get("frame_id", 0))
        rows.append(
            {
                "case_id": f"C{len(rows) + 1:02d}",
                "video_id": peak.get("video_id", path.stem.replace("_frame_risk_timeseries", "")),
                "start_frame": max(0, frame_id - 60),
                "end_frame": frame_id,
                "risk_source": peak.get("最大风险类型", ""),
                "instant_risk_peak": float(peak[risk_col]),
                "mtpnet_probability_peak": "",
                "warning_lead_time_s": "",
                "figure_path": str((path.parent / "plots" / "region_risk_timeseries.png").resolve()),
                "notes": (
                    f"Peak mechanism risk in {path.parent.name}, "
                    f"region={peak.get('region_id', '')}, t={peak.get('timestamp_s', '')}s. "
                    "MTPNet demo probability not available in chapter4 artifacts."
                ),
            }
        )
    write_csv(
        TABLES / "table_4_6_qualitative_cases.csv",
        rows,
        [
            "case_id",
            "video_id",
            "start_frame",
            "end_frame",
            "risk_source",
            "instant_risk_peak",
            "mtpnet_probability_peak",
            "warning_lead_time_s",
            "figure_path",
            "notes",
        ],
    )


def write_latency_table(source_metadata: dict) -> None:
    rows = []
    for model in MODELS:
        checkpoint_path = ROOT / "models" / model / "best_model.pt"
        if not checkpoint_path.exists():
            continue
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_name = checkpoint.get("model_name", model)
            input_size = int(checkpoint["input_size"])
            hidden_size = int(checkpoint.get("hidden_size", 64))
            num_layers = int(checkpoint.get("num_layers", 2))
            dropout = float(checkpoint.get("dropout", 0.1))
            sequence_length = int(checkpoint.get("sequence_length", source_metadata.get("window_size", 60)))
            network = build_model(model_name, input_size, hidden_size, num_layers, dropout)
            network.load_state_dict(checkpoint["model_state_dict"])
            network.eval()
            sample = torch.randn(1, sequence_length, input_size)
            with torch.no_grad():
                for _ in range(30):
                    network(sample)
                durations = []
                for _ in range(200):
                    start = time.perf_counter()
                    network(sample)
                    durations.append((time.perf_counter() - start) * 1000.0)
            values = np.array(durations, dtype=np.float64)
            rows.append(
                {
                    "model": model_name,
                    "sequence_length": sequence_length,
                    "input_size": input_size,
                    "mean_ms": float(np.mean(values)),
                    "p50_ms": float(np.percentile(values, 50)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "fps_equivalent": float(1000.0 / max(np.mean(values), 1e-9)),
                    "repeat": 200,
                    "warmup": 30,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "status": "measured_cpu",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model,
                    "sequence_length": "",
                    "input_size": "",
                    "mean_ms": "",
                    "p50_ms": "",
                    "p95_ms": "",
                    "fps_equivalent": "",
                    "repeat": 200,
                    "warmup": 30,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "status": f"failed: {exc}",
                }
            )
    write_csv(
        TABLES / "table_4_7_temporal_model_latency.csv",
        rows,
        [
            "model",
            "sequence_length",
            "input_size",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "fps_equivalent",
            "repeat",
            "warmup",
            "checkpoint",
            "status",
        ],
    )


def write_system_and_deployment_tables() -> None:
    write_csv(
        TABLES / "table_4_8_system_latency.csv",
        [
            {"module": "Vehicle detection", "mean_ms": "", "p50_ms": "", "p95_ms": "", "record_status": "not recorded by current video_analyzer run"},
            {"module": "Vehicle tracking", "mean_ms": "", "p50_ms": "", "p95_ms": "", "record_status": "not recorded by current video_analyzer run"},
            {"module": "Coordinate recovery", "mean_ms": "", "p50_ms": "", "p95_ms": "", "record_status": "not recorded by current video_analyzer run"},
            {"module": "Mechanism risk calculation", "mean_ms": "", "p50_ms": "", "p95_ms": "", "record_status": "not recorded by current video_analyzer run"},
            {"module": "MTPNet prediction", "mean_ms": "", "p50_ms": "", "p95_ms": "", "record_status": "see table_4_7_temporal_model_latency.csv for temporal-model-only latency"},
            {"module": "Total VRPS", "mean_ms": "", "p50_ms": "", "p95_ms": "", "record_status": "requires per-module profiling instrumentation"},
        ],
        ["module", "mean_ms", "p50_ms", "p95_ms", "record_status"],
    )
    write_csv(
        TABLES / "table_4_9_low_intrusion_deployment.csv",
        [
            {"item": "Existing camera reuse", "record": "Supported; experiments use roadside/video files as input without adding roadside sensing hardware."},
            {"item": "Existing speed sensor reuse", "record": "Not required for core pipeline; speed is estimated from tracking trajectories after calibration."},
            {"item": "New hardware requirement", "record": "No new in-road hardware in the software workflow; requires video source and one-time region/calibration JSON."},
            {"item": "Lane closure or traffic organization change", "record": "Not required by the video-analysis workflow."},
            {"item": "Single-region calibration time", "record": "Not measured in current artifacts; calibration file is reused during analysis."},
            {"item": "Performance change without speed correction", "record": "Not measured in current artifacts; requires an ablation run disabling speed/coordinate correction."},
        ],
        ["item", "record"],
    )


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    source_metadata = read_json(SOURCE_DATASET / "dataset_metadata.json")
    baseline_rows = collect_baseline_rows()
    write_baseline_table(baseline_rows)
    write_ablation_table(source_metadata)
    write_sensitivity_table(source_metadata, baseline_rows)
    write_qualitative_cases(source_metadata)
    write_latency_table(source_metadata)
    write_system_and_deployment_tables()
    print(f"updated {TABLES.resolve()}")


if __name__ == "__main__":
    main()
