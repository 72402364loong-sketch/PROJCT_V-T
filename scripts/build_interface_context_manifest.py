from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.data.splits import resolve_sample_ids


def _resolve_path(project_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def _subset_lookup(samples: pd.DataFrame, split_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for subset in ["train", "val", "test"]:
        for sample_id in resolve_sample_ids(samples, split_path, subset=subset):
            lookup[str(sample_id)] = subset
    return lookup


def _window_indices_json(indices: list[int]) -> str:
    return json.dumps(indices, ensure_ascii=False)


def build_manifest(project_root: Path, split_path: Path, min_context_windows: int) -> pd.DataFrame:
    samples = pd.read_csv(project_root / "data" / "processed" / "samples.csv")
    windows = pd.read_csv(project_root / "data" / "processed" / "windows.csv")
    subset_by_sample = _subset_lookup(samples, split_path)

    for column in ["t_grasp_stable", "t_if_enter", "t_if_exit"]:
        samples[column] = pd.to_numeric(samples[column], errors="coerce")

    samples = samples.loc[
        samples[["t_grasp_stable", "t_if_enter", "t_if_exit"]].notna().all(axis=1)
    ].copy()
    samples = samples.loc[
        (samples["t_grasp_stable"] < samples["t_if_enter"])
        & (samples["t_if_enter"] < samples["t_if_exit"])
    ].copy()

    windows = windows.copy()
    windows["window_center"] = pd.to_numeric(windows["window_center"], errors="coerce")
    windows["is_stable_mask"] = pd.to_numeric(windows["is_stable_mask"], errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, object]] = []
    for sample in samples.to_dict("records"):
        sample_id = str(sample["sample_id"])
        subset = subset_by_sample.get(sample_id)
        if subset is None:
            continue

        sample_windows = windows.loc[windows["sample_id"] == sample_id].sort_values("window_start").copy()
        sample_windows = sample_windows.reset_index(drop=True)
        if sample_windows.empty:
            continue

        t_grasp_stable = float(sample["t_grasp_stable"])
        t_if_enter = float(sample["t_if_enter"])
        t_if_exit = float(sample["t_if_exit"])

        context_mask = (sample_windows["window_center"] >= t_grasp_stable) & (
            sample_windows["window_center"] < t_if_enter
        )
        stable_context_mask = context_mask & (sample_windows["is_stable_mask"] == 1)
        interface_mask = (sample_windows["window_center"] >= t_if_enter) & (
            sample_windows["window_center"] <= t_if_exit
        )
        interface_mask &= sample_windows["phase_label"].astype(str) == "Interface"

        context_indices = sample_windows.index[context_mask].tolist()
        stable_context_indices = sample_windows.index[stable_context_mask].tolist()
        interface_indices = sample_windows.index[interface_mask].tolist()

        used_fallback_context = False
        selected_context_indices = stable_context_indices
        if not selected_context_indices:
            fallback_candidates = sample_windows.loc[
                context_mask & (sample_windows["phase_label"].astype(str) != "Interface")
            ]
            selected_context_indices = fallback_candidates.index.tolist()[-max(1, int(min_context_windows)) :]
            used_fallback_context = bool(selected_context_indices)

        rows.append(
            {
                "sample_id": sample_id,
                "object_id": str(sample["object_id"]),
                "subset": subset,
                "context_window_indices_json": _window_indices_json(context_indices),
                "stable_context_window_indices_json": _window_indices_json(stable_context_indices),
                "selected_context_window_indices_json": _window_indices_json(selected_context_indices),
                "interface_window_indices_json": _window_indices_json(interface_indices),
                "context_count": int(len(context_indices)),
                "stable_context_count": int(len(stable_context_indices)),
                "selected_context_count": int(len(selected_context_indices)),
                "interface_count": int(len(interface_indices)),
                "has_context": int(bool(selected_context_indices)),
                "used_fallback_context": int(used_fallback_context),
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    return manifest.sort_values(["subset", "object_id", "sample_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--output",
        default="data/processed/interface_context/fold1_manifest.csv",
        help="CSV path relative to project root unless absolute.",
    )
    parser.add_argument(
        "--min-context-windows",
        type=int,
        default=2,
        help="Fallback context length when stable context is unavailable.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    split_path = _resolve_path(project_root, args.split)
    output_path = _resolve_path(project_root, args.output)
    if split_path is None or output_path is None:
        raise RuntimeError("Both --split and --output must resolve to valid paths.")

    manifest = build_manifest(project_root, split_path, min_context_windows=int(args.min_context_windows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8")

    summary = {
        "output_path": str(output_path),
        "rows": int(len(manifest)),
        "has_context_rows": int(manifest["has_context"].sum()) if not manifest.empty else 0,
        "stable_context_rows": int((manifest["stable_context_count"] > 0).sum()) if not manifest.empty else 0,
        "fallback_rows": int(manifest["used_fallback_context"].sum()) if not manifest.empty else 0,
    }
    print(summary)


if __name__ == "__main__":
    main()
