import csv
import json
import pandas as pd
from pathlib import Path
import torch

PLACEHOLDER = "NOT_IMPLEMENTED"


def _to_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def _json_or_placeholder(x):
    if x is None:
        return PLACEHOLDER
    return json.dumps(x)


def _safe_index(values, idx):
    if values is None:
        return None
    if idx >= len(values):
        return None
    return values[idx]


def _safe_leaf(path_or_leaf):
    if path_or_leaf is None:
        return None
    if isinstance(path_or_leaf, list):
        return path_or_leaf[-1]
    return path_or_leaf


def _level_sample(level_dict, level, sample_idx):
    if level_dict is None:
        return None
    if level not in level_dict:
        return None
    if sample_idx >= len(level_dict[level]):
        return None
    return level_dict[level][sample_idx]


def _infer_num_samples(*arrays):
    for arr in arrays:
        if arr is not None:
            return len(arr)
    return 0


def write_csv(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print(f"No rows to save: {output_path}")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} clips to: {output_path}")


def append_epoch_log(row, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    safe_row = {
        key: (PLACEHOLDER if value is None else value)
        for key, value in row.items()
    }

    df = pd.DataFrame([safe_row])
    df.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False
    )


def append_prediction_log(
    epoch,
    pred_paths=None,
    target_paths=None,
    csv_path=None,
    pred_path_probs=None,
    target_path_probs=None,
    path_scores=None,
    probs_by_level=None,
    candidate_paths=None,
    pred_leaf=None,
    target_leaf=None,
):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pred_paths = _to_list(pred_paths)
    target_paths = _to_list(target_paths)
    pred_path_probs = _to_list(pred_path_probs)
    target_path_probs = _to_list(target_path_probs)
    path_scores = _to_list(path_scores)
    pred_leaf = _to_list(pred_leaf)
    target_leaf = _to_list(target_leaf)

    if probs_by_level is None:
        probs_by_level = {}
    else:
        probs_by_level = {
            level: _to_list(probs)
            for level, probs in probs_by_level.items()
        }

    num_samples = _infer_num_samples(
        target_paths,
        pred_paths,
        target_leaf,
        pred_leaf,
        target_path_probs,
        pred_path_probs,
        path_scores,
    )

    rows = []

    for sample_idx in range(num_samples):
        pred_path_i = _safe_index(pred_paths, sample_idx)
        target_path_i = _safe_index(target_paths, sample_idx)

        pred_leaf_i = _safe_index(pred_leaf, sample_idx)
        target_leaf_i = _safe_index(target_leaf, sample_idx)

        if pred_leaf_i is None:
            pred_leaf_i = _safe_leaf(pred_path_i)

        if target_leaf_i is None:
            target_leaf_i = _safe_leaf(target_path_i)

        correct_path = (
            pred_path_i == target_path_i
            if pred_path_i is not None and target_path_i is not None
            else None
        )

        correct_leaf = (
            pred_leaf_i == target_leaf_i
            if pred_leaf_i is not None and target_leaf_i is not None
            else None
        )

        row = {
            "epoch": epoch,
            "sample_idx": sample_idx,

            "pred_path": _json_or_placeholder(pred_path_i),
            "target_path": _json_or_placeholder(target_path_i),

            "pred_leaf": PLACEHOLDER if pred_leaf_i is None else pred_leaf_i,
            "target_leaf": PLACEHOLDER if target_leaf_i is None else target_leaf_i,

            "correct_path": PLACEHOLDER if correct_path is None else correct_path,
            "correct_leaf": PLACEHOLDER if correct_leaf is None else correct_leaf,

            "pred_path_prob": PLACEHOLDER if _safe_index(pred_path_probs, sample_idx) is None else _safe_index(pred_path_probs, sample_idx),
            "target_path_prob": PLACEHOLDER if _safe_index(target_path_probs, sample_idx) is None else _safe_index(target_path_probs, sample_idx),

            "path_scores": _json_or_placeholder(_safe_index(path_scores, sample_idx)),
            "candidate_paths": _json_or_placeholder(candidate_paths),
        }

        for level, probs in probs_by_level.items():
            row[f"level_{level}_probs"] = _json_or_placeholder(
                _safe_index(probs, sample_idx)
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )


def append_debug_prediction_log(
    epoch,
    pred_paths=None,
    target_paths=None,
    pred_path_probs=None,
    target_path_probs=None,
    path_scores=None,
    candidate_paths=None,
    probs_by_level=None,
    masks_by_level=None,
    masked_logits_by_level=None,
    raw_logits_by_level=None,
    csv_path=None,
    pred_leaf=None,
    target_leaf=None,
):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pred_paths = _to_list(pred_paths)
    target_paths = _to_list(target_paths)
    pred_path_probs = _to_list(pred_path_probs)
    target_path_probs = _to_list(target_path_probs)
    path_scores = _to_list(path_scores)
    pred_leaf = _to_list(pred_leaf)
    target_leaf = _to_list(target_leaf)

    probs_by_level = {
        level: _to_list(v)
        for level, v in (probs_by_level or {}).items()
    }

    masks_by_level = {
        level: _to_list(v)
        for level, v in (masks_by_level or {}).items()
    }

    masked_logits_by_level = {
        level: _to_list(v)
        for level, v in (masked_logits_by_level or {}).items()
    }

    raw_logits_by_level = {
        level: _to_list(v)
        for level, v in (raw_logits_by_level or {}).items()
    }

    all_levels = sorted(
        set(probs_by_level.keys())
        | set(masks_by_level.keys())
        | set(masked_logits_by_level.keys())
        | set(raw_logits_by_level.keys())
    )

    num_samples = _infer_num_samples(
        target_paths,
        pred_paths,
        target_leaf,
        pred_leaf,
        target_path_probs,
        pred_path_probs,
        path_scores,
    )

    rows = []

    for sample_idx in range(num_samples):
        pred_path_i = _safe_index(pred_paths, sample_idx)
        target_path_i = _safe_index(target_paths, sample_idx)

        pred_leaf_i = _safe_index(pred_leaf, sample_idx)
        target_leaf_i = _safe_index(target_leaf, sample_idx)

        if pred_leaf_i is None:
            pred_leaf_i = _safe_leaf(pred_path_i)

        if target_leaf_i is None:
            target_leaf_i = _safe_leaf(target_path_i)

        correct_path = (
            pred_path_i == target_path_i
            if pred_path_i is not None and target_path_i is not None
            else None
        )

        correct_leaf = (
            pred_leaf_i == target_leaf_i
            if pred_leaf_i is not None and target_leaf_i is not None
            else None
        )

        row = {
            "epoch": epoch,
            "sample_idx": sample_idx,

            "pred_path": _json_or_placeholder(pred_path_i),
            "target_path": _json_or_placeholder(target_path_i),

            "pred_leaf": PLACEHOLDER if pred_leaf_i is None else pred_leaf_i,
            "target_leaf": PLACEHOLDER if target_leaf_i is None else target_leaf_i,

            "correct_path": PLACEHOLDER if correct_path is None else correct_path,
            "correct_leaf": PLACEHOLDER if correct_leaf is None else correct_leaf,

            "pred_path_prob": PLACEHOLDER if _safe_index(pred_path_probs, sample_idx) is None else _safe_index(pred_path_probs, sample_idx),
            "target_path_prob": PLACEHOLDER if _safe_index(target_path_probs, sample_idx) is None else _safe_index(target_path_probs, sample_idx),

            "path_scores": _json_or_placeholder(_safe_index(path_scores, sample_idx)),
            "candidate_paths": _json_or_placeholder(candidate_paths),
        }

        for level in all_levels:
            row[f"level_{level}_probs"] = _json_or_placeholder(
                _level_sample(probs_by_level, level, sample_idx)
            )

            row[f"level_{level}_mask"] = _json_or_placeholder(
                _level_sample(masks_by_level, level, sample_idx)
            )

            row[f"level_{level}_masked_logits"] = _json_or_placeholder(
                _level_sample(masked_logits_by_level, level, sample_idx)
            )

            row[f"level_{level}_raw_logits"] = _json_or_placeholder(
                _level_sample(raw_logits_by_level, level, sample_idx)
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )


def save_runtime_summary(
    runtime_path,
    experiment_name,
    start_ts,
    end_ts,
    runtime_seconds,
):
    runtime_path = Path(runtime_path)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "started": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "finished": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60,
        "runtime_hours": runtime_seconds / 3600,
    }

    with open(runtime_path, "w") as f:
        json.dump(summary, f, indent=4)