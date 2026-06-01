import csv
import json
import pandas as pd
from pathlib import Path
import torch

def _to_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x

def write_csv(output_path, rows):

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

    df = pd.DataFrame([row])
    df.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False
    )

def append_prediction_log(epoch, pred_paths, target_paths, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pred_paths = pred_paths.detach().cpu().tolist()
    target_paths = target_paths.detach().cpu().tolist()

    rows = []
    for sample_idx, (pred_path, target_path) in enumerate(zip(pred_paths, target_paths)):
        rows.append({
            "epoch": epoch,
            "sample_idx": sample_idx,
            "pred_path": pred_path,
            "target_path": target_path,
            "pred_leaf": pred_path[-1],
            "target_leaf": target_path[-1],
            "correct_path": pred_path == target_path,
            "correct_leaf": pred_path[-1] == target_path[-1],
        })

    pd.DataFrame(rows).to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )

def append_prediction_log(epoch,pred_paths,target_paths,csv_path,pred_path_probs=None,target_path_probs=None,path_scores=None,probs_by_level=None,candidate_paths=None):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pred_paths = _to_list(pred_paths)
    target_paths = _to_list(target_paths)
    pred_path_probs = _to_list(pred_path_probs)
    target_path_probs = _to_list(target_path_probs)
    path_scores = _to_list(path_scores)

    if probs_by_level is not None:
        probs_by_level = {
            level: _to_list(probs)
            for level, probs in probs_by_level.items()
        }

    rows = []
    for sample_idx, (pred_path, target_path) in enumerate(zip(pred_paths, target_paths)):
        row = {
            "epoch": epoch,
            "sample_idx": sample_idx,
            "pred_path": json.dumps(pred_path),
            "target_path": json.dumps(target_path),
            "pred_leaf": pred_path[-1],
            "target_leaf": target_path[-1],
            "correct_path": pred_path == target_path,
            "correct_leaf": pred_path[-1] == target_path[-1],
        }

        if pred_path_probs is not None:
            row["pred_path_prob"] = pred_path_probs[sample_idx]

        if target_path_probs is not None:
            row["target_path_prob"] = target_path_probs[sample_idx]

        if path_scores is not None:
            row["path_scores"] = json.dumps(path_scores[sample_idx])

        if candidate_paths is not None:
            row["candidate_paths"] = json.dumps(candidate_paths)

        if probs_by_level is not None:
            for level, probs in probs_by_level.items():
                row[f"level_{level}_probs"] = json.dumps(probs[sample_idx])

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )


def append_debug_prediction_log(
    epoch,
    pred_paths,
    target_paths,
    pred_path_probs,
    target_path_probs,
    path_scores,
    candidate_paths,
    probs_by_level,
    masks_by_level,
    masked_logits_by_level,
    raw_logits_by_level,
    csv_path,
):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pred_paths = _to_list(pred_paths)
    target_paths = _to_list(target_paths)
    pred_path_probs = _to_list(pred_path_probs)
    target_path_probs = _to_list(target_path_probs)
    path_scores = _to_list(path_scores)

    probs_by_level = {level: _to_list(v) for level, v in probs_by_level.items()}
    masks_by_level = {level: _to_list(v) for level, v in masks_by_level.items()}
    masked_logits_by_level = {level: _to_list(v) for level, v in masked_logits_by_level.items()}
    raw_logits_by_level = {level: _to_list(v) for level, v in raw_logits_by_level.items()}

    rows = []

    for sample_idx, (pred_path, target_path) in enumerate(zip(pred_paths, target_paths)):
        row = {
            "epoch": epoch,
            "sample_idx": sample_idx,

            "pred_path": json.dumps(pred_path),
            "target_path": json.dumps(target_path),
            "pred_leaf": pred_path[-1],
            "target_leaf": target_path[-1],
            "correct_path": pred_path == target_path,
            "correct_leaf": pred_path[-1] == target_path[-1],

            "pred_path_prob": pred_path_probs[sample_idx],
            "target_path_prob": target_path_probs[sample_idx],

            "path_scores": json.dumps(path_scores[sample_idx]),
            "candidate_paths": json.dumps(candidate_paths),
        }

        for level in probs_by_level.keys():
            row[f"level_{level}_probs"] = json.dumps(probs_by_level[level][sample_idx])
            row[f"level_{level}_mask"] = json.dumps(masks_by_level[level][sample_idx])
            row[f"level_{level}_masked_logits"] = json.dumps(masked_logits_by_level[level][sample_idx])
            row[f"level_{level}_raw_logits"] = json.dumps(raw_logits_by_level[level][sample_idx])

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