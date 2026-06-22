
"""
InspectData helper functions for hierarchical audio classification experiments.

Expected experiment directory structure:
    experiment_dir/
        metrics.csv
        predictions.csv

metrics.csv example columns:
    epoch, train_loss, val_loss, train_path_log_prob, train_path_prob,
    val_path_log_prob, val_path_acc, val_leaf_acc, val_level_0_acc, ...

predictions.csv example columns:
    epoch, sample_idx, pred_path, target_path, pred_leaf, target_leaf,
    correct_path, correct_leaf

Usage in notebook:
    from inspect_data_helpers import *

    exp_dir = "results/ar_hierarchy_masked"
    metrics, preds = load_experiment(exp_dir)

    plot_training_dynamics(metrics)
    best_epoch = choose_best_epoch(metrics, metric="val_leaf_acc")

    epoch_preds = filter_predictions_by_epoch(preds, best_epoch)
    plot_leaf_confusion_matrix(epoch_preds, index_dict=hierarchy_index_dict)
    plot_level_confusion_matrix(epoch_preds, level=0, index_dict=hierarchy_index_dict)

    plot_hierarchical_error_breakdown(epoch_preds)
    plot_accuracy_per_depth(metrics, epoch=best_epoch)
    path_consistency_report(epoch_preds, allowed_edges={(0,1): allowed_01, (1,2): allowed_12})
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Loading / parsing
# ---------------------------------------------------------------------

def _parse_path(x):
    """Parse '[0, 1, 3]' strings into Python lists."""
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return list(ast.literal_eval(x))
        except Exception:
            return []
    return []


def load_experiment(experiment_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load metrics.csv and predictions.csv from an experiment directory.

    Returns
    -------
    metrics : pd.DataFrame
    predictions : pd.DataFrame
    """
    experiment_dir = Path(experiment_dir)
    metrics_path = experiment_dir / "metrics.csv"
    preds_path = experiment_dir / "predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics.csv at: {metrics_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Could not find predictions.csv at: {preds_path}")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(preds_path)

    if "pred_path" in predictions.columns:
        predictions["pred_path_parsed"] = predictions["pred_path"].apply(_parse_path)
    if "target_path" in predictions.columns:
        predictions["target_path_parsed"] = predictions["target_path"].apply(_parse_path)

    return metrics, predictions


def get_level_acc_columns(metrics: pd.DataFrame) -> list[str]:
    """Find columns like val_level_0_acc, val_level_1_acc, ... sorted by level number."""
    cols = []
    for c in metrics.columns:
        m = re.fullmatch(r"val_level_(\d+)_acc", c)
        if m:
            cols.append((int(m.group(1)), c))
    return [c for _, c in sorted(cols)]


def infer_num_levels_from_predictions(predictions: pd.DataFrame) -> int:
    """Infer hierarchy depth from pred_path/target_path."""
    path_col = None
    if "target_path_parsed" in predictions.columns:
        path_col = "target_path_parsed"
    elif "pred_path_parsed" in predictions.columns:
        path_col = "pred_path_parsed"

    if path_col is None or len(predictions) == 0:
        return 0

    return int(max(len(p) for p in predictions[path_col]))


def filter_predictions_by_epoch(predictions: pd.DataFrame, epoch: int) -> pd.DataFrame:
    """Return predictions for a selected epoch."""
    if "epoch" not in predictions.columns:
        raise ValueError("predictions.csv must contain an 'epoch' column.")
    out = predictions[predictions["epoch"] == epoch].copy()
    if out.empty:
        raise ValueError(f"No predictions found for epoch={epoch}.")
    return out


def choose_best_epoch(
    metrics: pd.DataFrame,
    metric: str = "val_leaf_acc",
    mode: str = "max",
) -> int:
    """
    Choose a best epoch automatically.

    Examples
    --------
    choose_best_epoch(metrics, "val_leaf_acc", "max")
    choose_best_epoch(metrics, "val_loss", "min")
    """
    if metric not in metrics.columns:
        raise ValueError(f"Metric '{metric}' not found. Available columns: {list(metrics.columns)}")

    if mode == "max":
        idx = metrics[metric].idxmax()
    elif mode == "min":
        idx = metrics[metric].idxmin()
    else:
        raise ValueError("mode must be 'max' or 'min'.")

    return int(metrics.loc[idx, "epoch"])


# ---------------------------------------------------------------------
# Training dynamics
# ---------------------------------------------------------------------

def plot_loss_curves(metrics: pd.DataFrame, figsize=(8, 5)):
    """Plot train_loss and val_loss over epochs."""
    fig, ax = plt.subplots(figsize=figsize)

    if "train_loss" in metrics.columns:
        ax.plot(metrics["epoch"], metrics["train_loss"], marker="o", label="Train loss")
    if "val_loss" in metrics.columns:
        ax.plot(metrics["epoch"], metrics["val_loss"], marker="o", label="Validation loss")

    ax.set_title("Loss curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_path_probability_curves(metrics: pd.DataFrame, figsize=(8, 5)):
    """
    Plot path probability / log-probability curves if present.

    Handles:
        train_path_log_prob
        val_path_log_prob
        train_path_prob
        val_path_prob
    """
    candidate_cols = [
        "train_path_log_prob",
        "val_path_log_prob",
        "train_path_prob",
        "val_path_prob",
    ]

    found = [c for c in candidate_cols if c in metrics.columns]
    if not found:
        print("No path probability/log-probability columns found.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for c in found:
        ax.plot(metrics["epoch"], metrics[c], marker="o", label=c)

    ax.set_title("Path probability / log-probability curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_leaf_vs_path_accuracy(metrics: pd.DataFrame, figsize=(8, 5)):
    """Plot validation leaf accuracy and path accuracy over epochs."""
    fig, ax = plt.subplots(figsize=figsize)

    if "val_leaf_acc" in metrics.columns:
        ax.plot(metrics["epoch"], metrics["val_leaf_acc"], marker="o", label="Validation leaf accuracy")
    if "val_path_acc" in metrics.columns:
        ax.plot(metrics["epoch"], metrics["val_path_acc"], marker="o", label="Validation path accuracy")

    ax.set_title("Leaf accuracy vs path accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_per_level_accuracy(metrics: pd.DataFrame, figsize=(8, 5)):
    """Plot all val_level_n_acc columns dynamically."""
    level_cols = get_level_acc_columns(metrics)

    if not level_cols:
        print("No val_level_n_acc columns found.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for c in level_cols:
        level = re.search(r"val_level_(\d+)_acc", c).group(1)
        ax.plot(metrics["epoch"], metrics[c], marker="o", label=f"Level {level}")

    ax.set_title("Validation accuracy per hierarchy level")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_training_dynamics(metrics: pd.DataFrame):
    """Convenience function for all training-dynamics plots."""
    plot_loss_curves(metrics)
    plot_path_probability_curves(metrics)
    plot_leaf_vs_path_accuracy(metrics)
    plot_per_level_accuracy(metrics)


# ---------------------------------------------------------------------
# Confusion matrices and metrics
# ---------------------------------------------------------------------

def _safe_div(a, b):
    return a / b if b != 0 else np.nan


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, list[int]]:
    """Compute confusion matrix without sklearn dependency."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = list(labels)

    idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    return cm, labels


def classification_metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    """
    Compute accuracy, macro precision, macro recall, macro specificity, macro F1,
    plus per-class values.

    Specificity for class k:
        TN / (TN + FP)
    """
    cm = np.asarray(cm)
    total = cm.sum()
    correct = np.trace(cm)

    accuracy = _safe_div(correct, total)

    per_class = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan

        per_class.append({
            "class_index": k,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        })

    per_class_df = pd.DataFrame(per_class)

    return {
        "accuracy": accuracy,
        "macro_precision": per_class_df["precision"].mean(skipna=True),
        "macro_recall": per_class_df["recall"].mean(skipna=True),
        "macro_specificity": per_class_df["specificity"].mean(skipna=True),
        "macro_f1": per_class_df["f1"].mean(skipna=True),
        "per_class": per_class_df,
    }


def print_classification_metrics(metrics_dict: Dict[str, Any], title: str = "Metrics"):
    """Pretty-print summary metrics from classification_metrics_from_cm."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy:          {metrics_dict['accuracy']:.4f}")
    print(f"Macro precision:   {metrics_dict['macro_precision']:.4f}")
    print(f"Macro recall:      {metrics_dict['macro_recall']:.4f}")
    print(f"Macro specificity: {metrics_dict['macro_specificity']:.4f}")
    print(f"Macro F1-score:    {metrics_dict['macro_f1']:.4f}")


def _labels_to_names(labels, level=None, index_dict=None):
    if index_dict is None:
        return [str(x) for x in labels]

    if level is None:
        mapping = index_dict
    else:
        mapping = index_dict.get(level, {})

    return [mapping.get(x, str(x)) for x in labels]


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
    figsize=(8, 7),
    print_metrics: bool = True,
):
    """
    Generic confusion matrix plot.

    Rows = true class
    Columns = predicted class
    """
    cm, used_labels = compute_confusion_matrix(y_true, y_pred, labels=labels)
    cm_display = cm.astype(float)

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_display, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0)

    if label_names is None:
        label_names = [str(x) for x in used_labels]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized count" if normalize else "Count")

    fmt = ".2f" if normalize else "d"
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            text = format(val, fmt) if normalize else format(int(val), fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    metrics_dict = classification_metrics_from_cm(cm)
    if print_metrics:
        print_classification_metrics(metrics_dict, title=f"{title} metrics")

    return cm, used_labels, metrics_dict


def plot_leaf_confusion_matrix(
    predictions_epoch: pd.DataFrame,
    index_dict: Optional[Dict[int, Dict[int, str]]] = None,
    normalize: bool = False,
    print_metrics: bool = True,
    name=""
):
    """Plot confusion matrix for leaf predictions."""
    if "target_leaf" not in predictions_epoch.columns or "pred_leaf" not in predictions_epoch.columns:
        raise ValueError("predictions must contain target_leaf and pred_leaf columns.")

    y_true = predictions_epoch["target_leaf"].astype(int).to_numpy()
    y_pred = predictions_epoch["pred_leaf"].astype(int).to_numpy()

    leaf_level = max(index_dict.keys()) if index_dict else None
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    names = _labels_to_names(labels, level=leaf_level, index_dict=index_dict)

    return plot_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        label_names=names,
        normalize=normalize,
        title=f"{name} Leaf confusion matrix",
        print_metrics=print_metrics,
    )


def get_level_targets_preds(predictions_epoch: pd.DataFrame, level: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract target/pred labels at a given hierarchy level from parsed paths."""
    if "target_path_parsed" not in predictions_epoch.columns or "pred_path_parsed" not in predictions_epoch.columns:
        raise ValueError("Expected target_path_parsed and pred_path_parsed columns. Use load_experiment().")

    y_true, y_pred = [], []
    for t_path, p_path in zip(predictions_epoch["target_path_parsed"], predictions_epoch["pred_path_parsed"]):
        if len(t_path) > level and len(p_path) > level:
            y_true.append(int(t_path[level]))
            y_pred.append(int(p_path[level]))

    return np.asarray(y_true), np.asarray(y_pred)


def plot_level_confusion_matrix(
    predictions_epoch: pd.DataFrame,
    level: int,
    index_dict: Optional[Dict[int, Dict[int, str]]] = None,
    normalize: bool = False,
    print_metrics: bool = True,
):
    """Plot confusion matrix for a specific hierarchy level."""
    y_true, y_pred = get_level_targets_preds(predictions_epoch, level)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    names = _labels_to_names(labels, level=level, index_dict=index_dict)

    return plot_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        label_names=names,
        normalize=normalize,
        title=f"Level {level} confusion matrix",
        print_metrics=print_metrics,
    )


def plot_all_level_confusion_matrices(
    predictions_epoch: pd.DataFrame,
    index_dict: Optional[Dict[int, Dict[int, str]]] = None,
    normalize: bool = False,
):
    """Plot one confusion matrix per hierarchy level."""
    n_levels = infer_num_levels_from_predictions(predictions_epoch)

    results = {}
    for level in range(n_levels):
        results[level] = plot_level_confusion_matrix(
            predictions_epoch,
            level=level,
            index_dict=index_dict,
            normalize=normalize,
        )
    return results


# ---------------------------------------------------------------------
# Hierarchical confusion / visualizations across levels
# ---------------------------------------------------------------------

def make_hierarchical_confusion_table(
    predictions_epoch: pd.DataFrame,
    index_dict: Optional[Dict[int, Dict[int, str]]] = None,
) -> pd.DataFrame:
    """
    Create a table showing true path vs predicted path counts.

    This is useful as a compact hierarchical confusion matrix:
        true_path_name -> pred_path_name -> count
    """
    rows = []

    for _, row in predictions_epoch.iterrows():
        t_path = row["target_path_parsed"]
        p_path = row["pred_path_parsed"]

        true_parts = []
        pred_parts = []

        for level, (t, p) in enumerate(zip(t_path, p_path)):
            if index_dict and level in index_dict:
                true_parts.append(index_dict[level].get(t, str(t)))
                pred_parts.append(index_dict[level].get(p, str(p)))
            else:
                true_parts.append(str(t))
                pred_parts.append(str(p))

        rows.append({
            "true_path": " / ".join(true_parts),
            "pred_path": " / ".join(pred_parts),
        })

    table = (
        pd.DataFrame(rows)
        .value_counts(["true_path", "pred_path"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return table


def plot_top_hierarchical_confusions(
    predictions_epoch: pd.DataFrame,
    index_dict: Optional[Dict[int, Dict[int, str]]] = None,
    top_k: int = 20,
    only_errors: bool = True,
    figsize=(12, 7),
):
    """
    Plot the most common full-path confusions.

    This is a practical substitute for a huge path-level confusion matrix.
    """
    table = make_hierarchical_confusion_table(predictions_epoch, index_dict=index_dict)

    if only_errors:
        table = table[table["true_path"] != table["pred_path"]]

    if table.empty:
        print("No hierarchical confusions to show.")
        return table

    top = table.head(top_k).copy()
    top["label"] = top["true_path"] + "  →  " + top["pred_path"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["label"][::-1], top["count"][::-1])
    ax.set_title(f"Top {min(top_k, len(top))} hierarchical path confusions")
    ax.set_xlabel("Count")
    ax.set_ylabel("True path → Predicted path")
    plt.tight_layout()
    plt.show()

    return table


def plot_path_correctness_by_level(
    predictions_epoch: pd.DataFrame,
    figsize=(8, 5),
):
    """
    Plot fraction of samples correct at each level.

    This recomputes level accuracies from predictions.csv instead of metrics.csv.
    """
    n_levels = infer_num_levels_from_predictions(predictions_epoch)

    accs = []
    labels = []

    for level in range(n_levels):
        y_true, y_pred = get_level_targets_preds(predictions_epoch, level)
        accs.append(np.mean(y_true == y_pred))
        labels.append(f"Level {level}")

    if "correct_leaf" in predictions_epoch.columns:
        accs.append(predictions_epoch["correct_leaf"].astype(bool).mean())
        labels.append("Leaf")

    if "correct_path" in predictions_epoch.columns:
        accs.append(predictions_epoch["correct_path"].astype(bool).mean())
        labels.append("Path")

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, accs)
    ax.set_title("Accuracy per hierarchy depth")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({"depth": labels, "accuracy": accs})


# ---------------------------------------------------------------------
# Hierarchical error analysis
# ---------------------------------------------------------------------

def first_error_level(target_path: Sequence[int], pred_path: Sequence[int]) -> Optional[int]:
    """
    Return the first hierarchy level where prediction differs from target.

    Returns None if full path is correct.
    """
    for level, (t, p) in enumerate(zip(target_path, pred_path)):
        if t != p:
            return level
    if len(target_path) != len(pred_path):
        return min(len(target_path), len(pred_path))
    return None


def hierarchical_error_breakdown(predictions_epoch: pd.DataFrame) -> pd.DataFrame:
    """
    Count where errors first appear in the hierarchy.

    Interpretation:
        first_error_level = 0 means the coarse class was already wrong.
        first_error_level = 1 means level 0 was correct, but level 1 was wrong.
        etc.
    """
    rows = []
    for _, row in predictions_epoch.iterrows():
        level = first_error_level(row["target_path_parsed"], row["pred_path_parsed"])
        rows.append({
            "first_error_level": level,
            "is_correct_path": level is None,
        })

    df = pd.DataFrame(rows)

    counts = (
        df["first_error_level"]
        .fillna("correct_path")
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )

    counts["fraction"] = counts["count"] / len(df)
    return counts


def plot_hierarchical_error_breakdown(
    predictions_epoch: pd.DataFrame,
    figsize=(8, 5),
):
    """
    Bar plot of where mistakes first appear in the hierarchy.
    """
    breakdown = hierarchical_error_breakdown(predictions_epoch)

    def sort_key(x):
        if x == "correct_path":
            return 999
        return int(x)

    breakdown = breakdown.sort_values("category", key=lambda s: s.map(sort_key))

    labels = [
        "Correct path" if x == "correct_path" else f"First error at level {int(x)}"
        for x in breakdown["category"]
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, breakdown["fraction"])
    ax.set_title("Hierarchical error breakdown")
    ax.set_ylabel("Fraction of samples")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    for i, v in enumerate(breakdown["fraction"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.show()

    return breakdown


def path_consistency_report(
    predictions_epoch: pd.DataFrame,
    allowed_edges: Optional[Dict[Tuple[int, int], Any]] = None,
) -> Dict[str, Any]:
    """
    Compute hierarchy path consistency.

    If predictions come from a masked/strict hierarchical model, violations may be zero.

    Parameters
    ----------
    allowed_edges:
        Optional dict mapping (parent_level, child_level) -> allowed matrix or dict.

        Example using matrices:
            allowed_edges = {
                (0, 1): allowed_01,   # shape [n_level0, n_level1], values 0/1 or bool
                (1, 2): allowed_12,
            }

        Example using dict:
            allowed_edges = {
                (0, 1): {0: [0, 1], 1: [2, 3]},
                (1, 2): {0: [2], 1: [3]},
            }

    Returns
    -------
    report dict with violation_count and violation_rate.
    """
    if allowed_edges is None:
        print(
            "No allowed_edges passed. Cannot verify structural violations. "
            "Returning only path length diagnostics."
        )

        lengths = predictions_epoch["pred_path_parsed"].apply(len)
        return {
            "num_samples": len(predictions_epoch),
            "pred_path_lengths": lengths.value_counts().to_dict(),
        }

    def is_edge_allowed(parent, child, edge_obj):
        if isinstance(edge_obj, dict):
            return child in edge_obj.get(parent, [])
        arr = np.asarray(edge_obj)
        if parent < 0 or child < 0:
            return False
        if parent >= arr.shape[0] or child >= arr.shape[1]:
            return False
        return bool(arr[parent, child])

    violations = []
    for _, row in predictions_epoch.iterrows():
        p = row["pred_path_parsed"]
        violated = False
        bad_edge = None

        for (parent_level, child_level), edge_obj in allowed_edges.items():
            if len(p) <= max(parent_level, child_level):
                violated = True
                bad_edge = (parent_level, child_level, None, None)
                break

            parent = int(p[parent_level])
            child = int(p[child_level])

            if not is_edge_allowed(parent, child, edge_obj):
                violated = True
                bad_edge = (parent_level, child_level, parent, child)
                break

        violations.append({
            "violated": violated,
            "bad_edge": bad_edge,
        })

    vdf = pd.DataFrame(violations)

    return {
        "num_samples": len(vdf),
        "violation_count": int(vdf["violated"].sum()),
        "violation_rate": float(vdf["violated"].mean()),
        "violation_details": vdf,
    }


def plot_path_consistency(
    predictions_epoch: pd.DataFrame,
    allowed_edges: Optional[Dict[Tuple[int, int], Any]] = None,
    figsize=(6, 5),
):
    """
    Plot valid vs invalid predicted paths.

    Requires allowed_edges to detect invalid paths.
    """
    report = path_consistency_report(predictions_epoch, allowed_edges=allowed_edges)

    if "violation_rate" not in report:
        print(report)
        return report

    valid_rate = 1.0 - report["violation_rate"]
    invalid_rate = report["violation_rate"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["Valid predicted path", "Invalid predicted path"], [valid_rate, invalid_rate])
    ax.set_title("Path consistency")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    ax.text(0, valid_rate + 0.01, f"{valid_rate:.3f}", ha="center")
    ax.text(1, invalid_rate + 0.01, f"{invalid_rate:.3f}", ha="center")

    plt.tight_layout()
    plt.show()

    print(f"Violation count: {report['violation_count']} / {report['num_samples']}")
    print(f"Violation rate:  {report['violation_rate']:.4f}")

    return report


def plot_accuracy_per_depth(
    metrics: Optional[pd.DataFrame] = None,
    predictions_epoch: Optional[pd.DataFrame] = None,
    epoch: Optional[int] = None,
    figsize=(8, 5),
):
    """
    Plot accuracy per depth.

    Two modes:
    1. From metrics.csv:
        plot_accuracy_per_depth(metrics=metrics, epoch=best_epoch)

    2. From predictions.csv for selected epoch:
        plot_accuracy_per_depth(predictions_epoch=epoch_preds)
    """
    if predictions_epoch is not None:
        return plot_path_correctness_by_level(predictions_epoch, figsize=figsize)

    if metrics is None:
        raise ValueError("Pass either metrics or predictions_epoch.")

    if epoch is None:
        epoch = choose_best_epoch(metrics, metric="val_leaf_acc", mode="max")

    row = metrics[metrics["epoch"] == epoch]
    if row.empty:
        raise ValueError(f"Epoch {epoch} not found in metrics.")

    row = row.iloc[0]

    labels = []
    values = []

    for c in get_level_acc_columns(metrics):
        level = re.search(r"val_level_(\d+)_acc", c).group(1)
        labels.append(f"Level {level}")
        values.append(float(row[c]))

    if "val_leaf_acc" in metrics.columns:
        labels.append("Leaf")
        values.append(float(row["val_leaf_acc"]))

    if "val_path_acc" in metrics.columns:
        labels.append("Path")
        values.append(float(row["val_path_acc"]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values)
    ax.set_title(f"Accuracy per depth at epoch {epoch}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.show()

    return pd.DataFrame({"depth": labels, "accuracy": values})


# ---------------------------------------------------------------------
# One-call inspection helpers
# ---------------------------------------------------------------------

def inspect_epoch(
    experiment_dir: str | Path,
    epoch: Optional[int] = None,
    best_metric: str = "val_leaf_acc",
    index_dict: Optional[Dict[int, Dict[int, str]]] = None,
    normalize_cm: bool = False,
):
    """
    Convenience helper for notebook use.

    Loads an experiment, chooses epoch if needed, and plots:
        - leaf confusion matrix
        - all level confusion matrices
        - hierarchical error breakdown
        - accuracy per depth
        - top hierarchical confusions
    """
    metrics, preds = load_experiment(experiment_dir)

    if epoch is None:
        epoch = choose_best_epoch(metrics, metric=best_metric, mode="max")
        print(f"Selected best epoch by {best_metric}: {epoch}")
    else:
        print(f"Using manually selected epoch: {epoch}")

    epoch_preds = filter_predictions_by_epoch(preds, epoch)

    plot_leaf_confusion_matrix(epoch_preds, index_dict=index_dict, normalize=normalize_cm)
    plot_all_level_confusion_matrices(epoch_preds, index_dict=index_dict, normalize=normalize_cm)
    plot_hierarchical_error_breakdown(epoch_preds)
    plot_accuracy_per_depth(metrics=metrics, epoch=epoch)
    confusions = plot_top_hierarchical_confusions(epoch_preds, index_dict=index_dict, top_k=20)

    return {
        "metrics": metrics,
        "predictions": preds,
        "epoch_predictions": epoch_preds,
        "selected_epoch": epoch,
        "top_hierarchical_confusions": confusions,
    }


# =====================================================================
# V2 additions: robust new-log parsing, epoch-range views, multi-run comparison
# =====================================================================

from dataclasses import dataclass, field


def _parse_list_like(x):
    """Parse stringified list columns such as probabilities, candidate paths and path scores."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s in {"", "NOT_IMPLEMENTED", "None", "nan"}:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, np.ndarray):
                return parsed.tolist()
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        except Exception:
            return []
    return []


def _coerce_numeric_columns(df: pd.DataFrame, exclude: Optional[set[str]] = None) -> pd.DataFrame:
    """Convert metric-like columns to numeric while preserving explicitly excluded columns."""
    out = df.copy()
    exclude = exclude or set()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == object:
            out[c] = out[c].replace({"NOT_IMPLEMENTED": np.nan, "None": np.nan, "nan": np.nan})
            converted = pd.to_numeric(out[c], errors="ignore")
            out[c] = converted
    return out


def _parse_bool_series(s: pd.Series) -> pd.Series:
    """Robustly parse True/False columns that may be strings."""
    if s.dtype == bool:
        return s
    return s.map(lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true")


def filter_metrics_by_epoch_range(metrics: pd.DataFrame, epoch_range: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """
    Filter metrics to an inclusive epoch range, e.g. epoch_range=[2, 12].
    If epoch_range is None, returns a copy of all rows.
    """
    if epoch_range is None:
        return metrics.copy()
    if len(epoch_range) != 2:
        raise ValueError("epoch_range must be None or a two-value sequence like [2, 12].")
    lo, hi = int(epoch_range[0]), int(epoch_range[1])
    if lo > hi:
        lo, hi = hi, lo
    out = metrics[(metrics["epoch"] >= lo) & (metrics["epoch"] <= hi)].copy()
    if out.empty:
        raise ValueError(f"No metric rows found in epoch range [{lo}, {hi}].")
    return out


def filter_predictions_by_epoch_range(predictions: pd.DataFrame, epoch_range: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """
    Filter predictions to an inclusive epoch range, e.g. epoch_range=[2, 12].
    For confusion matrices, usually still select one epoch with filter_predictions_by_epoch().
    """
    if epoch_range is None:
        return predictions.copy()
    if len(epoch_range) != 2:
        raise ValueError("epoch_range must be None or a two-value sequence like [2, 12].")
    lo, hi = int(epoch_range[0]), int(epoch_range[1])
    if lo > hi:
        lo, hi = hi, lo
    out = predictions[(predictions["epoch"] >= lo) & (predictions["epoch"] <= hi)].copy()
    if out.empty:
        raise ValueError(f"No prediction rows found in epoch range [{lo}, {hi}].")
    return out


# Override load_experiment with a more robust version.
def load_experiment(
    experiment_dir: str | Path,
    epoch_range: Optional[Sequence[int]] = None,
    parse_prediction_lists: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load metrics.csv and predictions.csv from an experiment directory.

    New-log compatible:
      - Handles train_leaf_acc, val_target_path_prob, val_pred_path_prob.
      - Converts NOT_IMPLEMENTED in numeric columns to NaN.
      - Parses pred_path, target_path, candidate_paths, path_scores, and level_n_probs.
      - Optionally filters both files to an inclusive epoch range, e.g. [2, 12].
    """
    experiment_dir = Path(experiment_dir)
    metrics_path = experiment_dir / "metrics.csv"
    preds_path = experiment_dir / "predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics.csv at: {metrics_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Could not find predictions.csv at: {preds_path}")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(preds_path)

    metrics = _coerce_numeric_columns(metrics)

    list_like_cols = {"pred_path", "target_path", "path_scores", "candidate_paths"}
    list_like_cols |= {c for c in predictions.columns if re.fullmatch(r"level_\d+_probs", c)}
    predictions = _coerce_numeric_columns(predictions, exclude=list_like_cols)

    if "pred_path" in predictions.columns:
        predictions["pred_path_parsed"] = predictions["pred_path"].apply(_parse_path)
    if "target_path" in predictions.columns:
        predictions["target_path_parsed"] = predictions["target_path"].apply(_parse_path)

    if parse_prediction_lists:
        for c in ["path_scores", "candidate_paths"]:
            if c in predictions.columns:
                predictions[f"{c}_parsed"] = predictions[c].apply(_parse_list_like)
        for c in sorted([c for c in predictions.columns if re.fullmatch(r"level_\d+_probs", c)]):
            predictions[f"{c}_parsed"] = predictions[c].apply(_parse_list_like)

    for c in ["correct_path", "correct_leaf"]:
        if c in predictions.columns:
            predictions[c] = _parse_bool_series(predictions[c])

    metrics = filter_metrics_by_epoch_range(metrics, epoch_range)
    predictions = filter_predictions_by_epoch_range(predictions, epoch_range)

    return metrics, predictions


@dataclass
class ExperimentView:
    """Small notebook-friendly wrapper around one experiment and an optional epoch range."""
    exp_dir: str | Path
    name: Optional[str] = None
    epoch_range: Optional[Sequence[int]] = None
    metrics: pd.DataFrame = field(init=False)
    predictions: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.exp_dir = Path(self.exp_dir)
        if self.name is None:
            self.name = self.exp_dir.name
        self.metrics, self.predictions = load_experiment(self.exp_dir, epoch_range=self.epoch_range)

    def best_epoch(self, metric: str = "val_leaf_acc", mode: str = "max") -> int:
        return choose_best_epoch(self.metrics, metric=metric, mode=mode)

    def preds_at(self, epoch: Optional[int] = None, metric: str = "val_leaf_acc", mode: str = "max") -> pd.DataFrame:
        if epoch is None:
            epoch = self.best_epoch(metric=metric, mode=mode)
        return filter_predictions_by_epoch(self.predictions, epoch)


def load_many_experiments(
    exp_dirs: Sequence[str | Path],
    names: Optional[Sequence[str]] = None,
    epoch_range: Optional[Sequence[int]] = None,
) -> Dict[str, ExperimentView]:
    """Load several experiment folders for comparison plots."""
    if names is not None and len(names) != len(exp_dirs):
        raise ValueError("names must have same length as exp_dirs.")
    out = {}
    for i, exp_dir in enumerate(exp_dirs):
        name = names[i] if names is not None else Path(exp_dir).name
        out[name] = ExperimentView(exp_dir=exp_dir, name=name, epoch_range=epoch_range)
    return out


def _as_experiment_dict(experiments: Dict[str, Any] | Sequence[ExperimentView]) -> Dict[str, ExperimentView]:
    if isinstance(experiments, dict):
        return experiments
    return {e.name: e for e in experiments}


def plot_compare_metric(
    experiments: Dict[str, ExperimentView] | Sequence[ExperimentView],
    metric: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize=(9, 5),
):
    """Line plot of one metric across multiple runs."""
    experiments = _as_experiment_dict(experiments)
    fig, ax = plt.subplots(figsize=figsize)

    plotted = 0
    for name, exp in experiments.items():
        if metric not in exp.metrics.columns:
            print(f"Skipping {name}: metric '{metric}' not found.")
            continue
        ax.plot(exp.metrics["epoch"], exp.metrics[metric], marker="o", label=name)
        plotted += 1

    if plotted == 0:
        raise ValueError(f"Metric '{metric}' was not found in any experiment.")

    ax.set_title(title or metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or metric)
    if "acc" in metric or "prob" in metric:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_compare_core_curves(experiments: Dict[str, ExperimentView] | Sequence[ExperimentView]):
    """
    Recommended multi-run comparison plots:
      1. validation leaf accuracy
      2. validation path accuracy
      3. validation predicted path probability
      4. validation target path probability
      5. each hierarchy level accuracy
    """
    experiments = _as_experiment_dict(experiments)
    for metric, title in [
        ("val_leaf_acc", "Validation leaf accuracy"),
        ("val_path_acc", "Validation path accuracy"),
        ("val_pred_path_prob", "Validation predicted path probability"),
        ("val_target_path_prob", "Validation target path probability"),
    ]:
        if any(metric in e.metrics.columns for e in experiments.values()):
            plot_compare_metric(experiments, metric, title=title, ylabel=metric)

    all_level_cols = sorted(
        {c for e in experiments.values() for c in e.metrics.columns if re.fullmatch(r"val_level_\d+_acc", c)},
        key=lambda c: int(re.search(r"val_level_(\d+)_acc", c).group(1)),
    )
    for c in all_level_cols:
        level = re.search(r"val_level_(\d+)_acc", c).group(1)
        plot_compare_metric(experiments, c, title=f"Validation accuracy at hierarchy level {level}", ylabel="Accuracy")


def summarize_best_epochs(
    experiments: Dict[str, ExperimentView] | Sequence[ExperimentView],
    metric: str = "val_leaf_acc",
    mode: str = "max",
) -> pd.DataFrame:
    """Return a compact table of best epoch and key validation metrics per run."""
    experiments = _as_experiment_dict(experiments)
    rows = []
    wanted_cols = [
        "val_loss", "val_leaf_acc", "val_path_acc", "val_pred_path_prob", "val_target_path_prob",
        *sorted(
            {c for e in experiments.values() for c in e.metrics.columns if re.fullmatch(r"val_level_\d+_acc", c)},
            key=lambda c: int(re.search(r"val_level_(\d+)_acc", c).group(1)),
        ),
    ]
    for name, exp in experiments.items():
        epoch = exp.best_epoch(metric=metric, mode=mode)
        row = exp.metrics[exp.metrics["epoch"] == epoch].iloc[0]
        item = {"run": name, "selected_epoch": epoch, "selection_metric": metric}
        for c in wanted_cols:
            if c in exp.metrics.columns:
                item[c] = row[c]
        rows.append(item)
    return pd.DataFrame(rows).sort_values(metric if metric in wanted_cols else "selected_epoch", ascending=(mode == "min"))


def compare_hierarchical_error_breakdowns(
    experiments: Dict[str, ExperimentView] | Sequence[ExperimentView],
    epoch: Optional[int] = None,
    best_metric: str = "val_leaf_acc",
    mode: str = "max",
    figsize=(10, 5),
) -> pd.DataFrame:
    """
    Side-by-side bar plot of first-error-level breakdown for several runs.
    If epoch is None, each run uses its own best epoch by best_metric.
    """
    experiments = _as_experiment_dict(experiments)
    rows = []
    for name, exp in experiments.items():
        selected_epoch = epoch if epoch is not None else exp.best_epoch(best_metric, mode)
        ep = exp.preds_at(selected_epoch)
        b = hierarchical_error_breakdown(ep).copy()
        b["run"] = name
        b["epoch"] = selected_epoch
        rows.append(b)

    table = pd.concat(rows, ignore_index=True)
    table["category_label"] = table["category"].map(
        lambda x: "Correct path" if x == "correct_path" else f"First error L{int(x)}"
    )
    pivot = table.pivot(index="category_label", columns="run", values="fraction").fillna(0.0)

    order = sorted(
        pivot.index,
        key=lambda x: 999 if x == "Correct path" else int(re.search(r"L(\d+)", x).group(1)),
    )
    pivot = pivot.loc[order]

    ax = pivot.plot(kind="bar", figsize=figsize)
    ax.set_title("Hierarchical first-error breakdown across runs")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of samples")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    return table


def plot_compare_accuracy_per_depth_at_best_epoch(
    experiments: Dict[str, ExperimentView] | Sequence[ExperimentView],
    best_metric: str = "val_leaf_acc",
    mode: str = "max",
    figsize=(10, 5),
) -> pd.DataFrame:
    """Compare Level 0/1/2, leaf, and path accuracy at each run's selected best epoch."""
    experiments = _as_experiment_dict(experiments)
    rows = []
    for name, exp in experiments.items():
        epoch = exp.best_epoch(best_metric, mode)
        row = exp.metrics[exp.metrics["epoch"] == epoch].iloc[0]
        for c in get_level_acc_columns(exp.metrics):
            level = re.search(r"val_level_(\d+)_acc", c).group(1)
            rows.append({"run": name, "epoch": epoch, "depth": f"Level {level}", "accuracy": float(row[c])})
        if "val_leaf_acc" in exp.metrics.columns:
            rows.append({"run": name, "epoch": epoch, "depth": "Leaf", "accuracy": float(row["val_leaf_acc"])})
        if "val_path_acc" in exp.metrics.columns:
            rows.append({"run": name, "epoch": epoch, "depth": "Path", "accuracy": float(row["val_path_acc"])})

    table = pd.DataFrame(rows)
    order = sorted(
        table["depth"].unique(),
        key=lambda x: 997 if x == "Leaf" else 998 if x == "Path" else int(re.search(r"Level (\d+)", x).group(1)),
    )
    pivot = table.pivot(index="depth", columns="run", values="accuracy").loc[order]

    ax = pivot.plot(kind="bar", figsize=figsize)
    ax.set_title(f"Accuracy per hierarchy depth at best epoch ({best_metric})")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return table


def inspect_experiment_range(
    experiment_dir: str | Path,
    epoch_range: Optional[Sequence[int]] = None,
    best_metric: str = "val_leaf_acc",
    index_dict: Optional[Dict[int, Dict[int, str]]] = None,
    normalize_cm: bool = False,
):
    """
    Like inspect_epoch(), but initializes an ExperimentView with an epoch range.
    The best epoch is selected only inside that range.
    """
    exp = ExperimentView(experiment_dir, epoch_range=epoch_range)
    print(f"Loaded {exp.name}" + (f" for epochs {list(epoch_range)}" if epoch_range else ""))
    plot_training_dynamics(exp.metrics)
    epoch = exp.best_epoch(metric=best_metric, mode="max")
    print(f"Selected best epoch by {best_metric}: {epoch}")
    epoch_preds = exp.preds_at(epoch)

    plot_leaf_confusion_matrix(epoch_preds, index_dict=index_dict, normalize=normalize_cm)
    plot_all_level_confusion_matrices(epoch_preds, index_dict=index_dict, normalize=normalize_cm)
    plot_hierarchical_error_breakdown(epoch_preds)
    plot_accuracy_per_depth(metrics=exp.metrics, epoch=epoch)
    confusions = plot_top_hierarchical_confusions(epoch_preds, index_dict=index_dict, top_k=20)

    return {
        "experiment": exp,
        "metrics": exp.metrics,
        "predictions": exp.predictions,
        "epoch_predictions": epoch_preds,
        "selected_epoch": epoch,
        "top_hierarchical_confusions": confusions,
    }

def plot_leaf_confusion_conditioned_on_level(
    predictions_epoch,
    condition_level: int,
    condition_class,
    index_dict=None,
    normalize=False,
    print_metrics=True,
):
    """
    Plot leaf confusion matrix only for samples whose target path contains
    condition_class at condition_level.

    Example:
        plot_leaf_confusion_conditioned_on_level(
            epoch_preds,
            condition_level=1,
            condition_class=4,
            index_dict=hierarchy_index_dict
        )
    """

    subset = predictions_epoch[
        predictions_epoch["target_path_parsed"].apply(
            lambda p: len(p) > condition_level and p[condition_level] == condition_class
        )
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No samples found with target level {condition_level} = {condition_class}"
        )

    class_name = str(condition_class)
    if index_dict is not None and condition_level in index_dict:
        class_name = index_dict[condition_level].get(condition_class, class_name)

    return plot_leaf_confusion_matrix(
        subset,
        index_dict=index_dict,
        normalize=normalize,
        print_metrics=print_metrics,
    )

def plot_leaf_confusions_by_level_class(
    predictions_epoch,
    condition_level: int = 1,
    index_dict=None,
    normalize=False,
):
    classes = sorted({
        p[condition_level]
        for p in predictions_epoch["target_path_parsed"]
        if len(p) > condition_level
    })

    results = {}

    for cls in classes:
        name = str(cls)
        if index_dict is not None and condition_level in index_dict:
            name = index_dict[condition_level].get(cls, name)

        print(f"\nConditioned on level {condition_level}: {name}")

        results[cls] = plot_leaf_confusion_conditioned_on_level(
            predictions_epoch,
            condition_level=condition_level,
            condition_class=cls,
            index_dict=index_dict,
            normalize=normalize,
        )

    return results

def plot_leaf_confusion_given_correct_level(
    predictions_epoch,
    level: int,
    class_id: int,
    index_dict=None,
    normalize=True,
    print_metrics=True,
    name="",
):
    subset = predictions_epoch[
        predictions_epoch.apply(
            lambda r:
                len(r["target_path_parsed"]) > level
                and len(r["pred_path_parsed"]) > level
                and r["target_path_parsed"][level] == class_id
                and r["pred_path_parsed"][level] == class_id,
            axis=1
        )
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No samples found where target and prediction are both class {class_id} at level {level}."
        )

    class_name = str(class_id)
    if index_dict is not None and level in index_dict:
        class_name = index_dict[level].get(class_id, class_name)

    print(
        f"Conditioned on correctly predicted level {level}: {class_name} "
        f"({len(subset)} samples)"
    )

    return plot_leaf_confusion_matrix(
        subset,
        index_dict=index_dict,
        normalize=normalize,
        print_metrics=print_metrics,
        name=name
    )

