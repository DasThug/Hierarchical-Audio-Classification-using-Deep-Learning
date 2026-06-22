import pandas as pd
import numpy as np


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def find_best_epoch(df, column, mode="min"):
    """
    Finds the epoch where `column` is minimized or maximized.
    Ignores non-numeric values such as NOT_IMPLEMENTED.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    values = pd.to_numeric(df[column], errors="coerce")

    if values.isna().all():
        return np.nan

    if mode == "min":
        best_idx = values.idxmin()
    elif mode == "max":
        best_idx = values.idxmax()
    else:
        raise ValueError("mode must be either 'min' or 'max'")

    return int(df.loc[best_idx, "epoch"])


def get_row_for_epoch(df, epoch):
    """
    Returns the row corresponding to a given epoch.
    """
    row = df[df["epoch"] == epoch]

    if row.empty:
        raise ValueError(f"Epoch {epoch} not found in CSV.")

    return row.iloc[0]


def summarize_metrics_at_selected_epoch(
    csv_paths,
    metrics,
    selection_column="val_loss",
    selection_mode="min",
):
    results = {}
    selected_epochs = []

    for metric in metrics:
        results[metric] = []

    for path in csv_paths:
        df = pd.read_csv(path)

        best_epoch = find_best_epoch(
            df,
            column=selection_column,
            mode=selection_mode
        )

        selected_epochs.append(best_epoch)

        if np.isnan(best_epoch):
            for metric in metrics:
                results[metric].append(np.nan)
            continue

        row = get_row_for_epoch(df, best_epoch)

        for metric in metrics:
            if metric not in df.columns:
                results[metric].append(np.nan)
            else:
                results[metric].append(safe_float(row[metric]))

    summary = {}

    for metric, values in results.items():
        values = np.array(values, dtype=float)

        # If ANY value is NaN -> whole metric becomes NaN
        if np.isnan(values).any():
            mean = np.nan
            std = np.nan
        else:
            mean = values.mean()
            std = values.std(ddof=1)

        summary[metric] = {
            "mean": mean,
            "std": std,
            "values": values,
        }

    return summary, selected_epochs


if __name__ == "__main__":
    csv_paths = [
        # "outputs_local/outputs2/independent_hierarchy_cv_valfold1/metrics.csv",
        "outputs/naive_val1/metrics.csv",
        "outputs/naive_val2/metrics.csv",
        "outputs/naive_val3/metrics.csv",
        "outputs/naive_val4/metrics.csv",
        "outputs/naive_val5/metrics.csv",
        "outputs/naive_val6/metrics.csv",
        "outputs/naive_val7/metrics.csv",
        "outputs/naive_val8/metrics.csv",
        "outputs/naive_val9/metrics.csv",
    ]

    metrics = [
        "val_leaf_acc",
        "val_path_acc",
        "val_level_0_acc",
        "val_level_1_acc",
        "val_level_2_acc",
    ]

    summary, selected_epochs = summarize_metrics_at_selected_epoch(
        csv_paths=csv_paths,
        metrics=metrics,
        selection_column="epoch",
        selection_mode="min",
    )

    print("Selected epochs:")
    for path, epoch in zip(csv_paths, selected_epochs):
        print(f"  {path}: epoch {epoch}")

    # print(
    #     f"\nMean selected epoch: "
    #     f"{np.mean(selected_epochs):.2f} ± "
    #     f"{np.std(selected_epochs, ddof=1):.2f}"
    # )

    print("\nSummary:")
    for metric, stats in summary.items():
        print(f"\n{metric}")

        for path, epoch, value in zip(
            csv_paths,
            selected_epochs,
            stats["values"]
        ):
            print(
                f"  {path}"
                f" | epoch={epoch}"
                f" | value={value}"
            )

        mean = stats["mean"]
        std = stats["std"]

        if np.isnan(mean):
            print("  Mean ± Std: NaN")
        else:
            print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
    
    print("\nFinal Table Summary:")
    for metric, stats in summary.items():
        mean = stats["mean"]
        std = stats["std"]

        if np.isnan(mean):
            print(f"{metric}: NaN")
        else:
            print(f"{metric}: {mean:.4f} ± {std:.4f}")