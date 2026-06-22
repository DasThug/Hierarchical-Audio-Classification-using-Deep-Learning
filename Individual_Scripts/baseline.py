import pandas as pd
import ast
from pathlib import Path


NAIVE_PATH = [2, 4, 4]  # or [2, 4, 7]
EPOCH_TO_USE = 1


def parse_list(x):
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)


def make_naive_predictions(input_csv, output_csv, naive_path=NAIVE_PATH, epoch=EPOCH_TO_USE):
    df = pd.read_csv(input_csv)

    # Only keep one epoch because samples repeat across epochs
    df = df[df["epoch"] == epoch].copy()

    df["target_path"] = df["target_path"].apply(parse_list)

    df["pred_path"] = [naive_path for _ in range(len(df))]
    df["pred_leaf"] = naive_path[-1]

    df["correct_path"] = df["target_path"].apply(lambda t: t == naive_path)
    df["correct_leaf"] = df["target_leaf"] == naive_path[-1]

    # Keep only useful columns
    keep_cols = [
        "epoch",
        "sample_idx",
        "pred_path",
        "target_path",
        "pred_leaf",
        "target_leaf",
        "correct_path",
        "correct_leaf",
    ]

    df = df[keep_cols]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved naive predictions to: {output_csv}")

    return df


def make_naive_metrics(naive_predictions_csv, output_metrics_csv):
    df = pd.read_csv(naive_predictions_csv)

    df["pred_path"] = df["pred_path"].apply(parse_list)
    df["target_path"] = df["target_path"].apply(parse_list)

    path_acc = df["correct_path"].mean()
    leaf_acc = df["correct_leaf"].mean()

    level_accs = []
    for level in range(3):
        acc = df.apply(
            lambda row: row["pred_path"][level] == row["target_path"][level],
            axis=1
        ).mean()
        level_accs.append(acc)

    metrics = pd.DataFrame([{
        "epoch": 1,
        "train_loss": "NOT_IMPLEMENTED",
        "val_loss": "NOT_IMPLEMENTED",
        "train_path_log_prob": "NOT_IMPLEMENTED",
        "train_path_prob": "NOT_IMPLEMENTED",
        "train_leaf_acc": "NOT_IMPLEMENTED",
        "val_path_log_prob": "NOT_IMPLEMENTED",
        "val_path_acc": path_acc,
        "val_leaf_acc": leaf_acc,
        "val_target_path_prob": "NOT_IMPLEMENTED",
        "val_pred_path_prob": "NOT_IMPLEMENTED",
        "val_level_0_acc": level_accs[0],
        "val_level_1_acc": level_accs[1],
        "val_level_2_acc": level_accs[2],
    }])

    output_metrics_csv = Path(output_metrics_csv)
    output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(output_metrics_csv, index=False)
    print(f"Saved naive metrics to: {output_metrics_csv}")

    return metrics


if __name__ == "__main__":
    for i in [1,2,3,4,5,6,7,8,9]:
        input_csv = f"outputs_local/outputs2/flat_cv_valfold{i}/predictions.csv"

        naive_predictions_output = f"outputs/naive_val{i}/predictions.csv"
        naive_metrics_output = f"outputs/naive_val{i}/metrics.csv"

        make_naive_predictions(
            input_csv=input_csv,
            output_csv=naive_predictions_output,
            naive_path=[2, 4, 4],
            epoch=1
        )

        make_naive_metrics(
            naive_predictions_csv=naive_predictions_output,
            output_metrics_csv=naive_metrics_output
        )
    
    input_csv = f"outputs_local/outputs2/flat_final_test_fold10/predictions.csv"

    naive_predictions_output = f"outputs/naive_test10/predictions.csv"
    naive_metrics_output = "outputs/naive_test10/metrics.csv"
    make_naive_predictions(
        input_csv=input_csv,
        output_csv=naive_predictions_output,
        naive_path=[2, 4, 4],
        epoch=1
    )

    make_naive_metrics(
        naive_predictions_csv=naive_predictions_output,
        output_metrics_csv=naive_metrics_output
    )
