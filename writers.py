import csv
import pandas as pd
from pathlib import Path

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