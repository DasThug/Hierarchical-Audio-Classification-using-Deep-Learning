import torch
import numpy as np
import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hierarchies.hierarchies import get_hierarchy_tree
from model_frameworks.dataloader_utilities import AudioDataset, AudioTransform
from training.fit import safe_get
from configs import test_config


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
config = test_config.CONFIG

CHECKPOINT_PATH = Path("outputs_local/ar_hierarchy_masked_debug/models/model_epoch_61.pt")

SPLIT_TO_DEBUG = "test"   # "train" or "test"
TOP_K = 20
SORT_BY = "target_loss"   # "target_loss", "pred_prob", "wrong_pred_prob"


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_df = pd.read_csv(config["dataset_path"])
print(f"File path for audio exists? = {Path(dataset_df.iloc[0]['audio_path']).exists()}")

hierarchy_tree = get_hierarchy_tree(config["hierarchy"])

if config["dataset"] == "urbansound8k":
    train_data = dataset_df[dataset_df["fold"] != 10].reset_index(drop=True)
    test_data = dataset_df[dataset_df["fold"] == 10].reset_index(drop=True)
else:
    train_data, test_data = train_test_split(
        dataset_df,
        test_size=1 - config["train_size"],
        stratify=dataset_df["class_id"],
        random_state=config["seed"],
    )

print("FULL Dataset:", dataset_df.shape)
print("TRAIN Dataset:", train_data.shape)
print("TEST Dataset:", test_data.shape)

transform = AudioTransform(
    sample_rate=config["sample_rate"],
    n_mels=config["n_mels"],
    n_fft=config["n_fft"],
    hop_length=config["hop_length"],
)

training_set = AudioDataset(df=train_data, transform=transform, split="train")
testing_set = AudioDataset(df=test_data, transform=transform, split="test")

train_loader = DataLoader(
    training_set,
    batch_size=config["batch_size"],
    shuffle=False,          # important for stable debugging
    num_workers=config["num_workers"],
)

test_loader = DataLoader(
    testing_set,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
)

debug_loader = train_loader if SPLIT_TO_DEBUG == "train" else test_loader

model = config["model_class"](
    hierarchy_tree=hierarchy_tree,
    **config["model_kwargs"],
).to(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()


# ---------------------------------------------------------------------
# Helper: score any path under its own autoregressive conditions
# ---------------------------------------------------------------------
@torch.no_grad()
def score_paths_conditioned_on_path(model, raw_logits_by_level, paths):
    """
    Scores each path using the same conditional masking logic as model.predict().

    Args:
        model: MaskedHierarchicalVGG16
        raw_logits_by_level: dict[level] -> tensor [B, num_classes_at_level]
        paths: tensor [B, depth]

    Returns:
        dict with:
            path_log_prob: [B]
            path_prob: [B]
            level_log_probs: list length depth, each [B]
            level_probs: list length depth, each [B]
            argmax_by_level: list length depth, each [B]
            argmax_probs_by_level: list length depth, each [B]
    """
    batch_size = paths.shape[0]
    device = paths.device

    selected_log_probs = []
    selected_probs = []
    argmax_by_level = []
    argmax_probs_by_level = []

    for level in range(model.depth):
        if level == 0:
            parent_ids = None
        else:
            parent_ids = paths[:, level - 1]

        mask = model._get_mask(level, parent_ids, batch_size).to(device)
        raw_logits = raw_logits_by_level[level]

        _, log_probs = model._masked_log_softmax(raw_logits, mask)
        probs = log_probs.exp()

        child_ids = paths[:, level]

        selected_logp = log_probs.gather(1, child_ids.unsqueeze(1)).squeeze(1)
        selected_prob = selected_logp.exp()

        level_argmax = probs.argmax(dim=1)
        level_argmax_prob = probs.max(dim=1).values

        selected_log_probs.append(selected_logp)
        selected_probs.append(selected_prob)
        argmax_by_level.append(level_argmax)
        argmax_probs_by_level.append(level_argmax_prob)

    level_log_probs_tensor = torch.stack(selected_log_probs, dim=1)
    path_log_prob = level_log_probs_tensor.sum(dim=1)

    return {
        "path_log_prob": path_log_prob,
        "path_prob": path_log_prob.exp(),
        "level_log_probs": selected_log_probs,
        "level_probs": selected_probs,
        "argmax_by_level": argmax_by_level,
        "argmax_probs_by_level": argmax_probs_by_level,
    }


# ---------------------------------------------------------------------
# Debug loop
# ---------------------------------------------------------------------
records = []

with torch.no_grad():
    for idx_batch, batch_data in enumerate(debug_loader):
        print(f"Batch {idx_batch + 1}/{len(debug_loader)}")

        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch_data.items()
        }

        inputs = batch["input"]
        hierarchy_target = batch["hierarchy_target"].long()

        # Teacher-forced outputs: target path under correct parent conditions
        loss_outputs = model.training_step(
            x=inputs,
            hierarchy_target=hierarchy_target,
        )

        # Actual AR inference outputs
        pred_outputs = model.predict(inputs)

        pred_paths = pred_outputs["path"].long()
        raw_logits_by_level = pred_outputs["raw_logits"]

        # Score predicted path under predicted-path conditions
        pred_diag = score_paths_conditioned_on_path(
            model=model,
            raw_logits_by_level=raw_logits_by_level,
            paths=pred_paths,
        )

        # Score target path under target-path / teacher-forced conditions
        target_teacher_diag = score_paths_conditioned_on_path(
            model=model,
            raw_logits_by_level=raw_logits_by_level,
            paths=hierarchy_target,
        )

        # Score target labels under predicted-path conditions
        #
        # This answers:
        # "What probability would the target class get at each level if the
        # traversal followed the predicted path up to the parent?"
        target_under_pred_paths = pred_paths.clone()
        target_under_pred_paths[:, -1] = hierarchy_target[:, -1]

        # More general version:
        # for each level, use predicted parents before that level,
        # but select target class at the current level.
        target_probs_given_pred_prefix = []
        target_log_probs_given_pred_prefix = []
        target_valid_given_pred_prefix = []

        batch_size = inputs.shape[0]

        for level in range(model.depth):
            if level == 0:
                parent_ids = None
            else:
                parent_ids = pred_paths[:, level - 1]

            mask = model._get_mask(level, parent_ids, batch_size).to(device)
            raw_logits = raw_logits_by_level[level]

            _, log_probs = model._masked_log_softmax(raw_logits, mask)
            probs = log_probs.exp()

            target_l = hierarchy_target[:, level]

            valid = mask.gather(1, target_l.unsqueeze(1)).squeeze(1)

            selected_logp = log_probs.gather(1, target_l.unsqueeze(1)).squeeze(1)
            selected_prob = selected_logp.exp()

            # If invalid under predicted parent, probability is effectively 0
            selected_prob = torch.where(valid, selected_prob, torch.zeros_like(selected_prob))
            selected_logp = torch.where(
                valid,
                selected_logp,
                torch.full_like(selected_logp, float("-inf")),
            )

            target_probs_given_pred_prefix.append(selected_prob)
            target_log_probs_given_pred_prefix.append(selected_logp)
            target_valid_given_pred_prefix.append(valid)

        # Main scalar quantities
        teacher_forced_target_log_prob = loss_outputs["path_log_prob"]
        teacher_forced_target_prob = loss_outputs["path_prob"]
        teacher_forced_target_loss = -teacher_forced_target_log_prob

        pred_path_log_prob = pred_outputs["path_log_prob"]
        pred_path_prob = pred_outputs["path_prob"]
        pred_path_loss = -pred_path_log_prob

        # Sanity check: recomputed predicted-path prob should match predict()
        max_diff = (pred_diag["path_log_prob"] - pred_path_log_prob).abs().max().item()
        if max_diff > 1e-4:
            print(f"WARNING: recomputed pred path log-prob differs by {max_diff:.6f}")

        for i in range(batch_size):
            target_path = hierarchy_target[i].detach().cpu().tolist()
            pred_path = pred_paths[i].detach().cpu().tolist()

            records.append({
                "split": SPLIT_TO_DEBUG,
                "batch": idx_batch,
                "idx_in_batch": i,
                "global_idx": batch["index"][i].item(),

                "target_path": target_path,
                "pred_path": pred_path,
                "correct_path": pred_path == target_path,
                "correct_leaf": pred_path[-1] == target_path[-1],

                "target_path_loss_teacher_forced": teacher_forced_target_loss[i].item(),
                "target_path_prob_teacher_forced": teacher_forced_target_prob[i].item(),

                "pred_path_loss": pred_path_loss[i].item(),
                "pred_path_prob": pred_path_prob[i].item(),
                "pred_path_log_prob": pred_path_log_prob[i].item(),

                # Target path under correct teacher-forced traversal
                "teacher_forced_target_probs_by_level": [
                    target_teacher_diag["level_probs"][level][i].item()
                    for level in range(model.depth)
                ],
                "teacher_forced_argmax_by_level": [
                    target_teacher_diag["argmax_by_level"][level][i].item()
                    for level in range(model.depth)
                ],
                "teacher_forced_argmax_probs_by_level": [
                    target_teacher_diag["argmax_probs_by_level"][level][i].item()
                    for level in range(model.depth)
                ],

                # Predicted path under predicted traversal
                "pred_path_probs_by_level": [
                    pred_diag["level_probs"][level][i].item()
                    for level in range(model.depth)
                ],
                "pred_path_argmax_by_level": [
                    pred_diag["argmax_by_level"][level][i].item()
                    for level in range(model.depth)
                ],
                "pred_path_argmax_probs_by_level": [
                    pred_diag["argmax_probs_by_level"][level][i].item()
                    for level in range(model.depth)
                ],

                # Target class probability when conditioned on predicted parent path
                "target_probs_given_pred_prefix_by_level": [
                    target_probs_given_pred_prefix[level][i].item()
                    for level in range(model.depth)
                ],
                "target_valid_given_pred_prefix_by_level": [
                    bool(target_valid_given_pred_prefix[level][i].item())
                    for level in range(model.depth)
                ],
            })


# ---------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------
if SORT_BY == "target_loss":
    records_to_print = sorted(
        records,
        key=lambda r: r["target_path_loss_teacher_forced"],
        reverse=True,
    )
elif SORT_BY == "pred_prob":
    records_to_print = sorted(
        records,
        key=lambda r: r["pred_path_prob"],
        reverse=True,
    )
elif SORT_BY == "wrong_pred_prob":
    records_to_print = sorted(
        [r for r in records if not r["correct_path"]],
        key=lambda r: r["pred_path_prob"],
        reverse=True,
    )
else:
    raise ValueError(f"Unknown SORT_BY: {SORT_BY}")


# ---------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------
for r in records_to_print[:TOP_K]:
    print("\n" + "=" * 80)
    print("Split:", r["split"])
    print("Global idx:", r["global_idx"])

    print("Target path:", r["target_path"])
    print("Pred path:  ", r["pred_path"])

    print("Correct path:", r["correct_path"])
    print("Correct leaf:", r["correct_leaf"])

    print("\nPath-level probabilities:")
    print("Teacher-forced target loss:", r["target_path_loss_teacher_forced"])
    print("Teacher-forced target prob:", r["target_path_prob_teacher_forced"])
    print("Pred path loss:", r["pred_path_loss"])
    print("Pred path prob:", r["pred_path_prob"])

    print("\nPer-level comparison:")
    for level in range(model.depth):
        print(
            f"Level {level}: "
            f"target={r['target_path'][level]}, "
            f"pred={r['pred_path'][level]}, "
            f"TF_target_prob={r['teacher_forced_target_probs_by_level'][level]:.8f}, "
            f"TF_argmax={r['teacher_forced_argmax_by_level'][level]}, "
            f"TF_argmax_prob={r['teacher_forced_argmax_probs_by_level'][level]:.8f}, "
            f"pred_path_prob={r['pred_path_probs_by_level'][level]:.8f}, "
            f"pred_argmax={r['pred_path_argmax_by_level'][level]}, "
            f"pred_argmax_prob={r['pred_path_argmax_probs_by_level'][level]:.8f}, "
            f"target_prob_given_pred_prefix={r['target_probs_given_pred_prefix_by_level'][level]:.8f}, "
            f"target_valid_given_pred_prefix={r['target_valid_given_pred_prefix_by_level'][level]}"
        )