import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
from writers import append_epoch_log, append_prediction_log, append_debug_prediction_log, save_runtime_summary

# TRAIN FUNCTION
def train_one_epoch(
    model,                # CNN Class, Other Model Architectures
    dataloader,           # Training loader, (Dataloader object)
    optimizer,            # Optimizatch Adam
    # loss_fn,              # Criterion loss function, (Wraps a criterion to pass input and targets from batch) (ALL LOSS_FN SHOULD BE METHODS IN THE MODEL CLASS)
    device,               # torch.device
    metrics_fn=None,      # 
    augmentation_fn=None  # Augmentation function object, (etc. Mixup, ..)
):
    epoch_start_time = time.time()

    model.train()

    total_loss = 0.0
    total_path_log_prob = 0.0
    total_path_prob = 0.0
    num_batches = 0
    all_metrics = []

    for idx_batch, batch_data in enumerate(dataloader, 0):
        # Move tensors to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_data.items()}

        # Optional augmentation
        if augmentation_fn is not None:
            batch = augmentation_fn(batch)

        inputs = batch["input"]
        hierarchy_target = batch["hierarchy_target"]

        # Forward + model-specific loss
        outputs = model.training_step(x=inputs, hierarchy_target=hierarchy_target)

        loss = outputs["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging values
        total_loss += loss.item()
        total_path_log_prob += outputs["path_log_prob"].mean().item()
        total_path_prob += outputs["path_prob"].mean().item()
        num_batches += 1

        if idx_batch % 20 == 0:
            elapsed = time.time() - epoch_start_time

            print(
                f"[TRAIN]"
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"[+{elapsed/60:.1f} min] "
                f"Batch {idx_batch}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Path log-prob: {outputs['path_log_prob'].mean().item():.4f} | "
                f"Path prob: {outputs['path_prob'].mean().item():.4f}",
                flush=True
            )

        # Optional metrics
        if metrics_fn is not None:
            metric = metrics_fn(outputs, batch)
            all_metrics.append(metric)

    avg_loss = total_loss / num_batches
    avg_path_log_prob = total_path_log_prob / num_batches
    avg_path_prob = total_path_prob / num_batches
    avg_metric = np.mean(all_metrics) if all_metrics else None

    return {
        "loss": avg_loss,
        "path_log_prob": avg_path_log_prob,
        "path_prob": avg_path_prob,
        "metric": avg_metric,
    }


# VALIDATION/TEST FUNCTION
def validate(
    model,              # CNN Class, Other Model Architectures
    dataloader,         # Test/Validation loader, (Dataloader object)
    # loss_fn,              # Criterion loss function, (Wraps a criterion to pass input and targets from batch) (ALL LOSS_FN SHOULD BE METHODS IN THE MODEL CLASS)
    device,             # torch.device
    metrics_fn=None,     #
    debug_validation=False, #
):
    val_start_time = time.time()

    model.eval()

    total_loss = 0.0
    total_path_log_prob = 0.0
    total_correct_paths = 0
    total_correct_leaf = 0
    total_target_path_prob = 0.0
    total_pred_path_prob = 0.0
    total_samples = 0
    num_batches = 0

    level_correct = defaultdict(int)
    level_total = defaultdict(int)
    all_probs_by_level = defaultdict(list)

    all_metrics = []
    all_pred_paths = []
    all_target_paths = []
    all_pred_path_probs = []
    all_target_path_probs = []
    all_path_scores = []

    # Debug logging
    all_masks_by_level = defaultdict(list)
    all_masked_logits_by_level = defaultdict(list)
    all_raw_logits_by_level = defaultdict(list)


    with torch.no_grad():
        for idx_batch, batch_data in enumerate(dataloader, 0):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_data.items()}

            inputs = batch["input"]
            hierarchy_target = batch["hierarchy_target"]

            # Validation loss using teacher-forced true path
            loss_outputs = model.training_step(x=inputs, hierarchy_target=hierarchy_target)

            loss = loss_outputs["loss"]

            # Prediction using highest-scoring valid path
            pred_outputs = model.predict(inputs)
            pred_paths = pred_outputs["path"]

            # (Save target and prediction path probabilities)
            target_path_prob = loss_outputs["path_prob"]
            pred_path_prob = pred_outputs["path_prob"]

            total_target_path_prob += target_path_prob.sum().item()
            total_pred_path_prob += pred_path_prob.sum().item()

            all_target_path_probs.append(target_path_prob.detach().cpu())
            all_pred_path_probs.append(pred_path_prob.detach().cpu())
            all_path_scores.append(pred_outputs["path_scores"].detach().cpu())

            for level, probs in loss_outputs["probs"].items():
                all_probs_by_level[level].append(probs.detach().cpu())

            
            if debug_validation:
                for level, mask in loss_outputs["masks"].items():
                    all_masks_by_level[level].append(mask.detach().cpu())

                for level, masked_logits in loss_outputs["masked_logits"].items():
                    all_masked_logits_by_level[level].append(masked_logits.detach().cpu())

                for level, raw_logits in loss_outputs["raw_logits"].items():
                    all_raw_logits_by_level[level].append(raw_logits.detach().cpu())


            # Full path accuracy
            correct_paths = (pred_paths == hierarchy_target).all(dim=1)

            # Leaf accuracy
            pred_leaf = pred_paths[:, -1]
            target_leaf = hierarchy_target[:, -1]
            correct_leaf = pred_leaf == target_leaf

            # Per-level accuracy
            for level in range(hierarchy_target.shape[1]):
                level_correct[level] += (pred_paths[:, level] == hierarchy_target[:, level]).sum().item()
                level_total[level] += inputs.shape[0]

            total_loss += loss.item()
            total_path_log_prob += loss_outputs["path_log_prob"].mean().item()
            total_correct_paths += correct_paths.sum().item()
            total_correct_leaf += correct_leaf.sum().item()
            total_samples += inputs.shape[0]
            num_batches += 1

            all_pred_paths.append(pred_paths.detach().cpu())
            all_target_paths.append(hierarchy_target.detach().cpu())

            if idx_batch % 20 == 0:
                elapsed = time.time() - val_start_time

                print(
                    f"[VAL] "
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[+{elapsed/60:.1f} min] "
                    f"Batch {idx_batch}/{len(dataloader)}",
                    f"Loss: {loss.item():.4f} | "
                    f"Path acc: {correct_paths.float().mean().item():.4f}",
                    flush=True,
                )

            if metrics_fn is not None:
                metric = metrics_fn(
                    loss_outputs=loss_outputs,
                    pred_outputs=pred_outputs,
                    batch=batch,
                )
                all_metrics.append(metric)

    pred_paths_all = torch.cat(all_pred_paths, dim=0)
    target_paths_all = torch.cat(all_target_paths, dim=0)

    per_level_acc = {
        level: level_correct[level] / level_total[level]
        for level in level_correct
    }

    target_path_probs_all = torch.cat(all_target_path_probs, dim=0)
    pred_path_probs_all = torch.cat(all_pred_path_probs, dim=0)
    path_scores_all = torch.cat(all_path_scores, dim=0)

    probs_by_level_all = {
        level: torch.cat(probs_list, dim=0)
        for level, probs_list in all_probs_by_level.items()
    }


    masks_by_level_all = {}
    masked_logits_by_level_all = {}
    raw_logits_by_level_all = {}
    if debug_validation:

        probs_by_level_all = {
            level: torch.cat(values, dim=0)
            for level, values in all_probs_by_level.items()
        }

        masks_by_level_all = {
            level: torch.cat(values, dim=0)
            for level, values in all_masks_by_level.items()
        }

        masked_logits_by_level_all = {
            level: torch.cat(values, dim=0)
            for level, values in all_masked_logits_by_level.items()
        }

        raw_logits_by_level_all = {
            level: torch.cat(values, dim=0)
            for level, values in all_raw_logits_by_level.items()
        }

    return {
        "loss": total_loss / num_batches,
        "path_log_prob": total_path_log_prob / num_batches,
        "path_acc": total_correct_paths / total_samples,
        "leaf_acc": total_correct_leaf / total_samples,
        "per_level_acc": per_level_acc,

        "pred_paths": pred_paths_all,
        "target_paths": target_paths_all,
        "pred_leaf": pred_paths_all[:, -1],
        "target_leaf": target_paths_all[:, -1],

        "all_metrics": all_metrics,

        "target_path_prob": total_target_path_prob / total_samples,
        "pred_path_prob": total_pred_path_prob / total_samples,

        "target_path_probs": target_path_probs_all,
        "pred_path_probs": pred_path_probs_all,
        "path_scores": path_scores_all,
        "candidate_paths": pred_outputs["candidate_paths"],
        "probs_by_level": probs_by_level_all,

        "masks_by_level": masks_by_level_all,
        "masked_logits_by_level": masked_logits_by_level_all,
        "raw_logits_by_level": raw_logits_by_level_all,
    }


# EPOCH LOOP (FIT FUNCTION)
def fit(
    model,                  # CNN Class, Other Model Architectures
    train_loader,           # Training loader, (Dataloader object)
    val_loader,             # Test/Validation loader, (Dataloader object)
    optimizer,              # Optimizatch Adam
    # loss_fn,              # Criterion loss function, (Wraps a criterion to pass input and targets from batch) (ALL LOSS_FN SHOULD BE METHODS IN THE MODEL CLASS)
    device,                 # torch.device
    epochs,                 # number of Epochs
    metrics_fn=None,        # 
    augmentation_fn=None,   # Augmentation function object, (etc. Mixup, ..)
    scheduler=None,         # Update Learnin Rate at the end of an epoch, (etc. StepLR, ..)
    log_csv_path="outputs/epoch_logs.csv",
    prediction_csv_path="outputs/prediction_logs.csv",
    debug_validation=False,
    debug_csv_path="outputs/debug_predictions.csv",
    runtime_json_path=f"outputs/runtime_summary.json",
    experiment_name=None,
):
    
    experiment_start_time = time.time()
    experiment_start_timestamp = datetime.now()
    print(f"TRAINING STARTED: {experiment_start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    history = {
        # Save metrics and other measurements
        "train_loss": [],
        "val_loss": [],

        "train_path_log_prob": [],
        "train_path_prob": [],
        "val_path_log_prob": [],
        "val_target_path_prob": [],
        "val_pred_path_prob": [],

        "val_path_acc": [],
        "val_leaf_acc": [],
        "val_per_level_acc": [],

        "train_metric": [],
        "val_all_metrics": [],

        "val_pred_paths": [],
        "val_target_paths": [],

    }

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_start_timestamp = datetime.now()

        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            metrics_fn=metrics_fn,
            augmentation_fn=augmentation_fn,
        )

        val_stats = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            metrics_fn=metrics_fn,
            debug_validation=debug_validation,
        )
        # Both loops return "loss" and "metric"

        if scheduler is not None:
            # update the learning rate of the optimizer based on a predefined schedule.
            scheduler.step()

        # Logging
        epoch_end_timestamp = datetime.now()
        epoch_runtime = time.time() - epoch_start_time

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Started: {epoch_start_timestamp.strftime('%H:%M:%S')} | ")
        print(f"Finished: {epoch_end_timestamp.strftime('%H:%M:%S')} | ")
        print(f"Runtime: {epoch_runtime/60:.2f} min\n")
        print(f"[EPOCH {epoch + 1}] Train Loss: {train_stats['loss']:.4f}")
        print(f"[EPOCH {epoch + 1}] Val Loss:   {val_stats['loss']:.4f}")
        print(f"[EPOCH {epoch + 1}] Train Path LogProb: {train_stats['path_log_prob']:.4f}")
        print(f"[EPOCH {epoch + 1}] Val Path LogProb:   {val_stats['path_log_prob']:.4f}")
        print(f"[EPOCH {epoch + 1}] Val Path Acc:       {val_stats['path_acc']:.4f}")

        if metrics_fn is not None:
            print(f"[EPOCH {epoch + 1}] Train Metric: {train_stats['metric']}")
            print(f"[EPOCH {epoch + 1}] Val Metric:   {val_stats['metric']}")

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_path_log_prob"].append(train_stats["path_log_prob"])
        history["train_path_prob"].append(train_stats["path_prob"])
        history["val_path_log_prob"].append(val_stats["path_log_prob"])
        history["val_path_acc"].append(val_stats["path_acc"])
        history["train_metric"].append(train_stats["metric"])
        history["val_all_metrics"].append(val_stats["all_metrics"])
        history["val_leaf_acc"].append(val_stats["leaf_acc"])
        history["val_per_level_acc"].append(val_stats["per_level_acc"])
        history["val_target_path_prob"].append(val_stats["target_path_prob"])
        history["val_pred_path_prob"].append(val_stats["pred_path_prob"])

        # Remove these from history unless you really need them in memory:
        # history["val_pred_paths"].append(val_stats["pred_paths"].tolist())
        # history["val_target_paths"].append(val_stats["target_paths"].tolist())

        # Scalar epoch metrics CSV
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_path_log_prob": train_stats["path_log_prob"],
            "train_path_prob": train_stats["path_prob"],
            "val_path_log_prob": val_stats["path_log_prob"],
            "val_path_acc": val_stats["path_acc"],
            "val_leaf_acc": val_stats["leaf_acc"],
            "val_target_path_prob": val_stats["target_path_prob"],
            "val_pred_path_prob": val_stats["pred_path_prob"],
        }
        
        for level, acc in val_stats["per_level_acc"].items():
            epoch_log[f"val_level_{level}_acc"] = acc
        
        append_epoch_log(epoch_log, csv_path=log_csv_path)

        # Per-Sample validation predictions CSV
        append_prediction_log(
            epoch=epoch + 1,
            pred_paths=val_stats["pred_paths"],
            target_paths=val_stats["target_paths"],
            pred_path_probs=val_stats["pred_path_probs"],
            target_path_probs=val_stats["target_path_probs"],
            path_scores=val_stats["path_scores"],
            candidate_paths=val_stats["candidate_paths"],
            probs_by_level=val_stats["probs_by_level"],
            csv_path=prediction_csv_path,
        )

        if debug_validation and debug_csv_path is not None:
            append_debug_prediction_log(
                epoch=epoch + 1,
                pred_paths=val_stats["pred_paths"],
                target_paths=val_stats["target_paths"],
                pred_path_probs=val_stats["pred_path_probs"],
                target_path_probs=val_stats["target_path_probs"],
                path_scores=val_stats["path_scores"],
                candidate_paths=val_stats["candidate_paths"],
                probs_by_level=val_stats["probs_by_level"],
                masks_by_level=val_stats["masks_by_level"],
                masked_logits_by_level=val_stats["masked_logits_by_level"],
                raw_logits_by_level=val_stats["raw_logits_by_level"],
                csv_path=debug_csv_path,
            )
        
    experiment_end_time = time.time()
    experiment_end_timestamp = datetime.now()

    total_runtime_seconds = experiment_end_time - experiment_start_time
    total_runtime_hours = total_runtime_seconds / 3600

    print("TRAINING FINISHED")
    print(f"Started : {experiment_start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {experiment_end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runtime : {total_runtime_hours:.2f} hours")
    
    save_runtime_summary(
        runtime_path=runtime_json_path,
        experiment_name=experiment_name,
        start_ts=experiment_start_timestamp,
        end_ts=experiment_end_timestamp,
        runtime_seconds=total_runtime_seconds,
    )


    return history