import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
from writers import append_epoch_log, append_prediction_log, append_debug_prediction_log, save_runtime_summary

def safe_get(outputs, key, default=None):
    return outputs[key] if key in outputs else default

def safe_cat(tensor_list):
    return torch.cat(tensor_list, dim=0) if len(tensor_list) > 0 else None

def safe_cat_dict(d):
    return {
        level: torch.cat(values, dim=0)
        for level, values in d.items()
        if len(values) > 0
    }


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
    num_batches = 0

    total_correct_leaf = 0
    num_leaf_acc_samples = 0

    total_path_log_prob = 0.0
    num_path_log_prob_batches = 0

    total_path_prob = 0.0
    num_path_prob_batches = 0

    all_metrics = []

    for idx_batch, batch_data in enumerate(dataloader, 0):
        # Move tensors to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_data.items()}

        # Optional augmentation
        if augmentation_fn is not None:
            batch = augmentation_fn(batch)

        inputs = batch["input"]
        hierarchy_target = batch["hierarchy_target"] # REQUIREMENT: All models should recieve the full hierarchy target (flat classifiers only consider last index)

        # Forward + model-specific loss
        outputs = model.training_step(x=inputs, hierarchy_target=hierarchy_target)

        loss = safe_get(outputs, "loss") # REQUIREMENT: Loss is expected in all models.
        if loss is None:
            raise ValueError("model.training_step() must return a 'loss' key.")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging values
        total_loss += loss.item()

        prediction = safe_get(outputs, "train_prediction") # METRIC: Train prediction for flat leaf classifier. If not present: None
        if prediction is not None:
            target_leaf = hierarchy_target[:, -1]
            correct_leaf = prediction == target_leaf
            total_correct_leaf += correct_leaf.sum().item()
            num_leaf_acc_samples += correct_leaf.numel()

        path_log_prob = safe_get(outputs, "path_log_prob") # METRIC: Path log-prob is expected in leveled models. If not present: None
        if path_log_prob is not None: 
            total_path_log_prob += path_log_prob.mean().item()
            num_path_log_prob_batches += 1
        
        path_prob = safe_get(outputs, "path_prob") # METRIC: Path prob is expected in leveled models. If not present: None
        if path_prob is not None:
            total_path_prob += path_prob.mean().item()
            num_path_prob_batches += 1


        num_batches += 1

        if idx_batch % 20 == 0:
            elapsed = time.time() - epoch_start_time
            msg = (
                f"[TRAIN] "
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"[+{elapsed/60:.1f} min] "
                f"Batch {idx_batch}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )
            if path_log_prob is not None:
                msg += f" | Path log-prob: {path_log_prob.mean().item():.4f}"
            if path_prob is not None:
                msg += f" | Path prob: {path_prob.mean().item():.4f}"
            print(msg, flush=True)

        # Optional metrics
        if metrics_fn is not None:
            metric = metrics_fn(outputs, batch)
            all_metrics.append(metric)

    # Finalize metrics
    avg_loss = total_loss / num_batches
    avg_path_log_prob = (total_path_log_prob / num_path_log_prob_batches if num_path_log_prob_batches > 0 else None) # PACK METRIC: Average path log-prob per batch, (float) (None if not returned by model)
    avg_path_prob = (total_path_prob / num_path_prob_batches if num_path_prob_batches > 0 else None) # PACK METRIC: Average path prob per batch, (float) (None if not returned by model)

    avg_metric = np.mean(all_metrics) if all_metrics else None

    return {
        "loss": avg_loss,
        "path_log_prob": avg_path_log_prob,
        "path_prob": avg_path_prob,
        "metric": avg_metric,
        "train_leaf_acc": (total_correct_leaf / num_leaf_acc_samples if num_leaf_acc_samples > 0 else None),
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
    num_path_log_prob_samples = 0

    total_correct_paths = 0
    num_path_acc_samples = 0

    total_correct_leaf = 0
    num_leaf_acc_samples = 0

    total_target_path_prob = 0.0
    num_target_path_prob_samples = 0

    total_pred_path_prob = 0.0
    num_pred_path_prob_samples = 0

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
    all_pred_leaf = []
    all_target_leaf = []
    all_path_scores = []
    candidate_paths = None

    # Debug logging
    all_masks_by_level = defaultdict(list)
    all_masked_logits_by_level = defaultdict(list)
    all_raw_logits_by_level = defaultdict(list)


    with torch.no_grad():
        for idx_batch, batch_data in enumerate(dataloader, 0):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_data.items()}

            inputs = batch["input"]
            hierarchy_target = batch["hierarchy_target"] # REQUIREMENT: All models should recieve the full hierarchy target (flat classifiers only consider last index)

            # Validation loss using teacher-forced true path
            loss_outputs = model.training_step(x=inputs, hierarchy_target=hierarchy_target)

            loss = safe_get(loss_outputs, "loss") # REQUIREMENT: Loss is expected in all models.
            if loss is None:
                raise ValueError("model.training_step() must return a 'loss' key.")

            # Prediction using highest-scoring valid path
            pred_outputs = model.predict(inputs)

            candidate_paths_candidate = safe_get(pred_outputs, "candidate_paths") # METRIC: Candidate paths considered during prediction. Expected shape (batch_size, num_candidate_paths, path_length). If not present: None
            if candidate_paths is None and candidate_paths_candidate is not None:
                candidate_paths = candidate_paths_candidate

            pred_paths = safe_get(pred_outputs, "path") # METRIC: Predicted path list. Expected shape (batch_size, path_length). If not present: None
            prediction = safe_get(pred_outputs, "prediction") # METRIC: Predicted label (leaf). Expected shape (batch_size,). If not present: None

            # (Save target and prediction path probabilities)
            target_path_prob = safe_get(loss_outputs, "path_prob") # METRIC: Target path conditional prob is expected in leveled models. If not present: None
            pred_path_prob = safe_get(pred_outputs, "path_prob") # METRIC: Predicted path conditional prob is expected in leveled models. If not present: None

            if target_path_prob is not None:
                total_target_path_prob += target_path_prob.sum().item() # PACK METRIC: Total target path prob summed over all samples, (float) (None if not returned by model)
                num_target_path_prob_samples += target_path_prob.numel() 
                all_target_path_probs.append(target_path_prob.detach().cpu()) # PACK METRIC: List of target path probs for all samples, (list of floats) (None if not returned by model)

            if pred_path_prob is not None:
                total_pred_path_prob += pred_path_prob.sum().item() # PACK METRIC: Total predicted path prob summed over all samples, (float) (None if not returned by model)
                num_pred_path_prob_samples += pred_path_prob.numel()  
                all_pred_path_probs.append(pred_path_prob.detach().cpu()) # PACK METRIC: List of predicted path probs for all samples, (list of floats) (None if not returned by model)

            path_scores = safe_get(pred_outputs, "path_scores") # METRIC: Path scores for predicted paths. Expected shape (batch_size, num_candidate_paths). If not present: None
            if path_scores is not None:
                all_path_scores.append(path_scores.detach().cpu())

            target_probs_by_level = safe_get(loss_outputs, "probs") # METRIC: All per level probabilities for the target path. Expected shape (batch_size, num_levels, num_classes_per_level). If not present: None
            if target_probs_by_level is not None:
                for level, probs in target_probs_by_level.items():
                    all_probs_by_level[level].append(probs.detach().cpu()) # PACK METRIC: Dictionary of lists of per-level probabilities for the target path, (dict of list of tensors) (None if not returned by model)

        
            if debug_validation:
                masks = safe_get(loss_outputs, "masks") # METRIC: Masks applied at each level. Expected shape (batch_size, num_classes_per_level). If not present: None
                if masks is not None:
                    for level, mask in masks.items():
                        all_masks_by_level[level].append(mask.detach().cpu()) # PACK METRIC: Dictionary of lists of masks applied at each level, (dict of list of tensors) (None if not returned by model)

                masked_logits = safe_get(loss_outputs, "masked_logits") # METRIC: Masked logits at each level. Expected shape (batch_size, num_classes_per_level). If not present: None
                if masked_logits is not None:
                    for level, masked_logits in masked_logits.items():
                        all_masked_logits_by_level[level].append(masked_logits.detach().cpu()) # PACK METRIC: Dictionary of lists of masked logits at each level, (dict of list of tensors) (None if not returned by model)

                raw_logits = safe_get(loss_outputs, "raw_logits") # METRIC: Raw logits at each level (before masking). Expected shape (batch_size, num_classes_per_level). If not present: None
                if raw_logits is not None:
                    for level, raw_logits in raw_logits.items():
                        all_raw_logits_by_level[level].append(raw_logits.detach().cpu()) # PACK METRIC: Dictionary of lists of raw logits at each level, (dict of list of tensors) (None if not returned by model)


            # Full path accuracy
            correct_paths = None
            correct_leaf = None
            target_leaf = hierarchy_target[:, -1]

            # Leaf accuracy
            if pred_paths is not None:
                correct_paths = (pred_paths == hierarchy_target).all(dim=1)
                total_correct_paths += correct_paths.sum().item()
                num_path_acc_samples += correct_paths.numel()

                pred_leaf = pred_paths[:, -1]
                correct_leaf = pred_leaf == target_leaf

            elif prediction is not None:
                pred_leaf = prediction
                correct_leaf = pred_leaf == target_leaf

            else:
                pred_leaf = None

            if correct_leaf is not None:
                total_correct_leaf += correct_leaf.sum().item()
                num_leaf_acc_samples += correct_leaf.numel()

            # Per-level accuracy
            if pred_paths is not None:
                for level in range(hierarchy_target.shape[1]):
                    level_correct[level] += (pred_paths[:, level] == hierarchy_target[:, level]).sum().item()
                    level_total[level] += inputs.shape[0]

            total_loss += loss.item()

            path_log_prob = safe_get(loss_outputs, "path_log_prob") # METRIC: Path log-prob is expected in leveled models. If not present: None
            if path_log_prob is not None:
                total_path_log_prob += path_log_prob.sum().item()
                num_path_log_prob_samples += path_log_prob.numel()

            total_samples += inputs.shape[0]
            num_batches += 1

            if pred_paths is not None:
                all_pred_paths.append(pred_paths.detach().cpu())

            all_target_paths.append(hierarchy_target.detach().cpu())

            if pred_leaf is not None:
                all_pred_leaf.append(pred_leaf.detach().cpu())

            all_target_leaf.append(target_leaf.detach().cpu())

            if idx_batch % 20 == 0:
                elapsed = time.time() - val_start_time

                msg = (
                    f"[VAL] "
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[+{elapsed/60:.1f} min] "
                    f"Batch {idx_batch}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )
                if correct_paths is not None:
                    msg += f" | Path acc: {correct_paths.float().mean().item():.4f}"
                if correct_leaf is not None:
                    msg += f" | Leaf acc: {correct_leaf.float().mean().item():.4f}"
                print(msg, flush=True)


            if metrics_fn is not None:
                metric = metrics_fn(
                    loss_outputs=loss_outputs,
                    pred_outputs=pred_outputs,
                    batch=batch,
                )
                all_metrics.append(metric)

    pred_paths_all = safe_cat(all_pred_paths)
    target_paths_all = safe_cat(all_target_paths)
    pred_leaf_all = safe_cat(all_pred_leaf)
    target_leaf_all = safe_cat(all_target_leaf) 

    per_level_acc = {
        level: level_correct[level] / level_total[level]
        for level in level_correct
        if level_total[level] > 0
    }

    target_path_probs_all = safe_cat(all_target_path_probs)
    pred_path_probs_all = safe_cat(all_pred_path_probs)
    path_scores_all = safe_cat(all_path_scores)

    probs_by_level_all = safe_cat_dict(all_probs_by_level)

    masks_by_level_all = {}
    masked_logits_by_level_all = {}
    raw_logits_by_level_all = {}
    
    if debug_validation:
        probs_by_level_all = safe_cat_dict(all_probs_by_level)
        masks_by_level_all = safe_cat_dict(all_masks_by_level)
        masked_logits_by_level_all = safe_cat_dict(all_masked_logits_by_level)
        raw_logits_by_level_all = safe_cat_dict(all_raw_logits_by_level)

    return {
        "loss": total_loss / num_batches,

        "path_log_prob": (
            total_path_log_prob / num_path_log_prob_samples
            if num_path_log_prob_samples > 0
            else None
        ),

        "path_acc": (
            total_correct_paths / num_path_acc_samples
            if num_path_acc_samples > 0
            else None
        ),

        "leaf_acc": (
            total_correct_leaf / num_leaf_acc_samples
            if num_leaf_acc_samples > 0
            else None
        ),

        "per_level_acc": per_level_acc,

        "pred_paths": pred_paths_all,
        "target_paths": target_paths_all,
        "pred_leaf": pred_leaf_all,
        "target_leaf": target_leaf_all,

        "all_metrics": all_metrics,

        "target_path_prob": (
            total_target_path_prob / num_target_path_prob_samples
            if num_target_path_prob_samples > 0
            else None
        ),

        "pred_path_prob": (
            total_pred_path_prob / num_pred_path_prob_samples
            if num_pred_path_prob_samples > 0
            else None
        ),

        "target_path_probs": target_path_probs_all,
        "pred_path_probs": pred_path_probs_all,
        "path_scores": path_scores_all,
        "candidate_paths": candidate_paths,

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
    print(f"TRAINING STARTED: {experiment_name}, {experiment_start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    history = {
        # Save metrics and other measurements
        "train_loss": [],
        "val_loss": [],

        "train_leaf_acc": [],
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
        "val_pred_leaf": [],
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
        if train_stats["path_log_prob"] is not None:
            print(f"[EPOCH {epoch + 1}] Train Path LogProb: {train_stats['path_log_prob']:.4f}")

        if val_stats["path_log_prob"] is not None:
            print(f"[EPOCH {epoch + 1}] Val Path LogProb: {val_stats['path_log_prob']:.4f}")

        if val_stats["path_acc"] is not None:
            print(f"[EPOCH {epoch + 1}] Val Path Acc: {val_stats['path_acc']:.4f}")

        if val_stats["leaf_acc"] is not None:
            print(f"[EPOCH {epoch + 1}] Val Leaf Acc: {val_stats['leaf_acc']:.4f}")

        if metrics_fn is not None:
            print(f"[EPOCH {epoch + 1}] Train Metric: {train_stats['metric']}")
            print(f"[EPOCH {epoch + 1}] Val Metric:   {val_stats['all_metrics']}")

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_leaf_acc"].append(train_stats["train_leaf_acc"])
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
            "train_leaf_acc": train_stats["train_leaf_acc"],
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
            pred_leaf=val_stats["pred_leaf"],
            target_leaf=val_stats["target_leaf"],
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
                pred_leaf=val_stats["pred_leaf"],
                target_leaf=val_stats["target_leaf"],
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

    print(f"TRAINING FINISHED: {experiment_name}")
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