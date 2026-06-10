import torch
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from pathlib import Path
from hierarchies.hierarchies import get_hierarchy_tree
from sklearn.model_selection import train_test_split
from model_frameworks.dataloader_utilities import AudioDataset, AudioTransform
from torch.utils.data import DataLoader
from training.fit import safe_get

from configs import test_config, ar_hierarchy_masked, independent_hierarchy, flat


config = test_config.CONFIG # config


torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])

dataset_df = pd.read_csv(config["dataset_path"])
print(f"File path for audio exists? = {Path(dataset_df.iloc[0]['audio_path']).exists()}")
hierarchy_tree = get_hierarchy_tree(config["hierarchy"])

# Dataloader debugging: --------------------------------------------------------------------------------
if config["dataset"] == "urbansound8k":
    # Official UrbanSound8K protocol:
    # fold 10 → test, folds 1–9 → train
    train_data = dataset_df[dataset_df["fold"] != 10].reset_index(drop=True)
    test_data  = dataset_df[dataset_df["fold"] == 10].reset_index(drop=True)

else:
    train_data, test_data = train_test_split(
        dataset_df,
        test_size= 1 - config["train_size"],
        stratify= dataset_df["class_id"],   # Split the data so that the class proportions are preserved in each split. 
                                            # Requires a list of classes, which indices correspond to the sample in the dataset_df
        random_state= config["seed"],
    )
# Temporary debug subset
if config.get("debug_small_data", False):
    train_data = (
        train_data
        .sample(n=min(100, len(train_data)), random_state=config["seed"])
        .reset_index(drop=True)
    )

    test_data = (
        test_data
        .sample(n=min(10, len(test_data)), random_state=config["seed"])
        .reset_index(drop=True)
    )

    print("DEBUG SMALL DATA ENABLED")

print("FULL Dataset: {}".format(dataset_df.shape))  # (nr of samples, columns)
print("TRAIN Dataset: {}".format(train_data.shape)) # (nr of samples, columns)
print("TEST Dataset: {}".format(test_data.shape))   # (nr of samples, columns)

transform = AudioTransform(sample_rate=config["sample_rate"], n_mels=config["n_mels"], n_fft=config["n_fft"], hop_length=config["hop_length"]) 
training_set = AudioDataset(df=train_data, transform=transform, split="train")
testing_set = AudioDataset(df=test_data, transform=transform, split="test")   

train_params = {'batch_size': config["batch_size"],
                    'shuffle': True,           # NOTE: 'True' is seen as bad practice, why?
                    'num_workers': config["num_workers"]
                    }

test_params = {'batch_size': config["batch_size"],
                    'shuffle': False,           # NOTE: 'True' is seen as bad practice, why?
                    'num_workers': config["num_workers"]
                    }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda' if torch.cuda.is_available() else 'cpu'
model = config["model_class"](hierarchy_tree=hierarchy_tree, **config["model_kwargs"]) # Init model class
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"]) 

demo_training_loader = DataLoader(training_set, **train_params)
batch = next(iter(demo_training_loader)) 
print("Batch Keys:", batch.keys()) # keys for each sample in the batch
inputs = batch["input"]
print("Batch input:",inputs.shape) # Shape informtion of the input
targets = batch["target"]
print("Batch targets:",targets) # List of target, on index corresponding to sample (Will appear random, because of shuffle = True in dataloader params)


# Model debugging: --------------------------------------------------------------------------------

total_loss = 0.0
total_correct_leaf = 0
num_leaf_acc_samples = 0
total_path_log_prob = 0.0
num_path_log_prob_batches = 0
total_path_prob = 0.0
num_path_prob_batches = 0

model.eval() # dropout disabled (used in validation/test)

for idx_batch, batch_data in enumerate(training_loader, 0):
    # Move tensors to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_data.items()}

    inputs = batch["input"]
    hierarchy_target = batch["hierarchy_target"] # REQUIREMENT: All models should recieve the full hierarchy target (flat classifiers only consider last index)

    # Forward + model-specific loss
    with torch.no_grad():
        outputs = model.training_step(x=inputs, hierarchy_target=hierarchy_target)

    loss = safe_get(outputs, "loss") # REQUIREMENT: Loss is expected in all models.
    if loss is None:
        raise ValueError("model.training_step() must return a 'loss' key.")

    path_log_prob = safe_get(outputs, "path_log_prob")
    manual_loss = -path_log_prob.mean()

    print("CE/summed loss:", loss.item())
    print("Manual NLL loss:", manual_loss.item())
    print("Difference:", abs(loss.item() - manual_loss.item()))

    break

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



