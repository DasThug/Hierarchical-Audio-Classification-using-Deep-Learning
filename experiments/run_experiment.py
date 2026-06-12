import torch
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from hierarchies.hierarchies import get_hierarchy_tree
from model_frameworks.dataloader_utilities import AudioDataset, AudioTransform
from training.fit import fit


@dataclass
class RunContext:
    """ Data transfer object for experiment run context, to avoid long argument lists. """
    log_csv_path: str
    prediction_csv_path: str
    debug_validation: bool
    debug_csv_path: str
    runtime_json_path: str
    experiment_name: str
    models_dir: Path = None


def run_experiment(config):

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    dataset_df = pd.read_csv(config["dataset_path"])
    print(f"File path for audio exists? = {Path(dataset_df.iloc[0]['audio_path']).exists()}")

    hierarchy_tree = get_hierarchy_tree(config["hierarchy"])

    # Dataloader:
    if config["dataset"] == "urbansound8k":
        # Official UrbanSound8K protocol:
        # fold 10 → test, folds 1–9 → train
        test_fold = config.get("test_fold", 10)
        split_mode = config.get("split_mode", "final_test") # "cv" or "final_test"

        if split_mode == "cv":
            val_fold = config["val_fold"]
            train_data = dataset_df[(dataset_df["fold"] != test_fold) & (dataset_df["fold"] != val_fold)].reset_index(drop=True) # folds [1-9 ¬ val fold]
            test_data = dataset_df[dataset_df["fold"] == val_fold].reset_index(drop=True) # val fold
            print(f"CV split: train folds != {test_fold},{val_fold} | val fold = {val_fold}")

        elif split_mode == "final_test":
            train_data = dataset_df[dataset_df["fold"] != test_fold].reset_index(drop=True)
            test_data = dataset_df[dataset_df["fold"] == test_fold].reset_index(drop=True)
        
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")
        

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

    # remember to init the log-Mel spectrogram transformer
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

    # Model:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    
    model = config["model_class"](hierarchy_tree=hierarchy_tree, **config["model_kwargs"]) # Init model class
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]) # TODO: There is other parameters to explore: betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)


    # DEBUGGING: Check the dataloader output
    demo_training_loader = DataLoader(training_set, **train_params)
    batch = next(iter(demo_training_loader)) 
    print("Batch Keys:", batch.keys()) # keys for each sample in the batch
    inputs = batch["input"]
    print("Batch input:",inputs.shape) # Shape informtion of the input
    targets = batch["target"]
    print("Batch targets:",targets) # List of target, on index corresponding to sample (Will appear random, because of shuffle = True in dataloader params)


    out_dir = Path("outputs") / config["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    context = RunContext(
        log_csv_path=f"outputs/{config['experiment_name']}/metrics.csv",
        prediction_csv_path=f"outputs/{config['experiment_name']}/predictions.csv",
        debug_validation=config.get("debug_predictions", False),
        debug_csv_path=f"outputs/{config['experiment_name']}/debug_predictions.csv",
        runtime_json_path=f"outputs/{config['experiment_name']}/runtime_summary.json",
        experiment_name=config["experiment_name"],
        models_dir=models_dir,
    )

    history = fit(
        model=model,
        train_loader=training_loader,
        val_loader=testing_loader,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        metrics_fn=None,
        augmentation_fn=None,
        scheduler=None,
        run_context=context,
    )

    # torch.save(model.state_dict(), out_dir / "final_model.pt") # Save final model weights
    pd.Series(config).astype(str).to_json(out_dir / "config.json", indent=4)

    return history
