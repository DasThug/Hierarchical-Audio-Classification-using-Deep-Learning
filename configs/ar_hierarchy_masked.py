from model_frameworks.models import MaskedHierarchicalVGG16

CONFIG = {

    # experiment
    "experiment_name": "ar",

    # dataset
    "dataset": "urbansound8k",
    "dataset_path": "data/urbansound8k_compiled.csv",

    # hierarchy
    "hierarchy": "urbansound8k",

    # model
    "model_class": MaskedHierarchicalVGG16,

    "model_kwargs": {
        "dropout": 0.4,
        "feature_dim": 512,
        "hidden_dims": (256,),
        "backbone_name": "cnn10"
    },
    "num_workers": 0, # Set to 0 for Windows, or if you encounter issues with multiprocessing

    # training
    "train_size": 0.8,   # Only used if dataset doesn't have official train/test split
    "test_size": 0.2,    # Only used if dataset doesn't have official train/test split
    
    "split_mode": "final_test",   # "cv" or "final_test"
    "test_fold": 10,      # UrbanSound Specific (Recommended for all datasets to preprocess fold labelling externally)
    "val_fold": 9,        # UrbanSound Specific (Recommended for all datasets to preprocess fold labelling externally)

    "epochs": 40,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4, # Adam = 0, AdamW = 1e-2
    "batch_size": 4,

    # audio
    "sample_rate": 22050,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 320,

    # reproducibility
    "seed": 42,

    # debugging
    "debug_small_data": False,
    "debug_predictions": False,
}