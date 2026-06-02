from model_frameworks.models import IndependentMultiHeadVGG16, FlatVGG16, MaskedHierarchicalVGG16

CONFIG = {

    # experiment
    "experiment_name": "TEST_cnn",

    # dataset
    "dataset": "urbansound8k",
    "dataset_path": "data/urbansound8k_compiled.csv",

    # hierarchy
    "hierarchy": "urbansound8k",

    # model
    "model_class": FlatVGG16,

    "model_kwargs": {
        "dropout": 0.3,
        "feature_dim": 512
    },
    "num_workers": 0, # Set to 0 for Windows, or if you encounter issues with multiprocessing

    # training
    "train_size": 0.8,   # Only used if dataset doesn't have official train/test split
    "test_size": 0.2,    # Only used if dataset doesn't have official train/test split
    "epochs": 1,
    "learning_rate": 1e-5,
    "batch_size": 4,

    # audio
    "sample_rate": 22050,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 512,

    # reproducibility
    "seed": 42,

    # debugging
    "debug_small_data": True,
    "debug_predictions": True,
}