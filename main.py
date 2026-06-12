from copy import deepcopy
from experiments.run_experiment import run_experiment

from configs.test_config import CONFIG as TEST_CONFIG
from configs.ar_hierarchy_masked import CONFIG as AR_HIERARCHY_MASKED
from configs.independent_hierarchy import CONFIG as INDEPENDENT_HIERARCHY
from configs.flat import CONFIG as FLAT


# Prepare Audio Dataset Adapter:
# TODO: Done manually as of June 1. 2026 on HPC. Should be automated in the future.

# Test config
CONFIG_TEST = TEST_CONFIG
# run_experiment(CONFIG_1)


# Define experemet configurations to run:

# Iterations (seeds: Reproducibility , folds: Cross-Validation estimates)
seeds = [80]
cv_folds = [2,3,5,6,8,9]
test_fold = 10

procedure_type = 2

CONFIG_1 = AR_HIERARCHY_MASKED
CONFIG_2 = INDEPENDENT_HIERARCHY
CONFIG_3 = FLAT

base_configs = [CONFIG_1, CONFIG_2, CONFIG_3] # Enter confits to queue experiments for


# Single procedure (0)
if procedure_type == 0:

    for config in base_configs:
        run_experiment(config)


# Seed procedure (1)
if procedure_type == 1:


    for config in base_configs:
        name_identifier = config["experiment_name"]
        
        for seed in seeds:
            config["seed"] = seed
            config["experiment_name"] = f"{name_identifier}_seed{seed}"
            run_experiment(config)


# Cross-Validation procedure (2)
if procedure_type == 2:
    

    for base_config in base_configs:
        name_identifier = base_config["experiment_name"]

        # Stage 1: CV estimate
        for val_fold in cv_folds:
            config = deepcopy(base_config)
            config["split_mode"] = "cv"
            config["test_fold"] = test_fold
            config["val_fold"] = val_fold
            config["experiment_name"] = f"{name_identifier}_cv_valfold{val_fold}"
            run_experiment(config)

        # Stage 2: final held-out test
        config = deepcopy(base_config)
        config["split_mode"] = "final_test"
        config["test_fold"] = test_fold
        config["experiment_name"] = f"{name_identifier}_final_test_fold{test_fold}"
        run_experiment(config)


# Seeded Cross-Validation procedure (3)
if procedure_type == 3:
    raise NotImplementedError()


