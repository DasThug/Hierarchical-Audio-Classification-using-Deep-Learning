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

# Reproducibility seeds
seeds = [80, 81, 82, 83, 84, 85]

CONFIG_1 = AR_HIERARCHY_MASKED
CONFIG_2 = INDEPENDENT_HIERARCHY
CONFIG_3 = FLAT

for config in [TEST_CONFIG]:
    name_identifier = config["experiment_name"]
    
    for seed in seeds:
        config["seed"] = seed
        config["experiment_name"] = f"{name_identifier}_seed{seed}"
        run_experiment(config)


