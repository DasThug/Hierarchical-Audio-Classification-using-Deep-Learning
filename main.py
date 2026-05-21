from experiments.run_experiment import run_experiment
from configs.ar_hierarchy_masked import CONFIG as AR_HIERARCHY_MASKED_CONFIG

# Prepare Audio Dataset Adapter:




CONFIG_1 = AR_HIERARCHY_MASKED_CONFIG
# CONFIG_2 = ... # You can define more configs for different experiments
# CONFIG_3 = ...

run_experiment(CONFIG_1)