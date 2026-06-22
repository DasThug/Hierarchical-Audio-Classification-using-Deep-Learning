# Hierarchical-Audio-Classification-using-Deep-Learning

This is the repository for the Bachelor thesis project investigating hierarchical audio classification methods on audio datasets.

## Project Overview

This project currently compares three classification approaches:

- Flat CNN classifier
- Independent hierarchical classifier
- Autoregressive hierarchical classifier

The models are evaluated on a custom hierarchy built on top of the UrbanSound8K dataset.

## Repository Structure

```text
configs/                  Experiment configurations
data/                     Dataset properties and info
experiments/              Experiment loop procedures
figures/                  Output folder for certain figures and visualizations
hierarchies/              Hierarchy Tree class and defined hierarchy tree configurations (pr dataset)
Individual_Scripts        Single use scripts, contains baseline model and statistics utilities
metadata/                 Contains metadata, (Optional use for constructing final dataset csv)
model_frameworks/         Dataloader and Model utilities, defined modeles archiectures are contained here
outputs/                  Loaction of output experiment metrics from running experiments
plots/                    Output folder for certain figures and plots
training/                 Model training, validation loops and fit functions
utils/                    Basic notebook utility functions

main.py                   Main runner, configure types and numbers of queued experiments
datasetAdapter.py         Create, Load, Verify chosen dataset on machine
debug_model.py            Simplified experiment loop with debugging for careful evaluation of model dynamics
InspectData.ipynb         Utility notebook for inspecting output metrics
PipelineNotebook.ipynb    Utility notebook for simplified walkthrough of repository   

```

## Installation

```bash
git clone <repository>
cd <repository>

pip install -r requirements.txt
```

## Dataset

This project retrieves datasets using the Soundata python package. For other Soundata datasets, change the dataset_name and run the metadata generator in ```PipelineNotebook.ipynb```, afterwards run the metadata through ```datasetAdapter.py```.
It is also possible to use other datasets, without Soundata, as long as the following requirements at met, to match the framework:

- The dataset must have a dataset csv file in ```data/ ```, listing properties the mandatory columns: 
- ```clip_id```: unique sample identifier for debugging
- ```audio_path```: path for the raw audio waveform
- ```hierarchy```: must denote hierarchical label path as a list, where each index corresponds to the index at level "i"


- The dataset csv file is also recommended to include the following columns:
- ```class_id```: leaf-level class id
- ```class_label```: leaf-level class label
- ```fold```: predefined fold group

Place the dataset in:

```text
data/
```

## Hierarchy

Defining a designated hierarchy, tied to a dataset, is required before running hierarchical models. Define building function and retrieval logic in ```hierarchies/hierarchies.py/ ```, using methods from the hierarcy tree class in ```hierarchies/hierchyClass.py/ ```



## Running Experiments

Example:

```bash
python main.py
```
Thoroughly ensure that a configuration in ```configs/``` is valid, and used by the ```run_experiment``` in main. Further configure other experiment properties in main, for a designated experiment flow.

or directly pass a configuration in

```bash
python run_experiment.py
```


depending on your project structure.


## Inspect Data and Results:
After running a configuration, use the ```InspectData.ipynb``` notebook to calculate relevant evaluation metrics. Simply list the path of the experiment (```outputs/```) in any of the metric functions.



## Available Models

- Flat CNN
- Independent Multi-Head CNN
- Autoregressive Hierarchical CNN
