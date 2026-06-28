import soundata
from hierarchies.hierarchies import get_hierarchy_tree
from writers import write_csv
from pathlib import Path


DATASET = "urbansound8k" # Choose any available dataset from the 'soundata' library

dataset = soundata.initialize(DATASET)
try:
    print("Validating if Dataset files exists:")
    dataset.validate()
except:
    print("Dataset could not be validated, downloading:")
    dataset.download()  # download the dataset
    dataset.validate()  # validate that all the expected files are there

HIERARCHY_TREE = get_hierarchy_tree(DATASET)
HIERARCHY_TREE.print_tree()

# Save MetaDataset to CSV

# Output folder
out_dir = Path("metadata")
out_dir.mkdir(parents=True, exist_ok=True)
output_path = out_dir / "{}_metadata.csv".format(DATASET)

rows = []
for clip_id in dataset.clip_ids:
    clip = dataset.clip(clip_id)

    hierarchy_indices = HIERARCHY_TREE.get_path(leaf_idx=clip.class_id, output="indices")

    rows.append({
        "clip_id": clip.clip_id,
        "class_id": clip.class_id,
        "freesound_start_time": clip.freesound_start_time,
        "freesound_end_time": clip.freesound_end_time,
        "salience": clip.salience,
        "slice_file_name": clip.slice_file_name,
        
        "class_label": clip.class_label,
        "hierarchy": hierarchy_indices
    })
write_csv(output_path, rows)