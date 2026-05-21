import urllib.request
import soundata
from hierarchies.hierarchies import get_hierarchy_tree
from pathlib import Path
import pandas as pd
import ast


####################################################
DATASET = "urbansound8k" # Choose any available dataset from the 'soundata' library
####################################################


dataset = soundata.initialize(DATASET)
try:
    print("Validating if Dataset files exists:")
    dataset.validate()
except:
    print("Dataset could not be validated, downloading:")
    dataset.download()  # download the dataset
    dataset.validate()  # validate that all the expected files are there

HIERARCHY_TREE = get_hierarchy_tree(DATASET)

# Save Dataset to CSV
# We will save a designated dataframe (CSV) coressponding to a distinct dataset in data/.
# Therefore it will only be necessary to run the Data Adapter once pr. dataset and the constructed dataframe
# can just be directly passed straight into the AudioDataset henceforth.

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)


meta_path = Path(f"metadata/{DATASET}_metadata.csv")
df_meta = pd.read_csv(meta_path)

rows = []

for _, row in df_meta.iterrows():

    clip_id = row["clip_id"]

    # Get clip from soundata
    clip = dataset.clip(clip_id)

    # Build audio path
    audio_path = clip.audio_path  # already correct path

    # Parse hierarchy (stored as string in CSV)
    hierarchy = row["hierarchy"]
    if isinstance(hierarchy, str):
        hierarchy = ast.literal_eval(hierarchy)

    rows.append({
        "clip_id": clip_id,
        "audio_path": str(audio_path),
        "class_id": int(row["class_id"]),
        "class_label": row["class_label"],
        "fold": int(clip.fold),
        "hierarchy": hierarchy
    })

df_out = pd.DataFrame(rows)

# Save compiled dataset
output_path = out_dir / f"{DATASET}_compiled.csv"
df_out.to_csv(output_path, index=False)

print(f"Saved {len(df_out)} samples to: {output_path}")