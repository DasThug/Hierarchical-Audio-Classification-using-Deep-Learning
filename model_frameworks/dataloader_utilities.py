import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf

"""
We define a standard contract, that the Deep Learning model must recieve as input data:

{
    "input": Tensor,      # spectrogram (CNN input)
    "target": int,        # class label,                     → add multiple targets for multiclass learning
    "index": int,         # optional debugging
    "meta": dict          # optional future extensions,      → add "meta": {"hierarchy": tensor}
}

"""

class AudioTransform:
    """
    Convert raw audio waveforms into log-Mel spectrograms.

    This callable transform, applies a Mel spectrogram transformation
    using torchaudio and then performs logarithmic compression to
    produce log-Mel features, which are commonly used in audio and
    speech-related deep learning tasks.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of the input audio waveform. Default is 22050 Hz.
    n_mels : int, optional
        Number of Mel frequency bins to use. Default is 64.
    """

    def __init__(self, sample_rate=22050, n_mels=64): # Init Class params
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels
        )

    def __call__(self, waveform):
        x = self.mel(waveform)
        x = torch.log(x + 1e-6)
        return x


class AudioDataset(Dataset): 
    """
    Generic PyTorch Dataset for audio classification tasks.

    This dataset expects a precompiled pandas DataFrame where each row
    represents one audio sample and contains all necessary metadata to
    load and process the audio into a model-ready tensor.

    The dataset performs:
        1. Audio loading from file path
        2. Optional resampling to a fixed sample rate
        3. Conversion to mono
        4. Transformation into a spectrogram (or other feature representation)

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing at least the following required columns:

        REQUIRED COLUMNS:
        - "audio_path" : str
            Path to the audio file (.wav, .mp3, etc.). Can be absolute or relative.
        - "class_id" : int
            Integer class label used as target for training.

        OPTIONAL COLUMNS:
        - "class_label" : str
            Human-readable label (e.g., "dog_bark"). Used for debugging/analysis.
        - "hierarchy" : list or str
            Hierarchical label structure (e.g., ['human', 'speech']).
            Can be pre-encoded or stringified (will be passed through as-is).
        - "fold" : int
            Fold index for cross-validation (e.g., 1–10 in UrbanSound8K).
        - Any additional metadata columns will be ignored by the dataset
          but can be accessed via the "meta" dictionary.

    transform : callable
        A function or callable object that takes a waveform tensor of shape
        (1, num_samples) and returns a transformed tensor (e.g., spectrogram)
        of shape (1, mel_bins, time_steps).

    sample_rate : int, optional (default=22050)
        Target sample rate. Audio will be resampled if it differs from this value.
    
    split : str, 'train' or 'test'
        Determines if padding will randomize over epochs (for training data) or stay 
        the same across all epochs (for validation data).

    Returns
    -------
    dict
        A dictionary with the following keys:

        - "input" : torch.Tensor
            Transformed audio tensor (e.g., spectrogram) of shape (1, mel_bins, time_steps)
        - "target" : torch.Tensor
            Class label as a tensor of dtype torch.long
        - "index" : int
            Index of the sample in the dataset
        - "meta" : dict
            Dictionary containing optional metadata such as:
                - "label_name"
                - "hierarchy"
                - "fold"

    Notes
    -----
    - This dataset assumes that heavy preprocessing (e.g., dataset-specific logic,
      hierarchy construction, path resolution) has already been performed offline
      by a data adapter.
    - The dataset is designed to be model-agnostic and reusable across different
      audio classification tasks.
    """

    def __init__(self, df, transform, sample_rate=22050, split=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.sample_rate = sample_rate
        self.split = split

        required_cols = {"audio_path", "class_id"}
        required_split = {"train", "test", "val"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if split not in required_split:
            raise ValueError(f"Invalid split={split!r}. Expected one of {required_split}.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Direct path from CSV
        path = row["audio_path"]


        # TODO: DOWNGRADE TO PYTHON 3.11 TO USE TORCHCODEC, OR USE SOUNDFILE FOR NOW

        # waveform, sr = torchaudio.load(path) # using torchcodec (preinstalled)
        
        # mono
        # waveform = waveform.mean(dim=0, keepdim=True)

        # TODO: ----------------------------------------------------------------

        waveform, sr = sf.read(path) # usign soundfile
        waveform = torch.tensor(waveform, dtype=torch.float32)
        # stereo → mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        # add channel dim (important for CNN)
        waveform = waveform.unsqueeze(0)
        # TODO: ----------------------------------------------------------------


        # optional resample
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        # transform → spectrogram
        # Padding: All clips will be 4 seconds long
        target_len = int(self.sample_rate * 4)
        cur_len = waveform.shape[1]
        # If the clip is too long, -> Trim it
        if cur_len > target_len:
            trim_total = cur_len - target_len
            # Trim both sides: Randomly choose the start of a 4-second crop (during training)
            if self.split == "train":
                trim_left = torch.randint(0, trim_total + 1, (1,)).item()
            # Trim deterministically equally on both sides (during validation/testing)
            elif self.split in {"test", "val"}:
                trim_left = trim_total // 2
            
            trim_right = trim_left + target_len
            waveform = waveform[:, trim_left:trim_right] # Trim the clip, by reducing from left and right
        
        # If the clip is too short, -> Pad it
        elif cur_len < target_len:
            pad_total = target_len - cur_len

            # Pad each side randomly (during training)
            if self.split == "train":
                pad_left = torch.randint(0, pad_total + 1, (1,)).item()
            # Pad deterministically equally on both sides (during validation/testing)
            elif self.split in {"test", "val"}:
                pad_left = pad_total // 2
            
            pad_right = pad_total - pad_left
            waveform = F.pad(waveform, (pad_left, pad_right)) # Pad the clip, by adding zeros to left and right

            
        spec = self.transform(waveform)


        # Construct mapping of individual hierarchical level class targets:
        target_levels = {}
        hierarchical_targets = row.get("hierarchy_class_id", None) # [level_1, level_2, ..., level_n]
        for level in range(len(self.hierarchy_levels)):
            target_levels[f"target_level_{level}"] = (
                torch.tensor(hierarchical_targets[level], dtype=torch.long) # must be integer class indices for every hierarchy level
                if hierarchical_targets is not None else None
            )

        datapoint_data = {
            "input": spec,                                              # (1, mel, time)
            "target": torch.tensor(row["class_id"], dtype=torch.long),  # tensor('class_id')
            "index": idx,                                               # index: int
            "meta": {           
                "clip_id": row.get("clip_id", None),                    # clip_id: soundata clip id
                "label_name": row.get("class_label", None),             # label_name: str
                "hierarchy": row.get("hierarchy", None),                # [level_1, level_2, ..., level_n]
                "fold": int(row.get("fold", None))                      # int
            },
            **target_levels                                             #target_level_0, target_level_1, ..., target_level_n : tensor
        }

        # add target_levels to the datapoint data
        

        return datapoint_data