import torch
from model_frameworks.model_utilities import ConvBlock
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        self.blocks = nn.ModuleList([
            ConvBlock(1,   64,  num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(64,  128, num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(128, 256, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(256, 512, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(512, 512, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Expected input: (batch, 1, n_mels, time)

        for block in self.blocks:
            x = block(x)
        # Alternatively: let self.blocks = nn.Sequential(..ConvBlocks..) --> x = self.blocks(x)

        # Global pooling
        x_max = torch.amax(x, dim=[2, 3])
        x_mean = torch.mean(x, dim=[2, 3])
        x = x_max + x_mean

        x = self.classifier(x)

        return x
    

class Hierarchical_VGG16(nn.Module):
    def __init__(self, dropout=0.3, h_class_distribution=None, feature_dim=512):
        """
        h_class_disribution: Hierarchical class distribution, dict of {n : c}
        - n: level of the hierarchy (0 is top-level, 1 is second-level, etc.), must have an order (e.g., 0, 1, 2,...)
        - c: number of classes at that level
        """
        super().__init__()

        if h_class_distribution is None:
            h_class_distribution = {}
        self.h_class_distribution = dict(sorted(h_class_distribution.items()))

        self.blocks = nn.ModuleList([
            ConvBlock(1,   64,  num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(64,  128, num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(128, 256, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(256, 512, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(512, 512, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
        ])
        # Shape of the final feature map: x = [batch, 512, H, W]

        self.heads = nn.ModuleDict()
        prev_class_count = 0

        for level, num_classes in self.h_class_distribution.items():
            input_dim = feature_dim + prev_class_count
            self.heads[str(level)] = nn.Sequential(
                # Dense 1
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                # Dense 2
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                # Output layer
                nn.Linear(256, num_classes)
            )

            prev_class_count += num_classes


    def forward(self, x):
        # Expected input: (batch, 1, n_mels, time)

        for block in self.blocks:
            x = block(x)
        # Alternatively: let self.blocks = nn.Sequential(..ConvBlocks..) --> x = self.blocks(x)

        # Global pooling
        x_max = torch.amax(x, dim=[2, 3])
        x_mean = torch.mean(x, dim=[2, 3])
        h = x_max + x_mean # [batch, 512]

        logits_by_level = {} # To store logits for each level of the hierarchy -> passed to loss function later
        previous_probs = [] # To store probabilities from previous levels for conditioning the next level's predictions

        for level in self.h_class_distribution.keys():
            if previous_probs:
                conditioned_input = torch.cat([h] + previous_probs, dim=1)  # Concatenate features with previous level's probabilities quantities 
            else:
                conditioned_input = h  # When 'previous_probs' is empty, we just use the features 'h' as we are at the top level of the hierarchy
            # The conditioning space of the feature vector [h] depends on the hierarchy.. !!
            # A challenge for transfer learning: 
            # - try using embedding layers to compress probabilities into fixed-size vectors, so the feature vector space doesn't grow with the hierarchy levels 

            logits = self.heads[str(level)](conditioned_input)
            logits_by_level[level] = logits

            probs = torch.softmax(logits, dim=1)
            previous_probs.append(probs.detach()) # NOTE: While probabilities on L1 -> affects L2 and should backpropogate to L1 if L2 misclassifies, not detaching could break training
                                                  # NOTE: Toggeling could be an option..
        return logits_by_level