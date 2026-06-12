import torch
from model_frameworks.model_utilities import ConvBlock
import torch.nn as nn
import torch.nn.functional as F
from hierarchies.hierarchyClass import HierarchyTree

# CNN Backbone function:
def make_backbone(backbone_name:str, feature_dim = None):
    """
        Function for returning CNN backbone presets
        returns: nn.ModuleList, int: output feature_dim
    """
    if backbone_name == "vgg16":
        blocks = nn.ModuleList([
            ConvBlock(1,   64,  num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(64,  128, num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(128, 256, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(256, 512, num_convs=3, kernel_size=3, padding="same", pool=False),
            ConvBlock(512, 512, num_convs=3, kernel_size=3, padding="same", pool=False)
        ])

        feature_dim = 512
    
    elif backbone_name == "vggish":
        blocks = nn.ModuleList([
            ConvBlock(1,   64,  num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(64,  128, num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(128, 256, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(256, 512, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
            ConvBlock(512, 512, num_convs=3, kernel_size=3, padding="same", pool=True, pool_type="max", pool_size=2),
        ])

        feature_dim = 512
    
    elif  backbone_name == "cnn10":
        blocks = nn.ModuleList([
            ConvBlock(1,   64,  num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="avg", pool_size=2),
            ConvBlock(64,  128, num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="avg", pool_size=2),
            ConvBlock(128, 256, num_convs=2, kernel_size=3, padding="same", pool=True, pool_type="avg", pool_size=2),
            ConvBlock(256, 512, num_convs=2, kernel_size=3, padding="same", pool=False)
        ])

        feature_dim = 512
    
    elif  backbone_name == "cnn6":
        blocks = nn.ModuleList([
            ConvBlock(1,   64,  num_convs=1, kernel_size=5, padding="same", pool=True, pool_type="avg", pool_size=2),
            ConvBlock(64,  128, num_convs=1, kernel_size=5, padding="same", pool=True, pool_type="avg", pool_size=2),
            ConvBlock(128, 256, num_convs=1, kernel_size=5, padding="same", pool=True, pool_type="avg", pool_size=2),
            ConvBlock(256, 512, num_convs=1, kernel_size=5, padding="same", pool=False)
        ])
            
        feature_dim = 512
    
    else:
        raise ValueError(
            f"Unknown backbone_name='{backbone_name}'. "
            "Choose from: 'vgg16', 'cnn10', 'cnn6'."
        )

    return blocks, feature_dim



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

class old_FlatVGG16(nn.Module):
    """
    Flat VGG16 baseline compatible with the hierarchical training framework.

    - Uses VGG-style backbone.
    - Predicts only the leaf-level class.
    - Expects hierarchy_target with shape [batch, depth].
    - Uses hierarchy_target[:, -1] as the flat target.
    """

    def __init__(self,
            hierarchy_tree: HierarchyTree, 
            dropout=0.3,
            feature_dim=512,
            hidden_dims=(512, 256),
            mask_value=-1e9,
        ):

        super().__init__()

        self.num_classes = hierarchy_tree.num_classes_per_level()[hierarchy_tree.total_depth - 1]

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
            nn.Linear(256, self.num_classes),
        )

    def extract_features(self, x):
        # Expected input: [batch, 1, n_mels, time]
        for block in self.blocks:
            x = block(x)

        # global pooling
        x_max = torch.amax(x, dim=[2, 3])
        x_mean = torch.mean(x, dim=[2, 3])

        return x_max + x_mean

    def forward(self, x):
        """
        Returns raw leaf-level logits.
        Shape: [batch, num_leaf_classes]
        """
        h = self.extract_features(x)
        logits = self.classifier(h)
        return logits

    def compute_loss(self, logits, hierarchy_target):
        """
        Compute flat cross-entropy loss using only the leaf-level target.
        """
        if hierarchy_target.ndim == 1:
            leaf_target = hierarchy_target
        else:
            leaf_target = hierarchy_target[:, -1]

        leaf_target = leaf_target.to(device=logits.device, dtype=torch.long)

        loss = F.cross_entropy(logits, leaf_target)

        probs = torch.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        selected_log_prob = log_probs.gather(
            1,
            leaf_target.unsqueeze(1)
        ).squeeze(1)

        return {
            "loss": loss,
            "logits": logits,
            "probs": probs,
            "log_probs": log_probs,
            "path_log_prob": selected_log_prob,
            "path_prob": selected_log_prob.exp(),
            "leaf_target": leaf_target,
        }

    def forward_with_loss(self, x, hierarchy_target):
        logits = self.forward(x)
        loss_outputs = self.compute_loss(logits, hierarchy_target)
        loss_outputs["raw_logits"] = logits
        return loss_outputs

    @torch.no_grad()
    def predict(self, x):
        """
        Returns prediction in a format compatible with hierarchical validation.

        For the flat model, `path` is only the predicted leaf class with shape [batch, 1].
        If you want full hierarchy paths, you need to map leaf predictions back through
        the hierarchy tree externally.
        """
        logits = self.forward(x)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        pred_leaf = torch.argmax(log_probs, dim=1)

        return {
            "logits": logits,
            "raw_logits": logits,
            "log_probs": log_probs,
            "probs": probs,
            "leaf_pred": pred_leaf,
            "path": pred_leaf.unsqueeze(1),
            "path_log_prob": log_probs.gather(1, pred_leaf.unsqueeze(1)).squeeze(1),
            "path_prob": probs.gather(1, pred_leaf.unsqueeze(1)).squeeze(1),
        }


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


############################
# Updated models: 
############################

class FlatVGG16(nn.Module):
    """
    Flat VGG-style baseline compatible with the shared training/validation loop.

    - Shared VGG-style CNN backbone.
    - Single classifier head.
    - Predicts only the leaf-level class.
    - Uses hierarchy_target[:, -1] as the target.
    """

    def __init__(
        self,
        hierarchy_tree: HierarchyTree,
        dropout=0.3,
        feature_dim=512,
        hidden_dims=(512, 256),
        backbone_name="cnn10"
    ):
        super().__init__()

        self.hierarchy_tree = hierarchy_tree
        self.depth = hierarchy_tree.total_depth
        self.class_counts = dict(sorted(hierarchy_tree.num_classes_per_level().items()))
        self.feature_dim = feature_dim
        self.leaf_level = self.depth - 1
        self.num_classes = self.class_counts[self.leaf_level]

        # CNN Backbone 
        self.blocks, backbone_feature_dim = make_backbone(backbone_name=backbone_name)
        if feature_dim != backbone_feature_dim:
            raise ValueError("feature_dim must match backbone")

        # A single leaf classifier head that recieves a feature vector h
        self.classifier = self._make_head(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=self.num_classes,
            dropout=dropout,
        )

    def _make_head(self, input_dim, hidden_dims, output_dim, dropout):
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        # Pass input through CNN backbone: x -> blocks -> global pooling -> feature vector h
        # Expected input: [batch, 1, n_mels, time]
        for block in self.blocks:
            x = block(x)

        # global pooling
        x_max = torch.amax(x, dim=[2, 3])
        x_mean = torch.mean(x, dim=[2, 3])
        return x_max + x_mean

    def forward(self, x):
        h = self.extract_features(x)

        # compute raw logits for the leaf levelclassifier
        logits = self.classifier(h)
        return logits

    def compute_loss(self, logits, hierarchy_target):
        if hierarchy_target.ndim == 1:
            leaf_target = hierarchy_target
        else:
            leaf_target = hierarchy_target[:, -1]

        leaf_target = leaf_target.to(device=logits.device, dtype=torch.long)

        loss = F.cross_entropy(logits, leaf_target)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        selected_log_prob = log_probs.gather(
            1,
            leaf_target.unsqueeze(1)
        ).squeeze(1)

        return {
            "loss": loss,

            "path_log_prob": selected_log_prob, # Is just the single leaf log probability
            "path_prob": selected_log_prob.exp(), # Is just the single leaf probability

            "log_probs": {self.leaf_level: log_probs}, # one leaf key -> log probs Tensor for all leaf classes
            "probs": {self.leaf_level: probs}, # one leaf key -> probs Tensor for all leaf classes
            "raw_logits": {self.leaf_level: logits}, # one leaf key -> raw logits Tensor for all leaf classes

            # NOTE: Can possibly add a train prediction 
        }

    def training_step(self, x, hierarchy_target):
        # forward() returns raw logits by level
        logits = self.forward(x)

        # compute_loss validates targets and returns logits/log-probs/losses
        loss_outputs = self.compute_loss(logits=logits, hierarchy_target=hierarchy_target)

        # retrieve a prediction for the leaf metric during training 
        pred_leaf = torch.argmax(loss_outputs["log_probs"][self.leaf_level], dim=1)
        loss_outputs["train_prediction"] = pred_leaf
        return loss_outputs

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        pred_leaf = torch.argmax(log_probs, dim=1)

        pred_log_prob = log_probs.gather(
            1,
            pred_leaf.unsqueeze(1)
        ).squeeze(1)

        return {
            "prediction": pred_leaf, # instead of 'pred_path' for multi level models
            "path_log_prob": pred_log_prob,
            "path_prob": pred_log_prob.exp(),
            "raw_logits": {self.leaf_level: logits},
        }


class IndependentMultiHeadVGG16(nn.Module):
    """
    Independent multi-head VGG-style CNN.

    - Shared CNN backbone.
    - One independent classifier head per hierarchy level.
    """

    def __init__(
        self,
        hierarchy_tree: HierarchyTree,
        dropout=0.3,
        feature_dim=512,
        hidden_dims=(512, 256),
        backbone_name="cnn10"
    ):
        super().__init__()

        self.hierarchy_tree = hierarchy_tree
        self.depth = hierarchy_tree.total_depth
        self.class_counts = dict(sorted(hierarchy_tree.num_classes_per_level().items()))
        self.feature_dim = feature_dim

        expected_levels = list(range(self.depth))
        if list(self.class_counts.keys()) != expected_levels:
            raise ValueError(
                "Hierarchy levels must be contiguous and zero-indexed. "
                f"Expected {expected_levels}, got {list(self.class_counts.keys())}."
            )
        
        # CNN Backbone 
        self.blocks, backbone_feature_dim = make_backbone(backbone_name=backbone_name)
        if feature_dim != backbone_feature_dim:
            raise ValueError("feature_dim must match backbone")
        
        # Heads: one head per level, each receives only feature vector h
        self.heads = nn.ModuleDict()
        for level in range(self.depth):
            self.heads[str(level)] = self._make_head(
                input_dim=feature_dim,
                hidden_dims=hidden_dims,
                output_dim=self.class_counts[level],
                dropout=dropout,
            )

    def _make_head(self, input_dim, hidden_dims, output_dim, dropout):
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        # Pass input through CNN backbone: x -> blocks -> global pooling -> feature vector h
        # Expected input: [batch, 1, n_mels, time]
        for block in self.blocks:
            x = block(x)

        # global pooling
        x_max = torch.amax(x, dim=[2, 3])
        x_mean = torch.mean(x, dim=[2, 3])
        return x_max + x_mean

    def forward(self, x):
        h = self.extract_features(x)

        raw_logits_by_level = {}

        # compute raw logits for each level (heads only depend on h)
        for level in range(self.depth):
            raw_logits_by_level[level] = self.heads[str(level)](h)

        return raw_logits_by_level

    def compute_loss(self, raw_logits_by_level, hierarchy_target):
        first_logits = raw_logits_by_level[0]
        batch_size = first_logits.shape[0]
        device = first_logits.device

        if hierarchy_target.ndim == 1:
            hierarchy_target = hierarchy_target.unsqueeze(0)

        hierarchy_target = hierarchy_target.to(device=device, dtype=torch.long)

        if hierarchy_target.shape[0] != batch_size:
            raise ValueError("hierarchy_target batch size must match logits batch size")

        if hierarchy_target.shape[1] != self.depth:
            raise ValueError(
                f"hierarchy_target must contain one class index per hierarchy level. "
                f"Expected {self.depth}, got {hierarchy_target.shape[1]}."
            )

        log_probs_by_level = {}
        probs_by_level = {}

        losses = []
        selected_log_probs = []

        for level in range(self.depth):
            logits = raw_logits_by_level[level]
            target_l = hierarchy_target[:, level]

            loss_l = F.cross_entropy(logits, target_l)
            losses.append(loss_l)

            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()

            selected_logp = log_probs.gather(1,target_l.unsqueeze(1)).squeeze(1)
            selected_log_probs.append(selected_logp)

            log_probs_by_level[level] = log_probs
            probs_by_level[level] = probs

        total_loss = sum(losses)

        # This is the product of independent per-level probabilities assigned to the target labels.
        path_log_prob = torch.stack(selected_log_probs, dim=1).sum(dim=1)

        return {
            # Total training loss across all hierarchy levels
            # Tensor shape: []
            "loss": total_loss,

            # Per-level losses
            # losses[level] -> scalar tensor
            "losses": losses,

            # Per-level log probabilities
            # log_probs[level] -> Tensor[batch_size, num_classes_level]
            # Access: log_probs[level][batch_idx, class_idx]
            "log_probs": log_probs_by_level,

            # Per-level probabilities
            # probs[level] -> Tensor[batch_size, num_classes_level]
            # Access: probs[level][batch_idx, class_idx]
            "probs": probs_by_level,

            # Joint log-probability of the target path
            # Tensor[batch_size]
            # Access: path_log_prob[batch_idx]
            "path_log_prob": path_log_prob,

            # Joint probability of the target path
            # Tensor[batch_size]
            # Access: path_prob[batch_idx]
            "path_prob": path_log_prob.exp(),

            # Raw classifier outputs before softmax
            # raw_logits[level] -> Tensor[batch_size, num_classes_level]
            # Access: raw_logits[level][batch_idx, class_idx]
            "raw_logits": raw_logits_by_level,
        }

    def training_step(self, x, hierarchy_target):
        # forward() returns raw logits by level
        raw_logits_by_level = self.forward(x)

        # compute_loss validates targets and returns logits/log-probs/losses
        return self.compute_loss(raw_logits_by_level=raw_logits_by_level, hierarchy_target=hierarchy_target)

    @torch.no_grad()
    def predict(self, x):
        raw_logits_by_level = self.forward(x)

        preds_by_level = []
        selected_log_probs = []

        # Predict the most probable path level-by-level independently
        for level in range(self.depth):
            logits = raw_logits_by_level[level]
            log_probs = F.log_softmax(logits, dim=1)

            pred_l = torch.argmax(log_probs, dim=1)
            preds_by_level.append(pred_l)

            selected_logp = log_probs.gather(1, pred_l.unsqueeze(1)).squeeze(1)
            selected_log_probs.append(selected_logp)

        pred_path = torch.stack(preds_by_level, dim=1)
        path_log_prob = torch.stack(selected_log_probs, dim=1).sum(dim=1)

        return {
            "path": pred_path,
            "path_log_prob": path_log_prob,
            "path_prob": path_log_prob.exp(),
            "raw_logits": raw_logits_by_level,
        }



class MaskedHierarchicalVGG16(nn.Module):
    """
    Masked hierarchical VGG16 model.

    - Shared VGG-style CNN backbone producing feature vector `h`.
    - One global classifier head per hierarchy level that takes only `h`.
    - Conditioning implemented by hard-masking logits using the hierarchy masks.

    Training: teacher forcing with `hierarchy_target` (shape [batch, depth]).
    Inference: score all valid root-to-leaf paths and pick the highest joint log-prob.
    """

    def __init__(
        self,
        hierarchy_tree: HierarchyTree,
        dropout=0.3,
        feature_dim=512,
        hidden_dims=(512, 256),
        mask_value=-1e9,
        backbone_name="cnn10"
    ):
        super().__init__()

        self.hierarchy_tree = hierarchy_tree
        self.depth = hierarchy_tree.total_depth
        self.class_counts = dict(sorted(hierarchy_tree.num_classes_per_level().items()))
        self.feature_dim = feature_dim
        self.mask_value = mask_value

        expected_levels = list(range(self.depth))
        if list(self.class_counts.keys()) != expected_levels:
            raise ValueError(
                "Hierarchy levels must be contiguous and zero-indexed. "
                f"Expected {expected_levels}, got {list(self.class_counts.keys())}."
            )

        # CNN Backbone 
        self.blocks, backbone_feature_dim = make_backbone(backbone_name=backbone_name)
        if feature_dim != backbone_feature_dim:
            raise ValueError("feature_dim must match backbone")

        # Heads: one head per level, each receives only feature vector h
        self.heads = nn.ModuleDict()
        for level in range(self.depth):
            self.heads[str(level)] = self._make_head(
                input_dim=feature_dim,
                hidden_dims=hidden_dims,
                output_dim=self.class_counts[level],
                dropout=dropout,
            )

        # Hierarchy masks registered as buffers
        self._register_hierarchy_masks()

        # Cache of enumerated root-to-leaf paths (list of lists of ints)
        self._all_paths = None

    def _make_head(self, input_dim, hidden_dims, output_dim, dropout):
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _register_hierarchy_masks(self):
        """
        Precompute and register hierarchy masks as buffers for efficient masking during forward pass.
        - Level 0 mask: shape [num_classes_level0], True for valid root children.
        - Level >0 masks: shape [num_parent_classes, num_child_classes], True for valid parent-child relationships.
        """
        # Level 0: 1D mask of valid root children
        root_mask = torch.zeros(self.class_counts[0], dtype=torch.bool)
        for child_idx in self.hierarchy_tree.get_children("root", output="indices"):
            root_mask[child_idx] = True # all level-0 classes are valid children of the root.
        self.register_buffer("mask_level_0", root_mask) # store buffer inside model.

        # Levels >0: mask of shape [num_parent_classes, num_child_classes]
        for level in range(1, self.depth):
            parent_count = self.class_counts[level - 1] # classes at level l-1 
            child_count = self.class_counts[level]      # classes at level l
            mask = torch.zeros(parent_count, child_count, dtype=torch.bool) # shape [rows = num_parent_classes, cols = num_child_classes] Matrix

            for parent_idx in range(parent_count):
                child_indices = self.hierarchy_tree.get_children(
                    level=level - 1,
                    level_idx=parent_idx,
                    output="indices",
                )
                mask[parent_idx, child_indices] = True # Set true for valid parent-child relationships at this level of the hierarchy

            self.register_buffer(f"mask_level_{level}", mask) # store buffer inside model.

    def extract_features(self, x):
        # Pass input through CNN backbone: x -> blocks -> global pooling -> feature vector h
        # Expected input: [batch, 1, n_mels, time]
        for block in self.blocks:
            x = block(x)

        # global pooling
        x_max = torch.amax(x, dim=[2, 3])
        x_mean = torch.mean(x, dim=[2, 3])
        return x_max + x_mean

    def _get_mask(self, level, parent_ids, batch_size):
        """
        Return masks expanded for a batch.
        level 0: returns [batch, num_classes_level0]
        level >0: parent_ids is a tensor of parent indices (batch,) and returns [batch, num_child_classes]
        """
        mask = getattr(self, f"mask_level_{level}") # get mask buffer stored in model

        if level == 0:
            return mask.unsqueeze(0).expand(batch_size, -1) # For level 0, the same mask applies to all samples in the batch, so we expand it along the batch dimension.

        # parent_ids may be a single int (when enumerating a path) or a tensor
        return mask[parent_ids] # For level >0, index the mask with the parent_ids to get appropriate child mask for each sample in the batch. 
        # The resulting shape is [batch, num_child_classes] because we are selecting one row of the mask per sample based on its parent class index.

    def _masked_log_softmax(self, logits, mask):
        """
        Apply hard mask to logits then return masked_logits and log_probs.
        """
        masked_logits = logits.masked_fill(~mask, self.mask_value)
        log_probs = F.log_softmax(masked_logits, dim=1)
        return masked_logits, log_probs

    def _enumerate_paths(self):
        """
        Return a tensor [num_paths, depth] of all valid root-to-leaf paths
        where each path is a list of class indices (one per level).

        Uses `hierarchy_tree.leaf_nodes()` when available, otherwise falls
        back to a DFS using `get_children`. Caches the result on CPU.
        """
        if self._all_paths is not None:
            return self._all_paths

        paths = []
        try:
            leaf_nodes = list(self.hierarchy_tree.leaf_nodes())
            paths = [node.path(output="indices") for node in leaf_nodes]
        except Exception:
            raise ValueError("Warning: hierarchy_tree did not support leaf_nodes()")
            # Fallback DFS (can be applicable, but if leaf_nodes() is supported, it's likely more efficient than DFS, especially for large hierarchies)
            def dfs(level, parent_idx, current):
                if level == self.depth:
                    paths.append(current.copy())
                    return

                if level == 0:
                    children = self.hierarchy_tree.get_children("root", output="indices")
                else:
                    children = self.hierarchy_tree.get_children(level=level - 1, level_idx=parent_idx, output="indices")

                for c in children:
                    current.append(int(c))
                    dfs(level + 1, c, current)
                    current.pop()

            dfs(0, None, [])

        if len(paths) == 0:
            raise RuntimeError("No root-to-leaf paths found in hierarchy_tree.")

        # Cache as CPU tensor for portability; move to device when used.
        self._all_paths = torch.tensor(paths, dtype=torch.long, device="cpu")
        return self._all_paths
    
    def compute_loss(self, raw_logits_by_level, hierarchy_target):
        first_logits = raw_logits_by_level[0]
        batch_size = first_logits.shape[0]
        device = first_logits.device

        if hierarchy_target.ndim == 1:
            hierarchy_target = hierarchy_target.unsqueeze(0)

        hierarchy_target = hierarchy_target.to(device=device, dtype=torch.long)

        if hierarchy_target.shape[0] != batch_size:
            raise ValueError("hierarchy_target batch size must match logits batch size")

        if hierarchy_target.shape[1] != self.depth:
            raise ValueError(
                f"hierarchy_target must contain one class index per hierarchy level. "
                f"Expected {self.depth}, got {hierarchy_target.shape[1]}."
            )

        masked_logits_by_level = {}
        log_probs_by_level = {}
        probs_by_level = {}
        masks_by_level = {}

        losses = []
        selected_log_probs = []

        for level in range(self.depth):
            parent_ids = None if level == 0 else hierarchy_target[:, level - 1]
            mask = self._get_mask(level, parent_ids, batch_size).to(device)

            target_l = hierarchy_target[:, level]

            target_valid = mask.gather(1, target_l.unsqueeze(1)).squeeze(1)
            if not target_valid.all():
                invalid_indices = torch.where(~target_valid)[0]
                raise ValueError(
                    f"Invalid target at level {level}. "
                    f"Invalid batch indices: {invalid_indices.tolist()}"
                )

            raw_logits = raw_logits_by_level[level]

            masked_logits, log_probs = self._masked_log_softmax(raw_logits, mask)
            probs = log_probs.exp()

            loss_l = F.cross_entropy(masked_logits, target_l)
            losses.append(loss_l)

            selected_logp = log_probs.gather(1,target_l.unsqueeze(1)).squeeze(1)
            selected_log_probs.append(selected_logp)

            masked_logits_by_level[level] = masked_logits
            log_probs_by_level[level] = log_probs
            probs_by_level[level] = probs
            masks_by_level[level] = mask

        total_loss = sum(losses)
        path_log_prob = torch.stack(selected_log_probs, dim=1).sum(dim=1)

        return {
            "loss": total_loss,
            "losses": losses,
            "masked_logits": masked_logits_by_level,
            "log_probs": log_probs_by_level,
            "probs": probs_by_level,
            "masks": masks_by_level,
            "path_log_prob": path_log_prob,
            "path_prob": path_log_prob.exp(),
            "raw_logits": raw_logits_by_level,
        }
    
    def forward(self, x):
        h = self.extract_features(x)

        raw_logits_by_level = {}

        # compute raw logits for each level (heads only depend on h)
        for level in range(self.depth):
            raw = self.heads[str(level)](h)
            raw_logits_by_level[level] = raw
        
        return raw_logits_by_level

    def training_step(self, x, hierarchy_target):
        # forward() returns raw logits by level
        raw_logits_by_level = self.forward(x)

        # compute_loss validates targets and returns masked logits/log-probs/losses
        loss_outputs = self.compute_loss(raw_logits_by_level, hierarchy_target)

        # normalize keys to match expected API: use "logits" for masked logits
        masked_logits = loss_outputs.pop("masked_logits")
        loss_outputs["masked_logits"] = masked_logits
        loss_outputs["raw_logits"] = raw_logits_by_level

        return loss_outputs
        
    @torch.no_grad()
    def predict(self, x):
        raw_logits_by_level = self.forward(x)

        batch_size = x.shape[0]
        device = x.device

        all_paths = self._enumerate_paths().to(device)
        num_paths = all_paths.shape[0]

        path_scores = torch.empty(batch_size, num_paths, device=device)

        for p_idx in range(num_paths):
            path = all_paths[p_idx]
            score = torch.zeros(batch_size, device=device)

            for level in range(self.depth):
                parent_id = None if level == 0 else int(path[level - 1].item())
                child_id = int(path[level].item())

                mask = self._get_mask(level, parent_id, batch_size).to(device)
                raw_logits = raw_logits_by_level[level]

                _, log_probs = self._masked_log_softmax(raw_logits, mask)

                score = score + log_probs[:, child_id]

            path_scores[:, p_idx] = score

        best_idx = torch.argmax(path_scores, dim=1)
        best_paths = all_paths[best_idx].to(device)
        best_scores = path_scores[torch.arange(batch_size, device=device), best_idx]

        return {
            "candidate_paths": all_paths.cpu().tolist(),
            "path_scores": path_scores,
            "path": best_paths,
            "path_log_prob": best_scores,
            "path_prob": best_scores.exp(),
            "raw_logits": raw_logits_by_level,
        }


