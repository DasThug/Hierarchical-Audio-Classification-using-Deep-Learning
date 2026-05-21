import torch

def MaskedHierarchicalVGG16_LossWrapper(hierarchy_tree):

    def loss_fn(raw_logits_by_level:dict, hierarchy_target):
        first_logits = raw_logits_by_level[0]
        batch_size = first_logits.shape[0]
        device = first_logits.device

        if hierarchy_target.ndim == 1:
            hierarchy_target = hierarchy_target.unsqueeze(0)
        
        hierarchy_target = hierarchy_target.to(device=device, dtype=torch.long)

        if hierarchy_target.shape[0] != batch_size:
            raise ValueError("hierarchy_target batch size must match logits batch size")
        if hierarchy_target.shape[1] != hierarchy_tree.total_depth:
            raise ValueError(
                f"hierarchy_target must contain one class index per hierarchy level. "
                f"Expected {hierarchy_tree.total_depth}, got {hierarchy_target.shape[1]}."
            )

        masked_logits_by_level = {}
        log_probs_by_level = {}
        probs_by_level = {}
        masks_by_level = {}
    
        