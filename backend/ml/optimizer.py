import torch

def redistribute_attention(att_weights: tuple, sink_mask, classifications):
    
    classifications_list = [x == "detrimental" for x in classifications]

    classifications_mask = torch.tensor(classifications_list)

    detrimental_mask = torch.logical_and(sink_mask, classifications_mask)

    att_weights_copy = torch.stack(att_weights).clone()

    att_weights_copy[:, :, :, :, detrimental_mask] = 0

    denom = att_weights_copy.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    
    normalized_att_weights_copy = att_weights_copy / denom

    return normalized_att_weights_copy