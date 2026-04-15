import torch

def detect_sinks(att_weights: tuple, threshold: float = 0.2):

    stacked_tensors = torch.stack(att_weights)

    avg_attention = stacked_tensors.mean(dim=(0,2))

    att_received = avg_attention.sum(dim=1)

    normalized = att_received / att_received.sum(dim=-1, keepdim=True)

    sink_mask = normalized > threshold

    return sink_mask.squeeze(), normalized.squeeze()
