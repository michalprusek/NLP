"""
Patched utils.py without torchtext dependency.
Only includes functions needed for TextFlow model.
"""

import torch
from torch import nn
import numpy as np


def make_pos_cond(T, B, lengths, max_T):
    """Create positional conditioning for the model."""
    device = lengths.device

    p_plus_int = torch.arange(T, device=device)[:, None].repeat(1, B)[:, :, None]
    p_plus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_plus_oh.scatter_(2, p_plus_int, 1)

    p_minus_int = lengths[None, :] - 1 - torch.arange(T, device=device)[:, None]
    p_minus_int[p_minus_int < 0] = max_T - 1
    p_minus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_minus_oh.scatter_(2, p_minus_int[:, :, None], 1)

    pos_cond = torch.cat((p_plus_oh, p_minus_oh), -1)  # [T, B, max_T*2]

    return pos_cond


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverse padded sequences."""
    if batch_first:
        inputs = inputs.transpose(0, 1)

    if inputs.size(1) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')

    reversed_inputs = inputs.data.clone()
    for i, length in enumerate(lengths):
        time_ind = torch.LongTensor(list(reversed(range(length))))
        reversed_inputs[:length, i] = inputs[:, i][time_ind]

    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)

    return reversed_inputs
