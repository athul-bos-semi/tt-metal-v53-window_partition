# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import models.experimental.bloom_old.bloom_utils as bloom_utils


def merge_heads(x: torch.Tensor, num_heads, hidden_size, num_attention_heads) -> torch.Tensor:
    head_dim = hidden_size // num_attention_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    x = x.view(batch_size, num_heads, seq_length, head_dim)
    x = x.permute(0, 2, 1, 3)

    return x.reshape(1, batch_size, seq_length, num_heads * head_dim)


def tt_merge_heads(x: torch.Tensor, num_heads, hidden_size, num_attention_heads, device) -> torch.Tensor:
    head_dim = hidden_size // num_attention_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    tt_x = bloom_utils.torch2tt_tensor(x, device)
    tt_reshaped = ttnn.reshape_on_device(tt_x, batch_size, num_heads, seq_length, head_dim)

    p_reshaped = bloom_utils.tt2torch_tensor(tt_reshaped)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    p_permuted = p_reshaped.permute(0, 2, 1, 3)
    tt_permuted = bloom_utils.torch2tt_tensor(p_permuted, device)
    reshaped_2 = ttnn.reshape_on_device(tt_permuted, 1, batch_size, seq_length, num_heads * head_dim)

    return reshaped_2
