# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
    comp_and_get_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0, skip_for_blackhole
import math
import numpy as np


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power
    # if (2**math.log2(x) == x):
    #     return x
    # return 2**(int(x).bit_length())


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(max_start_pos, s):
    if max_start_pos <= 32:
        chunk_size = 32
    elif max_start_pos <= 64:
        chunk_size = 32
    elif max_start_pos <= 128:
        chunk_size = 32
    elif max_start_pos <= 1024:
        chunk_size = 128
    else:
        chunk_size = 512
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2 ** (i + 1)) != 0:
            break
    chunk_size = min(chunk_size, 2**i)
    return chunk_size


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def run_test_sdpa_decode_single_iter_single_device(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype=ttnn.bfloat16,
    cur_pos_tensor=False,
    speculation_length=128,
    lambda_=0.2,
    sharded_in=False,
    sharded_out=False,
    start_indices=None,
    causal=True,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR, False)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_indices = [s // 2 for _ in range(b)] if start_indices is None else start_indices
    max_start_idx = max(start_indices)
    scale = d**-0.5

    k_chunk_size = get_chunk_size(max_start_idx + 1, s)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size) if causal else s

    # Test various sequence lengths
    logger.debug(f"Testing with sequence length: {max_start_idx if causal else s}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    if causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        attn_mask = torch.bernoulli(
            torch.full(
                (b, nh, 1, padded_layer_len),
                0.25,
            )
        )
        attn_mask = attn_mask * torch.finfo(torch.float32).min

    Q = fa_rand(1, b, nh, d)

    tt_Q = ttnn.as_tensor(
        Q[:, :, :nh],
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
    )
    if causal:
        if cur_pos_tensor:
            start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)
            (
                tt_back_gt,
                tt_back_spec,
                tt_back_spec_lp_distance,
                tt_back_lp_norm_x,
            ) = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                lambda_=lambda_,
                speculative_chunk_size=speculation_length,
                cur_pos_tensor=start_indices_tt,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )
        else:
            (
                tt_back_gt,
                tt_back_spec,
                tt_back_spec_lp_distance,
                tt_back_lp_norm_x,
            ) = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                lambda_=lambda_,
                speculative_chunk_size=speculation_length,
                cur_pos=start_indices,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )
    else:
        raise NotImplementedError("Non-causal not implemented")

    tt_back_gt = ttnn.to_torch(tt_back_gt)
    tt_back_gt = tt_back_gt[:, :, :nh, :]
    tt_back_spec = ttnn.to_torch(tt_back_spec)
    tt_back_spec = tt_back_spec[:, :, :nh, :]
    tt_back_spec_lp_distance = ttnn.to_torch(tt_back_spec_lp_distance)
    tt_back_lp_norm_x = ttnn.to_torch(tt_back_lp_norm_x)

    ##########################################
    #### Expected Calculation ####
    ##########################################

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    expected_gt = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expected_gt = expected_gt.squeeze(2).unsqueeze(0)

    # get speculated output
    expected_spec = torch.zeros_like(Q_slice)
    for i in range(b):
        start_idx = start_indices[i]
        padded_start_idx = nearest_n(max_start_idx + 1, n=32)
        spec_last_chunk_start = padded_start_idx - speculation_length - 32
        Q_slice_i = Q_slice[[i]]
        K_slice_i = torch.cat(
            [K_slice[[i], :, :speculation_length, :], K_slice[[i], :, spec_last_chunk_start:padded_start_idx, :]], dim=2
        )
        V_slice_i = torch.cat(
            [V_slice[[i], :, :speculation_length, :], V_slice[[i], :, spec_last_chunk_start:padded_start_idx, :]], dim=2
        )
        attn_mask_slice_i = torch.cat(
            [
                attn_mask_slice[[i], :, :, :speculation_length],
                attn_mask_slice[[i], :, :, spec_last_chunk_start:padded_start_idx],
            ],
            dim=3,
        )

        expected_spec[i] = torch.nn.functional.scaled_dot_product_attention(
            Q_slice_i, K_slice_i, V_slice_i, attn_mask_slice_i, scale=scale, is_causal=False
        )  # b, nh, 1, d
    expected_spec = expected_spec.squeeze(2).unsqueeze(0)

    # checking speculation error using expected values from torch
    lp_distance = torch.linalg.vector_norm(expected_gt - expected_spec, ord=2, dim=(-2, -1))
    lp_norm_x = torch.linalg.vector_norm(expected_gt, ord=2, dim=(-2, -1))  # Calculate the Lp norm of x
    passing = torch.all(lp_distance < lambda_ * lp_norm_x)
    logger.debug(f"gt speculation passing: {passing}")

    ##########################################
    #### Comparison ####
    ##########################################

    non_skip_indices = torch.tensor(start_indices) != -1
    out_pass, out_pcc = comp_pcc(expected_gt[:, non_skip_indices], tt_back_gt[:, non_skip_indices], min_pcc)
    logger.debug(f"gt python vs pytorch: {out_pcc}")
    assert out_pass

    out_pass, out_pcc = comp_pcc(expected_spec[:, non_skip_indices], tt_back_spec[:, non_skip_indices], min_pcc)
    logger.debug(f"spec python vs pytorch: {out_pcc}")
    assert out_pass

    out_pass, out_pcc = comp_pcc(lp_distance, tt_back_spec_lp_distance, min_pcc)
    logger.debug(f"spec lp distance python vs pytorch: {out_pcc}")
    assert out_pass

    out_pass, out_pcc = comp_pcc(lp_norm_x, tt_back_lp_norm_x, min_pcc)
    logger.debug(f"lp norm output python vs pytorch: {out_pcc}")
    assert out_pass


@skip_for_blackhole("Unsupported on BH, see #12349")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    (
        [8, 8, 1, 32768, 128, (8, 8), True, False],  # Llama2-70B
        [4, 32, 8, 8192, 128, (8, 8), True, True],  # llama 3.1 8b
    ),
)
@pytest.mark.parametrize("speculation_length", [128, 256])
def test_sdpa_decode_single_device(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype,
    single_iter,
    cur_pos_tensor,
    speculation_length,
    use_program_cache,
):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    ttnn.device.DisablePersistentKernelCache()

    if single_iter:
        run_test_sdpa_decode_single_iter_single_device(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            cur_pos_tensor,
            speculation_length,
            sharded_in=False,
            sharded_out=False,
        )
    else:
        raise NotImplementedError("Multi-iter not implemented")
