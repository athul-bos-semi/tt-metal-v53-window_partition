import os
import torch
import pytest
import ttnn
from loguru import logger
from models.experimental.mochi.mod_rmsnorm import modulated_rmsnorm
from models.utility_functions import skip_for_grayskull
from models.experimental.mochi.common import compute_metrics
from genmo.mochi_preview.dit.joint_model.mod_rmsnorm import modulated_rmsnorm as ref_modulated_rmsnorm


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "S, D",
    [
        (44520, 3072),
        (118, 1536),
    ],
)
def test_modulated_rmsnorm(mesh_device, use_program_cache, reset_seeds, S, D):
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)

    # Create random input and scale tensors
    torch_input = torch.randn(1, 1, S, D)
    torch_scale = torch.randn(1, 1, 1, D) * 0.1  # Small scale modulation

    # Convert to TT tensors
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_scale = ttnn.from_torch(
        torch_scale,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TT implementation
    tt_output = modulated_rmsnorm(tt_input, tt_scale)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    # Reference implementation in PyTorch
    def reference_modulated_rmsnorm(x, scale, eps=1e-6):
        weight = 1.0 + scale
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps) * weight
        return x

    reference_output = reference_modulated_rmsnorm(torch_input, torch_scale)

    # Compute metrics
    pcc, mse, mae = compute_metrics(reference_output, tt_output_torch)
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    # Check if model meets requirements
    pcc_required = 0.99
    passing = pcc >= pcc_required

    if passing:
        logger.info("Modulated RMSNorm Passed!")
    else:
        logger.warning("Modulated RMSNorm Failed!")

    assert (
        passing
    ), f"Modulated RMSNorm output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."
