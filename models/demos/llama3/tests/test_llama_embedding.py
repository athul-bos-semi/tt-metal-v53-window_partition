# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    skip_for_batch_parallelism,
    skip_for_parallelism,
    skip_for_model_parallelism,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3.tt.llama_common import HostEmbedding


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
    "batch_dp_tp",
    [(1, 1, 2), (2, 2, 1), (4, 2, 1)],
    ids=lambda args: "batch_{}_dp_{}_tp_{}".format(*args),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_llama_embedding(max_seq_len, batch_dp_tp, mesh_device, use_program_cache, reset_seeds, ensure_gc):
    batch_size, data_parallel, tensor_parallel = batch_dp_tp

    skip, reason = skip_for_batch_parallelism(batch_size, data_parallel)
    if skip:
        pytest.skip(reason)
    skip, reason = skip_for_parallelism(
        mesh_device.get_num_devices() if mesh_device else 0, data_parallel, tensor_parallel
    )
    if skip:
        pytest.skip(reason)
    skip, reason = skip_for_model_parallelism(data_parallel)
    if skip:
        pytest.skip(reason)

    dtype = ttnn.bfloat16
    mesh_device.enable_async(True)

    if data_parallel > 1:
        mesh_device.reshape(ttnn.MeshShape(mesh_device.get_num_devices(), 1))

    model_args = TtModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        data_parallel=data_parallel,
        tensor_parallel=tensor_parallel,
        max_seq_len=max_seq_len,
    )
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    reference_emb = HostEmbedding(model_args)
    if model_args.is_vision():
        layer_name = "text_model.tok_embeddings.weight"
    else:
        layer_name = "tok_embeddings.weight"
    reference_emb.load_state_dict({"emb.weight": state_dict[layer_name]})

    tt_emb = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=dtype,
    )

    prompts = ["Joy"] * batch_size  # 32
    pt_input = torch.tensor([tokenizer.encode(prompt, bos=False, eos=False) for prompt in prompts])
    reference_output = reference_emb(pt_input)
    logger.info(f"reference_output: {reference_output.shape}")

    tt_input = ttnn.from_torch(
        pt_input.squeeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, None) if model_args.num_devices_dp > 1 else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_output = tt_emb(tt_input)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=model_args.cluster_shape),
    )[:32].view(reference_output.shape)
    logger.info(f"tt_output_torch: {tt_output_torch.shape}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llama_embedding Passed!")
    else:
        logger.warning("Llama_embedding Failed!")

    assert passing, f"Llama_embedding output does not meet PCC requirement {0.99}."
