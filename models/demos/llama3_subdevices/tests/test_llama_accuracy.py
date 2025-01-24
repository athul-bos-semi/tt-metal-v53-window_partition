# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import bz2
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_common import (
    get_prefill_rot_mat,
    HostEmbedding,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.demos.llama3_subdevices.demo.demo import preprocess_inputs_prefill
from pathlib import Path


def get_accuracy_thresholds(model_name: str, device_name: str, optimizations: LlamaOptimizations):
    """Parse accuracy thresholds from PERF.md for the given model, optimization mode, and device."""
    # Get model size (e.g., "1b", "3b", etc.)
    model_size = model_name.split("-")[1].lower()

    # Read PERF.md
    perf_file = Path(__file__).parent.parent / "PERF.md"
    with open(perf_file, "r") as f:
        content = f.read()

    # Split into sections based on optimization mode
    sections = content.split("## ")
    target_section = next(s for s in sections if s.startswith(f"LlamaOptimizations.{optimizations.__name__}\n"))

    # Parse the table and find the row for our model and device
    rows = [
        line.split("|")[1:]  # Each row starts with a separator
        for line in target_section.replace(" ", "").split("\n")
        if f"|{model_size}|{device_name}|" in line
    ]
    if not rows:
        raise ValueError(
            f"Could not find accuracy data for {model_size} on {device_name} in {optimizations.__name__} mode"
        )

    assert (
        len(rows) == 1
    ), f"Found multiple rows for {model_size} on {device_name} in {optimizations.__name__} mode in PERF.md"
    row = rows[0]
    top1_acc = float(row[2].strip())
    top5_acc = float(row[3].strip())

    # Allow for rounding
    return top1_acc - 0.5, top5_acc - 0.5


@torch.no_grad()
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "prefill_len, decode_len, max_seq_len",  # Max seqlen should be at least prefill_len + decode_len
    ((0, 128, 1024),),  # Prefill not yet supported with Prefetcher/on sub cores
)
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
    "optimizations",
    [
        # pytest.param(LlamaOptimizations.accuracy, id="accuracy"),
        pytest.param(LlamaOptimizations.performance, id="performance"),
    ],
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        # True,
        False,
    ),
    ids=(
        # "paged_attention",
        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "use_reference_file",
    [
        pytest.param(True, id="reference_file"),
        # pytest.param(False, id="reference_text"),
    ],
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_tt_model_acc(
    prefill_len,
    decode_len,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    use_reference_file,
    use_program_cache,
    reset_seeds,
    ensure_gc,
    is_ci_env,
):
    if is_ci_env and not use_reference_file:
        pytest.skip("CI test only runs vs reference file")

    dtype = ttnn.bfloat8_b

    mesh_device.enable_async(False)

    # Load model args and tokenizer
    model_args = TtModelArgs(
        mesh_device, optimizations=optimizations, max_batch_size=batch_size, max_seq_len=max_seq_len
    )

    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Load state_dict for TT model
    logger.info("Loading weights...")
    state_dict = model_args.load_state_dict()
    logger.info("Finished loading weights...")

    # Load the reference data
    model_size = model_args.model_name.split("-")[1].lower()  # e.g., "1b", "3b", "8b", "70b"

    if use_reference_file:
        # Existing reference file loading logic
        reference_data_file = f"models/demos/llama3/tests/reference_outputs/{model_size}.refpt"
        logger.info(f"Loading reference data from {reference_data_file}")
        assert os.path.exists(reference_data_file)
        reference_data = torch.load(reference_data_file)
        reference_tokens = reference_data["reference_tokens"]
        top5_tokens = reference_data["top5_tokens"]
    else:
        # Load and encode the reference text
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(current_file_path, "tale-of-two-cities.txt.bz2")
        with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
            text = f.read()

        # Encode text to tokens
        encoded_tokens = tokenizer.encode(text, bos=True, eos=False)
        total_length = prefill_len + decode_len + 1
        reference_tokens = torch.tensor(encoded_tokens[:total_length]).unsqueeze(0)
        top5_tokens = None  # Will be computed during inference

    N = prefill_len + decode_len
    input_ids = reference_tokens[:, : N + 1]  # Shape [1, N+1]

    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Initialize TT model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    # Initialize embedding
    embd = HostEmbedding(model_args)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # Skip prefill if prefill_len is 0
    if prefill_len > 0:
        logger.info(f"Starting prefill...")
        batch_id = 0
        input_prompts = [tokenizer.decode(reference_tokens[0, :prefill_len].tolist())]
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            model_args,
            instruct=False,
            max_generated_tokens=decode_len,
            max_prefill_len=prefill_len,
        )
        pt_prefill_input = [embd(input_tokens_prefill_pt[b]).view(1, prefill_lens[b], -1) for b in range(1)]

        # Pre-compute the rotational embedding matrix and send to device
        rot_mats_prefill = get_prefill_rot_mat(
            model_args.head_dim,
            model_args.max_seq_len,
            mesh_device,
            seq_len=prefill_lens[0],
            scale_factor=model_args.rope_scaling_factor,
        )

        prefill_input = model_args.prepare_residual_tensor_prefill(
            pt_prefill_input[batch_id],
        )

        tt_out = tt_model(
            prefill_input,
            current_pos=None,
            rot_mats=rot_mats_prefill,
            user_id=batch_id,
            mode="prefill",
            page_table=page_table_tt,
            get_last_token=((decoding_pos[batch_id] - 1) // 32) * 32,
        )

    # Start decoding
    logger.info(f"Starting decode...")
    generation_start_pos = prefill_len
    generation_length = decode_len

    # Initial positions
    decoding_pos = [generation_start_pos] * model_args.max_batch_size
    current_pos = torch.tensor([decoding_pos[b] for b in range(model_args.max_batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    # Get cos/sin matrices for the current position of each user
    rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

    # Print table header
    if use_reference_file:
        logger.info(f"{'Progress':<15}{'Correct':<8}{'True':<15}{'Actual':<15}{'Top 5 Predictions':<75}")
    else:
        logger.info(f"{'Progress':<15}{'Correct':<8}{'True':<15}{'Top 5 Predictions':<75}")
    logger.info("-" * 113)

    top1_correct = []
    top5_correct = []
    errors = []  # New list to store error information

    for i in range(generation_length):
        # Input is reference token at each step
        ref_token = input_ids[0, prefill_len + i].item()
        # Get the true next token (if available)
        true_token = input_ids[0, prefill_len + i + 1].item() if i < generation_length - 1 else None
        # Convert to torch tensor
        ref_token = torch.tensor([[ref_token]], dtype=torch.int32)  # Shape [1,1]
        # Get embedding
        pt_decode_input = embd(ref_token).view(1, 1, -1)
        # Prepare input for TT model
        decode_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # if tt_model.args.num_devices > 1:
        #     if tt_model.args.is_galaxy:
        #         tt_out_gathered = ttnn.all_gather(
        #             tt_out,
        #             dim=3,
        #             num_links=tt_model.args.num_all_gather_links,
        #             cluster_axis=0,
        #             mesh_device=mesh_device,
        #             topology=tt_model.args.ccl_topology(),
        #         )
        #     else:
        #         tt_out_gathered = ttnn.all_gather(tt_out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        #     ttnn.deallocate(tt_out)
        # else:
        #     tt_out_gathered = tt_out
        # tt_out_rm = ttnn.untilize(tt_out_gathered, use_multicore=True)
        # ttnn.deallocate(tt_out_gathered)
        # tt_out_tok = ttnn.argmax(
        #     tt_out_rm,
        #     dim=3,
        #     use_multicore=True if model_args.max_batch_size == 1 else False,
        # )
        # if not use_reference_file:
        #     tt_logits = ttnn.to_torch(tt_out_rm, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[0, 0, 0, :]
        # ttnn.deallocate(tt_out_rm)

        tt_logits = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[0, 0, 0, :]
        ttnn.deallocate(tt_out)

        tt_argmax_token = torch.argmax(tt_logits, dim=-1)

        # tt_argmax_token = ttnn.to_torch(tt_out_tok, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
        #     0, 0, 0, 0
        # ]

        ttnn.plus_one(current_pos_tensor)

        # Update rot_mats for next iteration
        current_pos += 1
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

        # Modify the accuracy checking section when using reference text
        if not use_reference_file:
            # Get probabilities from model output
            probs = torch.softmax(tt_logits, dim=-1)
            _, tt_top5_tokens = torch.topk(probs, k=5, dim=-1)

            # Check against actual next token
            true_token = input_ids[0, prefill_len + i + 1].item()
            top1_match = tt_argmax_token.item() == true_token
            top5_match = true_token in tt_top5_tokens
            ref_top5_text = [tokenizer.decode([t]) for t in tt_top5_tokens]
        else:
            # Existing logic for reference file comparison
            ref_top5_tokens = top5_tokens[prefill_len + i]
            top1_match = tt_argmax_token.item() == ref_top5_tokens[0].item()
            top5_match = tt_argmax_token in ref_top5_tokens
            ref_top5_text = [tokenizer.decode([t]) for t in ref_top5_tokens]

        # Check top-1 and top-5 accuracy
        top1_correct.append(top1_match)
        top5_correct.append(top5_match)
        true_match = (
            tt_argmax_token.item() == input_ids[0, prefill_len + i + 1].item() if i < generation_length - 1 else False
        )

        # Store error information vs reference model if top5 is incorrect
        if use_reference_file and not top5_match:
            context_start = max(0, prefill_len + i - 9)
            context_tokens = input_ids[0, context_start : prefill_len + i + 1]
            context_text = tokenizer.decode(context_tokens.tolist())
            incorrect_token = tokenizer.decode([tt_argmax_token])
            expected_tokens = [tokenizer.decode([t]) for t in ref_top5_tokens]
            errors.append(
                {
                    "position": prefill_len + i,
                    "context": context_text,
                    "incorrect": incorrect_token,
                    "expected": expected_tokens,
                    "predicted_id": tt_argmax_token.item(),
                    "expected_ids": ref_top5_tokens.tolist(),
                }
            )

        sanitize = lambda x: repr(x)[1:-1]  # Use repr() and remove the outer quotes

        # Decode tokens to text
        tt_argmax_text = tokenizer.decode([tt_argmax_token])
        true_text = tokenizer.decode([true_token]) if true_token is not None else "N/A"

        # Prepare table row
        progress_str = f"{i+1}/{generation_length}"
        correct = "x" if top1_match else ("-" if top5_match else ("!" if true_match else " "))
        tt_argmax_text = sanitize(tt_argmax_text)
        true_text = sanitize(true_text)
        ref_top5_str = " ".join(f"{sanitize(t):<14}" for t in ref_top5_text)

        # Print table row
        if use_reference_file:
            logger.info(f"{progress_str:<15}{correct:<8}{true_text:<15}{tt_argmax_text:<15}{ref_top5_str}")
        else:
            logger.info(f"{progress_str:<15}{correct:<8}{true_text:<15}{ref_top5_str}")

    # Compute accuracies over every 100 tokens
    num_tokens = len(top1_correct)
    num_segments = (num_tokens + 99) // 100
    for seg in range(num_segments):
        start = seg * 100
        end = min(start + 100, num_tokens)
        top1_acc = 100 * sum(top1_correct[start:end]) / (end - start)
        top5_acc = 100 * sum(top5_correct[start:end]) / (end - start)
        max_width = len(str(decode_len))
        logger.info(
            f"Tokens {start:{max_width}d}-{end:{max_width}d}: Top-1 accuracy: {top1_acc:3.0f} %, Top-5 accuracy: {top5_acc:3.0f} %"
        )

    # Report total accuracies
    total_top1_acc = 100 * sum(top1_correct) / num_tokens
    total_top5_acc = 100 * sum(top5_correct) / num_tokens
    logger.info(
        f"Total tokens {num_tokens}: Top-1 accuracy: {total_top1_acc:3.0f} %, Top-5 accuracy: {total_top5_acc:3.0f} %"
    )

    # Only show error summary when using reference files
    if use_reference_file:
        logger.info("\nError Summary (only showing errors where reference top-1 matches true token):")
        logger.info("-" * 120)
        for error in errors:
            true_token = input_ids[0, error["position"] + 1].item()
            if error["expected_ids"][0] == true_token:
                sanitize = lambda x: repr(x)[1:-1]  # Use repr() and remove the outer quotes
                context = sanitize(error["context"])
                incorrect = sanitize(error["incorrect"])
                expected = " | ".join(sanitize(t) for t in error["expected"])
                true_word = sanitize(tokenizer.decode([true_token]))
                logger.info(f"{error['position']}: {context}[{incorrect}] != [{expected}], true: [{true_word}]")

    # Get accuracy thresholds from PERF.md
    min_top1_acc, min_top5_acc = get_accuracy_thresholds(
        model_args.model_name,
        model_args.device_name,
        optimizations,
    )

    logger.info(f"Top-1: {total_top1_acc:.0f}% | Top-5: {total_top5_acc:.0f}%")
    assert (
        total_top1_acc >= min_top1_acc
    ), f"Top-1 accuracy {total_top1_acc:.1f}% is too low (expected >={min_top1_acc}%)"
    assert (
        total_top5_acc >= min_top5_acc
    ), f"Top-5 accuracy {total_top5_acc:.1f}% is too low (expected >={min_top5_acc}%)"
