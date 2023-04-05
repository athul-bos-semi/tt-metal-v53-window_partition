from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from libs import tt_lib as ttm
from loguru import logger

from transformers import T5Model
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.t5.t5_utils import tt2torch_tensor, torch2tt_tensor, read_model_config
from python_api_testing.models.t5.t5_layer_norm import TtT5LayerNorm


def run_test_T5LayerNorm_inference(device):
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    model_json_config = "tests/python_api_testing/models/t5/t5-small.json"
    config = read_model_config(model_json_config)

    # Module to test
    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[1].layer_norm
        base_address = f"decoder.block.0.layer.1.layer_norm"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1].layer_norm
        base_address = f"encoder.block.0.layer.1.layer_norm"

    # Prepare input
    torch.manual_seed(0)
    t5_layer_norm_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(t5_layer_norm_input)[0].unsqueeze(1)
    tt_T5LayerNorm_model = TtT5LayerNorm(config, hf_reference_model.state_dict(), base_address, device)

    # TT hardware execution
    tt_layer_norm_input = torch2tt_tensor(t5_layer_norm_input, device)

    tt_out = tt_T5LayerNorm_model(tt_layer_norm_input)
    tt_out = tt2torch_tensor(tt_out)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5LayerNorm_inference Passed!")
    else:
        logger.warning("test_T5LayerNorm_inference Failed!")


def test_T5LayerNorm_inference():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_T5LayerNorm_inference(device)
    ttm.device.CloseDevice(device)
