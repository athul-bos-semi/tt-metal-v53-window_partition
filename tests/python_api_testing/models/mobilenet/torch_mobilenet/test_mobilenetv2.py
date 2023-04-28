
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torchvision import models
# from transformers import AutoImageProcessor, MobileNetV2Model
import transformers

from loguru import logger
import pytest

from libs import tt_lib as ttl
from utility_functions import comp_allclose_and_pcc, comp_pcc

from mobilenetv2 import MobileNetV2Model as TtMobileNetv2Model

_batch_size = 1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', "Ops Fused"])
def test_mobilenetv2_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    with torch.no_grad():

        image_processor = transformers.AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
        torch_model = transformers.MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")

        torch_model.eval()

        state_dict = torch_model.state_dict()
        tt_model = TtMobileNetv2Model(config=torch_model.config, state_dict=state_dict)
        tt_model.eval()


        if fuse_ops:
            modules_to_fuse = [["conv_stem.first_conv.convolution", "conv_stem.first_conv.normalization"]]
            modules_to_fuse.extend([["conv_stem.conv_3x3.convolution", "conv_stem.conv_3x3.normalization"]])
            modules_to_fuse.extend([["conv_stem.reduce_1x1.convolution", "conv_stem.reduce_1x1.normalization"]])

            for i in range(16):
                modules_to_fuse.extend([[f"layer.{i}.expand_1x1.convolution", f"layer.{i}.expand_1x1.normalization"]])
                modules_to_fuse.extend([[f"layer.{i}.conv_3x3.convolution", f"layer.{i}.conv_3x3.normalization"]])
                modules_to_fuse.extend([[f"layer.{i}.reduce_1x1.convolution", f"layer.{i}.reduce_1x1.normalization"]])

            modules_to_fuse.extend([[f"conv_1x1.convolution", f"conv_1x1.normalization"]])

            tt_model = torch.ao.quantization.fuse_modules(tt_model, modules_to_fuse)

        torch_output = torch_model(image).last_hidden_state

        tt_output = tt_model(image)[0]

        passing = comp_pcc(torch_output, tt_output)
        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")
