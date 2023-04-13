
import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")

from tqdm import tqdm

import torch
import torchvision
from torchvision import models
from torchvision import transforms

from BasicBlock import BasicBlock
from libs import tt_lib as ttl
from common import ImageNet
from resnet import _resnet

from utility_functions import comp_allclose_and_pcc, comp_pcc
batch_size=1

def test_run_basicblock_inference():
    # inputs
    layer1_input = (1, 64, 64, 64)
    layer2_input = (1, 64, 64, 64)
    layer3_input = (1, 128, 32, 32)
    layer4_input = (1, 256, 16, 16)
    input_shape = layer1_input

    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        torch_resnet.eval()

        state_dict = torch_resnet.state_dict()
        tt_resnet18 = _resnet(BasicBlock, [2, 2, 2, 2], state_dict, device=device, host=host)

        torch_model1 = torch_resnet.layer1[0].conv1
        tt_model1 = tt_resnet18.layer1[0].conv1

        torch_model2 = torch_resnet.layer1[0].bn1
        tt_model2 = tt_resnet18.layer1[0].bn1

        # print(torch_model)
        # print(tt_model)
        input = torch.randn(input_shape)

        torch_output = torch_model1(input)
        tt_output = tt_model1(input)
        print(comp_allclose_and_pcc(torch_output, tt_output))


        tt_output = tt_model2(tt_output)
        torch_output = torch_model2(torch_output)

        print(comp_allclose_and_pcc(torch_output, tt_output))
        print(comp_allclose_and_pcc(torch_model2.weight, tt_model2.weight))
        print(comp_allclose_and_pcc(torch_model2.bias, tt_model2.bias))
        print(comp_allclose_and_pcc(torch_model2.running_mean, tt_model2.running_mean))
        print(comp_allclose_and_pcc(torch_model2.running_var, tt_model2.running_var))
        print(comp_allclose_and_pcc(torch_model2.num_batches_tracked, tt_model2.num_batches_tracked))
        print(tt_model2, torch_model2)


test_run_basicblock_inference()
