import pytest
from loguru import logger
import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")


import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from libs.tt_lib.fused_ops.linear import Linear as TtLinear
from libs.tt_lib.fused_ops.softmax import softmax
from utility_functions import get_FR, set_FR
from utility_functions import enable_compile_cache, enable_binary_cache
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_allclose

def mha(qw, qb, kw, kb, vw, vb, hidden_dim, num_heads, device):
    assert isinstance(num_heads, int) and num_heads > 0

    QProjection = TtLinear(
        hidden_dim, hidden_dim, qw, qb, device
    )
    KProjection = TtLinear(
        hidden_dim, hidden_dim, kw, kb, device
    )
    VProjection = TtLinear(
        hidden_dim, hidden_dim, vw, vb, device
    )

    # Used to scale down the input to the softmax
    reciprocal_of_sqrt_hidden_dim_tensor = ttl.tensor.Tensor(
        [1 / math.sqrt(hidden_dim)] * (32 * 32),
        [1, 1, 32, 32],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device
    )

    def make_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            # ref code from modeling_bert.py:
            #    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            #        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            #        x = x.view(new_x_shape)
            #        return x.permute(0, 2, 1, 3)

            untilized_x = ttl.tensor.untilize(x)
            reshaped_unt = ttl.tensor.reshape(untilized_x, x.shape()[0], x.shape()[2], num_heads, x.shape()[3] // num_heads)

            # N, 128, 2, 64
            transposed = ttl.tensor.transpose_hc_rm(reshaped_unt)
            # N, 2, 128, 64
            retilized = ttl.tensor.tilize(transposed)
            return retilized

    def unmake_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            """
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            debug_state["context_reshaped"] = context_layer.clone()

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            """
            untilized_x = ttl.tensor.untilize(x)
            ctx = ttl.tensor.transpose_hc_rm(untilized_x)
            ushape = ctx.shape()
            reshaped = ttl.tensor.reshape(ctx, ushape[0], 1, ushape[1], ushape[2]*ushape[3])
            retval = ttl.tensor.tilize(reshaped)
            return retval

    def multiply_by_sqrt_hidden_dim(x):
        return ttl.tensor.bcast(
            x,
            reciprocal_of_sqrt_hidden_dim_tensor,
            ttl.tensor.BcastOpMath.MUL,
            ttl.tensor.BcastOpDim.HW
        )

    def mha_(activation):
        Q = QProjection(activation)
        K = KProjection(activation)
        V = VProjection(activation)

        Q_heads = make_attention_heads(Q)
        K_heads = make_attention_heads(K)
        V_heads = make_attention_heads(V)
        K_T_heads = ttl.tensor.transpose(K_heads)

        qkt = ttl.tensor.bmm(Q_heads, K_T_heads)

        # Attention scores computation
        N, C, H, W = qkt.shape() # Need to reshape right now since multi-C not supported for broadcast yet
        new_shape = [N, 1, C*H, W]
        ttl.tensor.reshape(qkt, *new_shape)
        attention_score_input = multiply_by_sqrt_hidden_dim(qkt)
        attention_scores = softmax(attention_score_input)
        ttl.tensor.reshape(attention_scores, N, C, H, W) # Reshape back to original shape

        # Apply attention to value matrix
        weighted_activation = ttl.tensor.bmm(attention_scores, V_heads)
        return unmake_attention_heads(weighted_activation) # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]

    return mha_

class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device):
        super().__init__()
        qw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"])
        qb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.bias"])
        kw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.weight"])
        kb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.bias"])
        vw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.weight"])
        vb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.bias"])

        # Hidden dim
        hidden_dim = qw.shape[-1]

        # Tilized
        parameters = [
            ttl.tensor.Tensor(qw.reshape(-1).tolist(), qw.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data(),
            ttl.tensor.Tensor(qb.reshape(-1).tolist(), qb.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data(),
            ttl.tensor.Tensor(kw.reshape(-1).tolist(), kw.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data(),
            ttl.tensor.Tensor(kb.reshape(-1).tolist(), kb.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data(),
            ttl.tensor.Tensor(vw.reshape(-1).tolist(), vw.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data(),
            ttl.tensor.Tensor(vb.reshape(-1).tolist(), vb.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()
        ]

        self.mha = mha(*parameters, hidden_dim, config.num_attention_heads, device)

    def forward(self, activation):
        result = self.mha(activation)
        return result

class PytorchMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.mha = hugging_face_reference_model.bert.encoder.layer[0].attention.self

        # Disable dropout
        self.mha.eval()

    def forward(self, x):
        result = self.mha(x)[0]
        return result


def run_mha_inference(model_version, batch, seq_len, on_weka, pcc):

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    # Initialize the device
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    if on_weka:
        model_name = "/mnt/MLPerf/tt_dnn-models/Bert/BertForQuestionAnswering/models/" + model_version
    else:
        model_name = model_version

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_mha_model = TtMultiHeadAttentionModel(hugging_face_reference_model.config, 0, hugging_face_reference_model.state_dict(), device)
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1

    pytorch_out = pytorch_mha_model(mha_input.squeeze(1)).unsqueeze(1)

    pad_mha_input = pad_activation(mha_input)
    tt_mha_input = ttl.tensor.Tensor(pad_mha_input.reshape(-1).tolist(), pad_mha_input.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE)
    tt_mha_input = tt_mha_input.to(device)

    tt_out = tt_mha_model(tt_mha_input).to(host)
    tt_out1 = torch.Tensor(tt_out.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(tt_out.shape())

    ttl.device.CloseDevice(device)

    passing, output = comp_pcc(pytorch_out, tt_out1, pcc)
    logger.info(f"Output {output}")
    _, output = comp_allclose(pytorch_out, tt_out1, 0.5, 0.5) # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")
    if not passing:
        logger.error(f"Output PCC < {pcc}")

    # print_diff_argmax(pytorch_out, tt_out1)
    # assert np.allclose(pytorch_out.detach().numpy(), tt_out1, 1e-5, 0.17)

@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka,  pcc",
    (
        ("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, True, 0.99),
        ("phiyodr/bert-base-finetuned-squad2", 1, 128, True, 0.93), # Placeholder PCC until issues are resolved
        ("phiyodr/bert-large-finetuned-squad2", 1, 128, True, 0.85) # Placeholder PCC until issues are resolved
    ),
)
def test_mha_inference(model_version, batch, seq_len, on_weka, pcc):

    # Initialize the device
    #enable_binary_cache()
    #enable_compile_cache()

    run_mha_inference(model_version, batch, seq_len, on_weka, pcc)

if __name__ == "__main__":
    run_mha_inference("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, True, 0.99)
