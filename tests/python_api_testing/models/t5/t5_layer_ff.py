import torch
from libs import tt_lib as ttm

from python_api_testing.models.t5.t5_layer_norm import TtT5LayerNorm
from python_api_testing.models.t5.t5_dense_act_dense import TtT5DenseActDense
from python_api_testing.models.t5.t5_dense_gated_act_dense import TtT5DenseGatedActDense


# class T5LayerFF(nn.Module):
#    def __init__(self, config: T5Config):
#        super().__init__()
#        if config.is_gated_act:
#            self.DenseReluDense = T5DenseGatedActDense(config)
#        else:
#            self.DenseReluDense = T5DenseActDense(config)

#        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
#        self.dropout = nn.Dropout(config.dropout_rate)

#    def forward(self, hidden_states):
#        forwarded_states = self.layer_norm(hidden_states)
#        forwarded_states = self.DenseReluDense(forwarded_states)
#        hidden_states = hidden_states + self.dropout(forwarded_states)
#        return hidden_states


class TtT5LayerFF(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        if "is_gated_act" in config and config["is_gated_act"]:
            self.DenseReluDense = TtT5DenseGatedActDense(config, state_dict, f"{base_address}.DenseReluDense", device)
        else:
            self.DenseReluDense = TtT5DenseActDense(config, state_dict, f"{base_address}.DenseReluDense", device)

        self.layer_norm = TtT5LayerNorm(config, state_dict, f"{base_address}.layer_norm", device)
        self.dropout = torch.nn.Dropout(config["dropout_rate"])

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        #hidden_states = hidden_states + self.dropout(forwarded_states)
        hidden_states = ttm.tensor.add(hidden_states, forwarded_states)
        return hidden_states
