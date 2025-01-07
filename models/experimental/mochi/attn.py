import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
import torch
from typing import Tuple, Optional

from models.experimental.mochi.mod_rmsnorm import modulated_rmsnorm
from models.experimental.mochi.common import (
    matmul_config,
    as_sharded_tensor,
    col_parallel_linear,
)


class TtAsymmetricAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        layer_num,
        dtype,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        update_y: bool = True,
        out_bias: bool = True,
        attention_mode: str = "flash",
        softmax_scale=None,
        # Disable LoRA by default ...
        qkv_proj_lora_rank: int = 0,
        qkv_proj_lora_alpha: int = 0,
        qkv_proj_lora_dropout: float = 0.0,
        out_proj_lora_rank: int = 0,
        out_proj_lora_alpha: int = 0,
        out_proj_lora_dropout: float = 0.0,
    ):
        super().__init__()
        # Assert that all LoRA ranks are 0 since we don't support LoRA in this implementation
        assert qkv_proj_lora_rank == 0, "LoRA not supported - qkv_proj_lora_rank must be 0"
        assert qkv_proj_lora_alpha == 0, "LoRA not supported - qkv_proj_lora_alpha must be 0"
        assert qkv_proj_lora_dropout == 0.0, "LoRA not supported - qkv_proj_lora_dropout must be 0.0"
        assert out_proj_lora_rank == 0, "LoRA not supported - out_proj_lora_rank must be 0"
        assert out_proj_lora_alpha == 0, "LoRA not supported - out_proj_lora_alpha must be 0"
        assert out_proj_lora_dropout == 0.0, "LoRA not supported - out_proj_lora_dropout must be 0.0"

        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.state_dict = state_dict
        self.state_dict_prefix = state_dict_prefix
        self.weight_cache_path = weight_cache_path
        self.layer_num = layer_num
        self.dtype = dtype
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.n_local_heads = num_heads // self.num_devices
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(f"dim_x={dim_x} should be divisible by num_heads={num_heads}")

        # Input layers.
        self.qkv_bias = qkv_bias
        self.out_bias = out_bias

        # Define the qkv and output projections
        self.qkv_x, self.qkv_x_bias = self._col_parallel_qkv(
            "qkv_x", self.qkv_bias, weight_cache_path, state_dict, state_dict_prefix
        )
        self.qkv_y, self.qkv_y_bias = self._col_parallel_qkv(
            "qkv_y", self.qkv_bias, weight_cache_path, state_dict, state_dict_prefix
        )

        self.proj_x, self.proj_x_bias = col_parallel_linear(
            "proj_x", self.out_bias, weight_cache_path, state_dict, state_dict_prefix, self.mesh_device
        )
        if update_y:
            self.proj_y, self.proj_y_bias = col_parallel_linear(
                "proj_y", self.out_bias, weight_cache_path, state_dict, state_dict_prefix, self.mesh_device
            )

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = self._create_rmsmorn(".q_norm_x")
        self.k_norm_x = self._create_rmsmorn(".k_norm_x")
        self.q_norm_y = self._create_rmsmorn(".q_norm_y")
        self.k_norm_y = self._create_rmsmorn(".k_norm_y")

        self.X_MM_SEQ_LEN = 512

        self.qkv_x_config = matmul_config(self.X_MM_SEQ_LEN, dim_x, 3 * self.n_local_heads * self.head_dim, (8, 8))
        self.proj_x_config = matmul_config(self.X_MM_SEQ_LEN, dim_x // self.num_devices, dim_x, (8, 8))

    def _create_rmsmorn(self, key):
        return RMSNorm(
            device=self.mesh_device,
            dim=self.head_dim,
            state_dict=self.state_dict,
            state_dict_prefix=self.state_dict_prefix,
            weight_cache_path=self.weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=key,
        )

    def _col_parallel_qkv(self, name, bias, weight_cache_path, state_dict, state_dict_prefix):
        """
        Shuffle QKV weights to group heads such that they can be column parallel
        """
        torch_weight = lambda name, suffix: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.{suffix}"], -2, -1)
        w = torch_weight(name, "weight")
        b = torch_weight(name, "bias") if bias else None

        def shuffle_heads(tensor):
            # Given torch tensor with output features in the last dimension,
            # shuffle heads to allow for column parallel computation
            in_dim = tensor.shape[0]
            tensor = tensor.reshape(in_dim, 3, self.num_devices, self.n_local_heads, -1)  # [ID, 3, ND, NLH, DH]
            tensor = tensor.permute(0, 2, 1, 3, 4)  # [ID, ND, 3, NLH, DH]
            tensor = tensor.reshape(in_dim, -1)  # [ID, ND*3*NLH*DH]
            return tensor

        w = as_sharded_tensor(
            shuffle_heads(w),
            self.mesh_device,
            dim=-1,
            cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.weight"),
        )
        if b is not None:
            b = as_sharded_tensor(
                shuffle_heads(b),
                self.mesh_device,
                dim=-1,
                cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.bias"),
            )
        return w, b

    def run_qkv_y(self, y):
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Compute QKV projection
        qkv_y = ttnn.linear(
            y,
            self.qkv_y,
            bias=self.qkv_y_bias,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Use nlp_create_qkv_heads here as well because text will be concatenated on the sequence dimension
        q_y, k_y, v_y = ttnn.experimental.nlp_create_qkv_heads(
            qkv_y,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Apply normalization
        q_y = self.q_norm_y(q_y, mode="prefill")
        k_y = self.k_norm_y(k_y, mode="prefill")
        return q_y, k_y, v_y

    def prepare_qkv(
        self,
        x: ttnn.Tensor,  # (B, M, dim_x)
        y: ttnn.Tensor,  # (B, L, dim_y)
        *,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        uncond: bool = False,
    ):
        """Prepare QKV tensors for attention computation.

        Args:
            x: Visual token features
            y: Text token features
            scale_x: Modulation for visual features
            scale_y: Modulation for text features
            rope_cos: Cosine component for rotary position embedding
            rope_sin: Sine component for rotary position embedding
            trans_mat: Transformation matrix for rotary embeddings
            uncond: Whether to run unconditional attention

        Returns:
            q, k, v: Query, key and value tensors prepared for attention
        """
        B = x.shape[1]
        seq_x = x.shape[2]
        seq_y = y.shape[2]
        assert B == 1, f"Batch size must be 1, got {B}"
        assert x.shape[0] == 1, f"x dim0 must be 1, got {x.shape[0]}"
        assert (
            seq_x % self.X_MM_SEQ_LEN == 0
        ), f"Visual sequence length must be divisible by {self.X_MM_SEQ_LEN}, got {seq_x}"
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        x = modulated_rmsnorm(x, scale_x)

        # Process visual features
        x = ttnn.reshape(x, (1, seq_x // self.X_MM_SEQ_LEN, self.X_MM_SEQ_LEN, x.shape[3]))
        qkv_x = ttnn.linear(
            x,
            self.qkv_x,
            bias=self.qkv_x_bias,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.qkv_x_config,
        )
        qkv_x = ttnn.reshape(qkv_x, (1, B, seq_x, qkv_x.shape[3]))

        # Split qkv_x into q, k, v
        # Need to get these tensors to shape [B, H, L, D] for rotary embeddings
        q_x, k_x, v_x = ttnn.experimental.nlp_create_qkv_heads(
            qkv_x,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Apply normalization and rotary embeddings to visual features
        q_x = self.q_norm_x(q_x, mode="prefill")
        q_x = ttnn.experimental.rotary_embedding_llama(q_x, rope_cos, rope_sin, trans_mat)
        k_x = self.k_norm_x(k_x, mode="prefill")
        k_x = ttnn.experimental.rotary_embedding_llama(k_x, rope_cos, rope_sin, trans_mat)

        B, _, N, _ = q_x.shape

        # Process text features if present
        if B == 1:
            if not uncond:
                # Process text features
                # TODO: Removing until we support padded sequences
                # y = y[:, :text_seqlen] # Remove padding tokens
                y = modulated_rmsnorm(y, scale_y)
                q_y, k_y, v_y = self.run_qkv_y(y)

                # Concatenate visual and text features on the sequence dimension
                q = ttnn.concat([q_x, q_y], dim=2)
                k = ttnn.concat([k_x, k_y], dim=2)
                v = ttnn.concat([v_x, v_y], dim=2)

            else:
                q, k, v = q_x, k_x, v_x
            # # TODO: Pad up to necessary length
            attn_padded_len = 44 * 1024  # TODO: GENERALIZE!
            q = ttnn.pad(q, [q.shape[0], q.shape[1], attn_padded_len, q.shape[3]], [0, 0, 0, 0], value=0)
            k = ttnn.pad(k, [k.shape[0], k.shape[1], attn_padded_len, k.shape[3]], [0, 0, 0, 0], value=0)
            v = ttnn.pad(v, [v.shape[0], v.shape[1], attn_padded_len, v.shape[3]], [0, 0, 0, 0], value=0)
            print("after pad")
            print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        else:
            assert False, "Batch size > 1 not supported"

        return q, k, v

    def run_attention(
        self,
        q: ttnn.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        k: ttnn.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        v: ttnn.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        *,
        B: int,
        attn_mask,
    ) -> ttnn.Tensor:
        """Run attention computation.

        Args:
            q: Query tensor (total <= B * (N + L), num_heads, head_dim)
            k: Key tensor (total <= B * (N + L), num_heads, head_dim)
            v: Value tensor (total <= B * (N + L), num_heads, head_dim)
            B: Batch size (must be 1)

        Returns:
            out: Attention output tensor (total, local_dim)
        """
        assert B == 1, "Only batch size 1 is supported"
        assert q.shape[0] == B, "Batch size must be B"
        assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"

        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=256,  # TODO: Make this dynamic
            k_chunk_size=512,
            exp_approx_mode=False,
        )

        # Run attention
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            attn_mask=attn_mask,
        )  # (B, num_heads, seq_len, head_dim)

        # Reshape output
        out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out

    def post_attention(
        self,
        out: ttnn.Tensor,  # (total <= B * (N + L), local_dim)
        B: int,
        M: int,
        L: int,
        dtype: ttnn.bfloat16,
        uncond: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Post attention processing to split and project visual and text features.

        Args:
            out: (total <= B * (N + L), local_dim) Combined attention output
            B: Batch size (must be 1)
            M: Number of visual tokens
            L: Number of text tokens
            dtype: Data type of tensors
            valid_token_indices: Indices of valid tokens

        Returns:
            x: (B, M, dim_x) Visual token features
            y: (B, L, dim_y) Text token features
        """
        assert B == 1, "Batch size must be 1"
        assert out.shape[1] == B, "Batch size must be 1"
        N = M  # No context parallel support yet
        local_dim = self.dim_x // self.num_devices  # No head parallel support yet

        print(f"B: {B}, M: {M}, L: {L}, N: {N}")

        # Split sequence into visual and text tokens, adding back padding
        if B == 1:
            # out = ttnn.reshape(out, (B, -1, local_dim))
            if not uncond:
                print(f"in post_attention: {out.shape=}")
                # Split into visual and text features
                x = out[:, :, :N, :]  # (B, N, local_dim)
                y = out[:, :, N : N + L, :]  # (B, <=L, local_dim)
                # Pad text features if needed
                if y.shape[2] < L:
                    raise ValueError("Not supporting padded text features")
                    # y = ttnn.pad(y, (0, 0, 0, L - y.shape[1], 0, 0))  # (B, L, local_dim)
            else:
                # Empty prompt case
                x = out[:, :, :N, :]
                # TODO: see if I can remove Y from uncond case
                y = ttnn.as_tensor(
                    torch.zeros(1, B, L, local_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
        else:
            raise ValueError("Batch size > 1 not supported")

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        x = ttnn.reshape(x, (1, x.shape[2] // self.X_MM_SEQ_LEN, self.X_MM_SEQ_LEN, x.shape[3]))
        if self.num_devices > 1:
            x = ttnn.all_gather(
                x,
                dim=-1,
            )
        x = ttnn.linear(
            x,
            self.proj_x,
            bias=self.proj_x_bias,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.proj_x_config,
        )
        x = ttnn.reshape(x, (1, 1, x.shape[1] * x.shape[2], x.shape[3]))

        if self.update_y:
            if self.num_devices > 1:
                y = ttnn.all_gather(
                    y,
                    dim=-1,
                )
            y = ttnn.linear(
                y,
                self.proj_y,
                bias=self.proj_y_bias,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return x, y

    def forward(
        self,
        x: ttnn.Tensor,
        y: ttnn.Tensor,
        *,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        packed_indices,
        attn_mask,
        uncond: bool = False,
    ) -> ttnn.Tensor:
        # input is replicated
        B, L = y.shape[1], y.shape[2]
        M = x.shape[2]

        # output is head-sharded q, k, v
        q, k, v = self.prepare_qkv(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            uncond=uncond,
        )

        # output is col-sharded
        out = self.run_attention(
            q,
            k,
            v,
            B=B,
            attn_mask=attn_mask,
        )

        # output is col-sharded x, y
        x, y = self.post_attention(
            out,
            B=B,
            M=M,
            L=L,
            dtype=out.dtype,
            uncond=uncond,
        )

        return x, y
