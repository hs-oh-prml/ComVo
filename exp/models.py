from typing import Optional

import torch
from torch import nn
from models.cvnn import *


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class PhaseQuantizationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, levels):
        r = torch.abs(z)
        theta = torch.angle(z)

        # Phase quantization
        step = 2 * torch.pi / levels
        quant_theta = torch.round(theta / step) * step

        # Save for backward
        ctx.save_for_backward(theta, quant_theta)

        # Reconstruct
        z_quant = r * torch.exp(1j * quant_theta)
        return z_quant

    @staticmethod
    def backward(ctx, grad_output):
        z, z_quant = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None


class PhaseQuantizationLayer(torch.nn.Module):

    def __init__(self, levels=16):
        super().__init__()
        self.levels = levels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return PhaseQuantizationFunction.apply(z, self.levels)


class ConvNeXtBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = cConv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = cLayerNorm(dim)
        self.pwconv1 = cLinear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = cGelu()
        self.pwconv2 = cLinear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ComVo(Backbone):

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        n_quantization=0,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = cConv1d(input_channels, dim, kernel_size=7, padding=3)

        self.adanorm = adanorm_num_embeddings is not None
        self.norm = cLayerNorm(dim)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = cLayerNorm(dim)
        self.apply(self._init_weights)
        self.n_quantization = n_quantization
        if n_quantization != 0:
            self.q_pha = PhaseQuantizationLayer(n_quantization)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get("bandwidth_id", None)
        x = self.embed(x)
        if self.n_quantization != 0:
            x = self.q_pha(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x
