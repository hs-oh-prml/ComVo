import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import (
    gelu,
    leaky_relu,
)
import math
from torch import Tensor
from typing import Optional, Tuple

from torch.autograd import Function


def complexGelu(inp):
    return torch.complex(gelu(inp.real), gelu(inp.imag))


def complexLeakyRelu(inp):
    return torch.complex(leaky_relu(inp.real, 0.1), leaky_relu(inp.imag, 0.1))


class cGelu(nn.Module):
    @staticmethod
    def forward(inp):
        return complexGelu(inp)


class ComplexLinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        original_shape = inp.shape
        if inp.ndim > 2:
            inp_flat = inp.contiguous().view(-1, inp.shape[-1])
        else:
            inp_flat = inp

        inp_block = torch.cat([inp_flat.real, inp_flat.imag], dim=1)

        A = weight.real
        B = weight.imag
        top = torch.cat([A, -B], dim=1)
        bottom = torch.cat([B, A], dim=1)
        weight_block = torch.cat([top, bottom], dim=0)

        out_block = F.linear(inp_block, weight_block, bias=None)
        out_features = weight.shape[0]
        out_real = out_block[:, :out_features]
        out_imag = out_block[:, out_features:]
        output_flat = torch.complex(out_real, out_imag)

        if bias is not None:
            output_flat = output_flat + bias.view(1, -1)

        ctx.save_for_backward(inp_flat, weight, bias, weight_block)
        ctx.original_shape = original_shape
        return output_flat.view(*original_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        inp_flat, weight, bias, weight_block = ctx.saved_tensors
        original_shape = ctx.original_shape

        if grad_output.ndim > 2:
            grad_output_flat = grad_output.contiguous().view(-1, grad_output.shape[-1])
        else:
            grad_output_flat = grad_output

        grad_out_block = torch.cat(
            [grad_output_flat.real, grad_output_flat.imag], dim=1
        )

        grad_inp_block = grad_out_block @ weight_block
        in_features = inp_flat.shape[1]
        grad_inp_flat = torch.complex(
            grad_inp_block[:, :in_features], grad_inp_block[:, in_features:]
        )

        inp_block = torch.cat([inp_flat.real, inp_flat.imag], dim=1)
        grad_weight_block = grad_out_block.transpose(0, 1) @ inp_block
        out_features = weight.shape[0]
        in_features = weight.shape[1]

        grad_top_left = grad_weight_block[:out_features, :in_features]
        grad_top_right = grad_weight_block[:out_features, in_features:]
        grad_bottom_left = grad_weight_block[out_features:, :in_features]
        grad_bottom_right = grad_weight_block[out_features:, in_features:]
        grad_A = grad_top_left + grad_bottom_right
        grad_B = grad_bottom_left - grad_top_right
        grad_weight = torch.complex(grad_A, grad_B)
        grad_bias = (
            grad_output_flat.sum(dim=0, keepdim=True) if bias is not None else None
        )

        grad_inp = grad_inp_flat.view(*original_shape)
        return grad_inp, grad_weight, grad_bias


class cLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.cfloat)
        )

        self.bias = nn.Parameter(
            torch.zeros(1, out_features, dtype=torch.cfloat), requires_grad=bias
        )

        nn.init.trunc_normal_(self.weight.data.real, std=0.02)
        nn.init.trunc_normal_(self.weight.data.imag, std=0.02)
        if bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, inp):
        if inp.dtype != torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        return ComplexLinearFunction.apply(inp, self.weight, self.bias)


class ComplexConv1dFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, stride, padding, dilation, groups):
        B, C, L = inp.shape
        if groups == C and weight.shape[0] == C:
            inp_real = torch.view_as_real(inp)
            inp_block = inp_real.permute(0, 1, 3, 2).reshape(B, 2 * C, L)
            k = weight.shape[-1]
            weight_real = torch.view_as_real(weight)
            W_r = weight_real[..., 0]
            W_i = weight_real[..., 1]
            top = torch.cat([W_r, -W_i], dim=1)
            bottom = torch.cat([W_i, W_r], dim=1)
            weight_block = torch.stack([top, bottom], dim=1).reshape(2 * C, 2, k)
            effective_groups = groups
            ctx.branch = "depthwise"
        else:
            inp_block = torch.cat([inp.real, inp.imag], dim=1)
            k = weight.shape[-1]
            O = weight.shape[0]
            A = weight.real
            B_mat = weight.imag
            top = torch.cat([A, -B_mat], dim=1)
            bottom = torch.cat([B_mat, A], dim=1)
            weight_block = torch.cat([top, bottom], dim=0)
            effective_groups = groups
            ctx.branch = "nondepthwise"
        out_block = F.conv1d(
            inp_block,
            weight_block,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=effective_groups,
        )
        if ctx.branch == "depthwise":
            out = torch.view_as_complex(
                out_block.reshape(B, C, 2, -1).permute(0, 1, 3, 2).contiguous()
            )
        else:
            O = weight.shape[0]
            out = torch.complex(out_block[:, :O, :], out_block[:, O:, :])
        if bias is not None:
            out = out + bias.view(1, -1, 1)
        ctx.save_for_backward(inp, weight, bias, weight_block)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_shape = inp.shape
        ctx.inp_block_shape = inp_block.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias, weight_block = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        B, C, L_in = ctx.input_shape
        k = weight.shape[-1]

        if ctx.branch == "depthwise":
            grad_out_real = torch.view_as_real(grad_output)
            grad_out_block = grad_out_real.permute(0, 1, 3, 2).reshape(
                B, 2 * C, grad_output.shape[-1]
            )
            effective_groups = groups
            grad_inp_block = F.grad.conv1d_input(
                tuple(ctx.inp_block_shape),
                weight_block,
                grad_out_block,
                stride,
                padding,
                dilation,
                groups=effective_groups,
            )
            inp_real = torch.view_as_real(inp)
            inp_block = inp_real.permute(0, 1, 3, 2).reshape(B, 2 * C, L_in)
            grad_weight_block = torch.nn.grad.conv1d_weight(
                inp_block,
                weight_block.shape,
                grad_out_block,
                stride,
                padding,
                dilation,
                groups=effective_groups,
            )
            grad_weight_block_reshaped = grad_weight_block.reshape(C, 2, 2, k)
            grad_top = grad_weight_block_reshaped[:, 0, :, :]
            grad_bottom = grad_weight_block_reshaped[:, 1, :, :]
            grad_W_r = grad_top[:, 0, :] + grad_bottom[:, 1, :]
            grad_W_i = -grad_top[:, 1, :] + grad_bottom[:, 0, :]
            grad_weight = torch.complex(grad_W_r, grad_W_i).unsqueeze(1)
            grad_bias = grad_output.sum(dim=(0, 2)) if bias is not None else None
        else:
            O = weight.shape[0]
            grad_out_block = torch.cat([grad_output.real, grad_output.imag], dim=1)
            effective_groups = groups
            grad_inp_block = F.grad.conv1d_input(
                tuple(ctx.inp_block_shape),
                weight_block,
                grad_out_block,
                stride,
                padding,
                dilation,
                groups=effective_groups,
            )
            grad_inp_real, grad_inp_imag = torch.split(grad_inp_block, C, dim=1)
            grad_inp = torch.complex(grad_inp_real, grad_inp_imag)
            inp_block = torch.cat([inp.real, inp.imag], dim=1)
            grad_weight_block = torch.nn.grad.conv1d_weight(
                inp_block,
                weight_block.shape,
                torch.cat([grad_output.real, grad_output.imag], dim=1),
                stride,
                padding,
                dilation,
                groups,
            )
            C_in_group = weight.shape[1]
            grad_top_left = grad_weight_block[:O, :C_in_group, :]
            grad_top_right = grad_weight_block[:O, C_in_group:, :]
            grad_bottom_left = grad_weight_block[O:, :C_in_group, :]
            grad_bottom_right = grad_weight_block[O:, C_in_group:, :]
            grad_A = grad_top_left + grad_bottom_right
            grad_B = grad_bottom_left - grad_top_right
            grad_weight = torch.complex(grad_A, grad_B)
            grad_bias = grad_output.sum(dim=(0, 2)) if bias is not None else None

        if ctx.branch != "depthwise":
            grad_inp_block = (
                grad_inp_block.reshape(B, C, 2, L_in).permute(0, 1, 3, 2).contiguous()
            )
            grad_inp = torch.view_as_complex(grad_inp_block)
        else:
            grad_inp_block = (
                grad_inp_block.reshape(B, C, 2, L_in).permute(0, 1, 3, 2).contiguous()
            )
            grad_inp = torch.view_as_complex(grad_inp_block)

        return grad_inp, grad_weight, grad_bias, None, None, None, None


class cConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(
                (out_channels, in_channels // groups, kernel_size), dtype=torch.cfloat
            )
        )
        self.bias = (
            nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat))
            if bias
            else None
        )

        nn.init.trunc_normal_(self.weight.data.real, std=0.02)
        nn.init.trunc_normal_(self.weight.data.imag, std=0.02)
        if bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, inp):
        if inp.dtype != torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        return ComplexConv1dFunction.apply(
            inp,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ComplexConv2dFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, stride, padding, dilation, groups):
        B, C_in, H, W = inp.shape
        inp_block = torch.cat([inp.real, inp.imag], dim=1)
        A = weight.real
        B_mat = weight.imag
        top = torch.cat([A, -B_mat], dim=1)
        bottom = torch.cat([B_mat, A], dim=1)
        weight_block = torch.cat([top, bottom], dim=0)
        out_block = F.conv2d(
            inp_block,
            weight_block,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        C_out = weight.shape[0]
        out_real = out_block[:, :C_out, :, :]
        out_imag = out_block[:, C_out:, :, :]
        output = torch.complex(out_real, out_imag)

        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)

        ctx.save_for_backward(inp, weight, bias, weight_block)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_shape = inp.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias, weight_block = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        B, C_in, H_in, W_in = ctx.input_shape
        grad_out_block = torch.cat([grad_output.real, grad_output.imag], dim=1)
        if not isinstance(stride, (tuple, list)):
            stride = (stride, stride)
        if not isinstance(padding, (tuple, list)):
            padding = (padding, padding)
        if not isinstance(dilation, (tuple, list)):
            dilation = (dilation, dilation)
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        kernel_h, kernel_w = weight.shape[2], weight.shape[3]

        H_out = (H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
        W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
        out_pad_h = H_in - (
            (H_out - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + 1
        )
        out_pad_w = W_in - (
            (W_out - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + 1
        )
        output_padding = (max(out_pad_h, 0), max(out_pad_w, 0))
        grad_inp_block = F.conv_transpose2d(
            grad_out_block,
            weight_block,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )
        grad_inp_real = grad_inp_block[:, :C_in, :, :]
        grad_inp_imag = grad_inp_block[:, C_in:, :, :]
        grad_inp = torch.complex(grad_inp_real, grad_inp_imag)
        inp_block = torch.cat([inp.real, inp.imag], dim=1)
        grad_weight_block = torch.nn.grad.conv2d_weight(
            inp_block,
            weight_block.shape,
            grad_out_block,
            stride,
            padding,
            dilation,
            groups,
        )
        C_out = weight.shape[0]
        C_in_group = weight.shape[1]
        grad_top_left = grad_weight_block[:C_out, :C_in_group, :, :]
        grad_top_right = grad_weight_block[:C_out, C_in_group:, :, :]
        grad_bottom_left = grad_weight_block[C_out:, :C_in_group, :, :]
        grad_bottom_right = grad_weight_block[C_out:, C_in_group:, :, :]
        grad_A = grad_top_left + grad_bottom_right
        grad_B = grad_bottom_left - grad_top_right
        grad_weight = torch.complex(grad_A, grad_B)
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None

        return grad_inp, grad_weight, grad_bias, None, None, None, None


class cConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=0,
        bias=True,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        assert (
            in_channels % groups == 0
        ), "In_channels should be an integer multiple of groups."
        assert (
            out_channels % groups == 0
        ), "Out_channels should be an integer multiple of groups."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias = (
            nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat))
            if bias
            else None
        )
        self.weight = nn.Parameter(
            torch.randn(
                (out_channels, in_channels // groups, *kernel_size), dtype=torch.cfloat
            )
        )

        nn.init.trunc_normal_(self.weight.data.real, std=0.02)
        nn.init.trunc_normal_(self.weight.data.imag, std=0.02)
        if bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, inp):
        if inp.dtype != torch.cfloat:
            inp = torch.complex(inp, torch.zeros_like(inp))
        return ComplexConv2dFunction.apply(
            inp,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ComplexNormLayer(nn.Module):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma_rr = nn.Parameter(torch.zeros(channels) + math.sqrt(0.5))
            self.gamma_ii = nn.Parameter(torch.zeros(channels) + math.sqrt(0.5))
            self.gamma_ri = nn.Parameter(torch.zeros(channels))
            self.beta_r = nn.Parameter(torch.zeros(channels))
            self.beta_i = nn.Parameter(torch.zeros(channels))

    def normalize(
        self,
        real: Tensor,
        imag: Tensor,
        dim: int,
        mean_r: Optional[Tensor] = None,
        mean_i: Optional[Tensor] = None,
        Vrr: Optional[Tensor] = None,
        Vii: Optional[Tensor] = None,
        Vri: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        if mean_r is None:
            mean_r = real.mean(dim, keepdim=True)
            mean_i = imag.mean(dim, keepdim=True)

        real = real - mean_r
        imag = imag - mean_i

        if Vrr is None:
            Vrr = real.pow(2).mean(dim, keepdim=True) + self.eps
            Vii = imag.pow(2).mean(dim, keepdim=True) + self.eps
            Vri = (real * imag).mean(dim, keepdim=True) + self.eps
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri**2)
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st
        r = real
        i = imag
        real = Wrr * r + Wri * i
        imag = Wri * r + Wii * i

        if self.affine:
            shape = [1, self.channels] + [1] * (real.ndim - 2)
            real = (
                self.gamma_rr.view(*shape) * real
                + self.gamma_ri.view(*shape) * imag
                + self.beta_r.view(*shape)
            )
            imag = (
                self.gamma_ri.view(*shape) * real
                + self.gamma_ii.view(*shape) * imag
                + self.beta_i.view(*shape)
            )

        return real, imag, mean_r, mean_i, Vrr, Vii, Vri


class cLayerNorm(ComplexNormLayer):

    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2]

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        inp = inp.transpose(1, 2)
        real = torch.real(inp)
        imag = torch.imag(inp)

        real, imag, *_ = self.normalize(real, imag, dim=self.reduced_dim)
        return torch.complex(real, imag).transpose(1, 2)
