"""Koopman neural operator blocks used by PeSANet."""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dEncoder(nn.Module):
    """Simple 1x1 convolutional encoder for standalone KNO2d use."""

    def __init__(self, input_channels: int, op_size: int) -> None:
        super().__init__()
        self.layer = nn.Conv2d(input_channels, op_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Conv2dDecoder(nn.Module):
    """Simple 1x1 convolutional decoder for standalone KNO2d use."""

    def __init__(self, output_channels: int, op_size: int) -> None:
        super().__init__()
        self.layer = nn.Conv2d(op_size, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ChannelAttention(nn.Module):
    """Channel attention block applied separately to real and imaginary parts."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class ComplexChannelAttention(nn.Module):
    """Complex-valued channel attention with residual connection."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.ca1 = ChannelAttention(in_channels, reduction_ratio)
        self.ca2 = ChannelAttention(in_channels, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real_part = x.real
        imag_part = x.imag
        att_real = self.ca1(real_part)
        att_imag = self.ca2(imag_part)
        out_real = real_part * att_real - imag_part * att_imag
        out_imag = real_part * att_imag + imag_part * att_real
        return torch.complex(out_real, out_imag) + x


class KoopmanOperator2D(nn.Module):
    """2D Fourier-domain Koopman time marching layer."""

    def __init__(self, op_size: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.op_size = op_size
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.scale = 1.0 / (op_size * op_size)
        self.ca = ComplexChannelAttention(op_size)
        self.koopman_matrix = nn.Parameter(
            self.scale
            * torch.rand(op_size, op_size, modes_x, modes_y, dtype=torch.cfloat)
        )

    @staticmethod
    def time_marching(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btxy,tfxy->bfxy", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        x_ft = self.ca(x_ft)

        modes_x = min(self.modes_x, x_ft.size(-2))
        modes_y = min(self.modes_y, x_ft.size(-1))
        weights = self.koopman_matrix[:, :, :modes_x, :modes_y]

        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :modes_x, :modes_y] = self.time_marching(
            x_ft[:, :, :modes_x, :modes_y], weights
        )
        out_ft[:, :, -modes_x:, :modes_y] = self.time_marching(
            x_ft[:, :, -modes_x:, :modes_y], weights
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class KNO2d(nn.Module):
    """2D Koopman neural operator with optional custom encoder and decoder."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        op_size: int = 32,
        modes_x: int = 30,
        modes_y: int = 30,
        decompose: int = 2,
        linear_type: bool = True,
        input_channels: int = 2,
        output_channels: int = 2,
    ) -> None:
        super().__init__()
        self.op_size = op_size
        self.decompose = decompose
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.linear_type = linear_type

        self.enc = encoder if encoder is not None else Conv2dEncoder(input_channels, op_size)
        self.dec = decoder if decoder is not None else Conv2dDecoder(output_channels, op_size)
        self.koopman_layer = KoopmanOperator2D(op_size, modes_x, modes_y)
        self.w0 = nn.Conv2d(op_size, op_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.enc(x))
        for _ in range(self.decompose):
            update = self.koopman_layer(x)
            x = x + update if self.linear_type else torch.tanh(x + update)
        return self.dec(x)
