"""PeSANet model for 2D Gray-Scott dynamics."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from peasnet.models.kno import KNO2d


def periodic_padding(x: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """Apply periodic padding to the last two spatial dimensions."""

    if padding == 0:
        return x
    x_pad = torch.cat((x[:, :, :, -padding:], x, x[:, :, :, :padding]), dim=3)
    x_pad = torch.cat((x_pad[:, :, -padding:, :], x_pad, x_pad[:, :, :padding, :]), dim=2)
    return x_pad


def _laplace_kernel() -> torch.Tensor:
    return torch.tensor(
        [[[[0.0, 0.0, -1.0 / 12.0, 0.0, 0.0],
           [0.0, 0.0, 4.0 / 3.0, 0.0, 0.0],
           [-1.0 / 12.0, 4.0 / 3.0, -5.0, 4.0 / 3.0, -1.0 / 12.0],
           [0.0, 0.0, 4.0 / 3.0, 0.0, 0.0],
           [0.0, 0.0, -1.0 / 12.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )


class GrayScottCell(nn.Module):
    """Physics-inspired recurrent cell from the original GS_KNOATTSplittest.py."""

    def __init__(
        self,
        hidden_channels: int = 16,
        dx: float = 1.0 / 128.0,
        dt: float = 0.5,
        diffusion_upper_bound: float = 2e-5,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.input_channels = 2
        self.hidden_channels = hidden_channels
        self.input_kernel_size = 5
        self.input_stride = 1
        self.dx = dx
        self.dt = dt
        self.mu_up = diffusion_upper_bound

        self.CA = nn.Parameter(torch.empty((), dtype=torch.float32).uniform_(-1.0, 1.0))
        self.CB = nn.Parameter(torch.empty((), dtype=torch.float32).uniform_(-1.0, 1.0))

        self.W_laplace = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        with torch.no_grad():
            self.W_laplace.weight.copy_(_laplace_kernel() / (self.dx ** 2))
        self.W_laplace.weight.requires_grad = False

        self.Wh1_u = nn.Conv2d(2, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.Wh2_u = nn.Conv2d(2, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.Wh3_u = nn.Conv2d(2, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.Wh4_u = nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0)
        self.Wh1_v = nn.Conv2d(2, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.Wh2_v = nn.Conv2d(2, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.Wh3_v = nn.Conv2d(2, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.Wh4_v = nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0)

        self.init_filter(
            [
                self.Wh1_u,
                self.Wh2_u,
                self.Wh3_u,
                self.Wh4_u,
                self.Wh1_v,
                self.Wh2_v,
                self.Wh3_v,
                self.Wh4_v,
            ],
            c=init_scale,
        )

    @staticmethod
    def init_filter(filters: Iterable[nn.Conv2d], c: float) -> None:
        for layer in filters:
            nn.init.xavier_uniform_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(c)
                if layer.bias is not None:
                    layer.bias.zero_()

    def _rhs(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_pad = periodic_padding(h, padding=2)
        u_pad = h_pad[:, 0:1]
        v_pad = h_pad[:, 1:2]

        u_res = self.mu_up * torch.sigmoid(self.CA) * self.W_laplace(u_pad)
        u_res = u_res + self.Wh4_u(self.Wh1_u(h_pad) * self.Wh2_u(h_pad) * self.Wh3_u(h_pad))

        v_res = self.mu_up * torch.sigmoid(self.CB) * self.W_laplace(v_pad)
        v_res = v_res + self.Wh4_v(self.Wh1_v(h_pad) * self.Wh2_v(h_pad) * self.Wh3_v(h_pad))
        return u_res, v_res

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_prev = h[:, 0:1]
        v_prev = h[:, 1:2]

        k1_u, k1_v = self._rhs(h)
        u1 = u_prev + k1_u * self.dt / 2.0
        v1 = v_prev + k1_v * self.dt / 2.0

        k2_u, k2_v = self._rhs(torch.cat((u1, v1), dim=1))
        u2 = u_prev + k2_u * self.dt / 2.0
        v2 = v_prev + k2_v * self.dt / 2.0

        k3_u, k3_v = self._rhs(torch.cat((u2, v2), dim=1))
        u3 = u_prev + k3_u * self.dt
        v3 = v_prev + k3_v * self.dt

        k4_u, k4_v = self._rhs(torch.cat((u3, v3), dim=1))

        u_next = u_prev + (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u) * self.dt / 6.0
        v_next = v_prev + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * self.dt / 6.0
        next_state = torch.cat((u_next, v_next), dim=1)
        return next_state, next_state


class LatentEncoder(nn.Module):
    """Convolutional encoder used before the Koopman operator."""

    def __init__(self) -> None:
        super().__init__()
        self.enc_conv1 = nn.Conv2d(2, 8, kernel_size=5, stride=2, padding=0)
        self.enc_conv2 = nn.Conv2d(8, 32, kernel_size=5, stride=2, padding=0)
        self.enc_conv3 = nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=0)
        self.enc_relu3 = nn.CELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_relu3(self.enc_conv1(periodic_padding(x, padding=2)))
        x = self.enc_relu3(self.enc_conv2(periodic_padding(x, padding=2)))
        return self.enc_conv3(periodic_padding(x, padding=2))


class LatentDecoder(nn.Module):
    """Decoder matching the original pixel-shuffle reconstruction path."""

    def __init__(self) -> None:
        super().__init__()
        self.dec_conv1 = nn.Conv2d(2, 16, kernel_size=11, stride=1, padding=0)
        self.dec_relu1 = nn.ReLU()
        self.dec_conv2 = nn.Conv2d(16, 2, kernel_size=11, stride=1, padding=0)
        self.px = nn.PixelShuffle(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.px(x)
        x = self.dec_conv1(periodic_padding(x, padding=5))
        x = self.dec_relu1(x)
        return self.dec_conv2(periodic_padding(x, padding=5))


class ConvAutoencoder(nn.Module):
    """Autoencoder wrapper retained for checkpoint compatibility."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = LatentEncoder()
        self.decoder = LatentDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class PesaNet(nn.Module):
    """PeSANet recurrent model with Gray-Scott cell and spectral attention update."""

    def __init__(
        self,
        step: int = 1,
        effective_step: Iterable[int] | None = None,
        init_state: torch.Tensor | None = None,
        modes_x: int = 8,
        modes_y: int = 8,
        dx: float = 1.0 / 128.0,
        dt: float = 0.5,
    ) -> None:
        super().__init__()
        self.step = step
        self.effective_step = list(range(step)) if effective_step is None else list(effective_step)
        self.init_state = init_state
        self.kno = KNO2d(LatentEncoder(), LatentDecoder(), op_size=128, modes_x=modes_x, modes_y=modes_y)
        self.crnn_cell = GrayScottCell(dx=dx, dt=dt)

    def forward(
        self,
        init_state: torch.Tensor | None = None,
        return_list: bool = False,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | None]:
        if init_state is not None:
            self.init_state = init_state
        if self.init_state is None:
            raise ValueError("init_state must be provided either in __init__ or forward().")

        h = self.init_state
        outputs = [h.unsqueeze(1)]
        second_last_state = None

        for step_idx in range(self.step):
            kno_update = self.kno(h)
            h, _ = self.crnn_cell(h)
            h = h + kno_update

            if step_idx == self.step - 2:
                second_last_state = h.clone()

            if step_idx in self.effective_step:
                outputs.append(h.unsqueeze(1))

        if return_list:
            return outputs, second_last_state
        return torch.cat(outputs, dim=1), second_last_state


RCNN = PesaNet
