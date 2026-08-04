"""StyleTTS2 waveform GAN: MultiPeriod + MultiResSpec discriminators.

Ported from yl4579/StyleTTS2 Modules/discriminators.py + losses.py
(LSGAN + feature matching + TPRLS). Used as aux d_loss / loss_gen while
Kokoro stays frozen and TimbreEncoder is the only trainable generator path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _as_b1t(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.view(1, 1, -1)
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


def match_wav_pair(
    y: torch.Tensor, y_hat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    y = _as_b1t(y)
    y_hat = _as_b1t(y_hat)
    n = min(y.shape[-1], y_hat.shape[-1])
    if n < 256:
        raise ValueError(f"wav too short for GAN: {n}")
    return y[..., :n].contiguous(), y_hat[..., :n].contiguous()


def stft_mag(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        return_complex=True,
    )
    return torch.abs(x_stft).transpose(2, 1)


class SpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        window: str = "hann_window",
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                    )
                ),
            ]
        )
        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y: torch.Tensor):
        fmap = []
        y = y.squeeze(1)
        y = stft_mag(
            y, self.fft_size, self.shift_size, self.win_length, self.window
        )
        y = y.unsqueeze(1)
        for d in self.discriminators:
            y = F.leaky_relu(d(y), LRELU_SLOPE)
            fmap.append(y)
        y = self.out(y)
        fmap.append(y)
        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_lengths: list[int] | None = None,
        window: str = "hann_window",
    ):
        super().__init__()
        fft_sizes = fft_sizes or [1024, 2048, 512]
        hop_sizes = hop_sizes or [120, 240, 50]
        win_lengths = win_lengths or [600, 1200, 240]
        self.discriminators = nn.ModuleList(
            [
                SpecDiscriminator(fs, hs, wl, window)
                for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        channels: tuple[int, ...] = (32, 128, 512, 1024, 1024),
    ):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        c0, c1, c2, c3, c4 = channels
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        c0,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        c0,
                        c1,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        c1,
                        c2,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        c2,
                        c3,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(c3, c4, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(c4, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        channels: tuple[int, ...] = (32, 128, 512, 1024, 1024),
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p, channels=channels) for p in periods]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    loss = fmap_r[0][0].new_zeros(())
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs) -> torch.Tensor:
    loss = disc_real_outputs[0].new_zeros(())
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        loss = loss + torch.mean((1 - dr) ** 2) + torch.mean(dg**2)
    return loss


def generator_loss(disc_outputs) -> torch.Tensor:
    loss = disc_outputs[0].new_zeros(())
    for dg in disc_outputs:
        loss = loss + torch.mean((1 - dg) ** 2)
    return loss


def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs) -> torch.Tensor:
    loss = disc_real_outputs[0].new_zeros(())
    tau = 0.04
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        m_dg = torch.median(dr - dg)
        mask = dr < dg + m_dg
        if mask.any():
            l_rel = torch.mean((((dr - dg) - m_dg) ** 2)[mask])
            loss = loss + tau - F.relu(tau - l_rel)
    return loss


def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs) -> torch.Tensor:
    return discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs)


class StyleTTS2GAN(nn.Module):
    """MPD + MSD wrappers matching StyleTTS2 DiscriminatorLoss / GeneratorLoss."""

    def __init__(self, slim: bool = False):
        super().__init__()
        if slim:
            channels = (16, 64, 128, 256, 256)
            periods = (2, 3, 5, 7)
        else:
            channels = (32, 128, 512, 1024, 1024)
            periods = (2, 3, 5, 7, 11)
        self.mpd = MultiPeriodDiscriminator(periods=periods, channels=channels)
        self.msd = MultiResSpecDiscriminator()

    def d_loss(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        y, y_hat = match_wav_pair(y, y_hat)
        y_df_r, y_df_g, _, _ = self.mpd(y, y_hat)
        y_ds_r, y_ds_g, _, _ = self.msd(y, y_hat)
        loss = (
            discriminator_loss(y_df_r, y_df_g)
            + discriminator_loss(y_ds_r, y_ds_g)
            + discriminator_TPRLS_loss(y_df_r, y_df_g)
            + discriminator_TPRLS_loss(y_ds_r, y_ds_g)
        )
        return loss.mean()

    def g_loss(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        y, y_hat = match_wav_pair(y, y_hat)
        y_df_r, y_df_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        loss = (
            generator_loss(y_df_g)
            + generator_loss(y_ds_g)
            + feature_loss(fmap_f_r, fmap_f_g)
            + feature_loss(fmap_s_r, fmap_s_g)
            + generator_TPRLS_loss(y_df_r, y_df_g)
            + generator_TPRLS_loss(y_ds_r, y_ds_g)
        )
        return loss.mean()
