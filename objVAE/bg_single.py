import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


def activation(act: str):
    if act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "elu":
        return nn.ELU()
    elif act == "silu":
        return nn.SiLU()
    elif act == "mish":
        return nn.Mish()
    else:
        raise NotImplementedError


def norm(norm: str, num_channels: int):
    if norm == "none":
        return nn.Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm == "group":
        return nn.GroupNorm(num_channels // 4, num_channels)
    elif norm == "instance":
        return nn.InstanceNorm2d(num_channels)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    else:
        raise NotImplementedError


def encoder_block(in_filters, out_filters, act_str, norm_str, norm_first):

    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        activation(act_str),
        norm(norm_str, out_filters),
    )


def decoder_block(in_filters, out_filters, act_str, norm_str, norm_first):
    return nn.Sequential(
        nn.ConvTranspose2d(in_filters, out_filters, 1, padding=0),
        activation(act_str),
        norm(norm_str, out_filters),
    )


class objBG(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        hidden_dim=128,
        image_size=256,
        encoder_depth=4,
        encoder_channels=32,
        encoder_activation="gelu",
        encoder_norm="group",
        decoder_depth=4,
        decoder_channels=32,
        decoder_activation="silu",
        decoder_norm="instance",
        position_embedding="sine",
        position_scale=10,
        position_dim=8,
    ):
        super(objBG, self).__init__()
        self.image_size = image_size

        embed_size = image_size // 2**encoder_depth

        self.encoder = nn.Sequential(
            encoder_block(
                in_channels, encoder_channels, encoder_activation, encoder_norm, False
            ),
            *[
                encoder_block(
                    encoder_channels,
                    encoder_channels,
                    encoder_activation,
                    encoder_norm,
                    False,
                )
                for _ in range(encoder_depth - 1)
            ],
            nn.Flatten(),
            nn.Linear(encoder_channels * embed_size * embed_size, hidden_dim * 2),
        )

        self.decoder = BroadcastDecoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            image_size=image_size,
            decoder_depth=decoder_depth,
            decoder_channels=decoder_channels,
            decoder_activation=decoder_activation,
            decoder_norm=decoder_norm,
            position_embedding=position_embedding,
            position_scale=position_scale,
            position_dim=position_dim,
        )

    def forward(self, x):

        # interpolate x to self.image_size
        reshaped = x
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            reshaped = F.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear"
            )

        # (B, C, H, W) -> (B, hidden_dim * 2)
        y = self.encoder(reshaped)

        # (B, hidden_dim), (B, hidden_dim)
        z_bg_mu, z_bg_logvar = y.chunk(2, dim=1)

        # print(torch.min(z_bg_logvar), torch.max(z_bg_logvar))

        z_bg_std = torch.exp(0.5 * z_bg_logvar) + 1e-8
        z_bg_posterior = Normal(z_bg_mu, z_bg_std)

        z_bg = z_bg_posterior.rsample()

        # compute kl divergence
        z_bg_prior = Normal(torch.zeros_like(z_bg_mu), torch.ones_like(z_bg_std))
        kl_bg = kl_divergence(z_bg_posterior, z_bg_prior).sum(dim=1).mean()

        bg = self.decoder(z_bg, x.shape[-2], x.shape[-1])

        return bg, z_bg, kl_bg

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)


class SpatialBroadcast(pl.LightningModule):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self, position_embedding="sine", position_scale=10, position_dim=8):
        super().__init__()
        self.position_embedding = position_embedding
        self.position_scale = position_scale
        self.position_dim = position_dim

    def sine_embedding(self, width, height):
        import math

        # xy_grid: (batch_size, num_entities, 2, x.shape[2], x.shape[3])
        # xy_coords: (batch_size, num_entities, 2, x.shape[2] * x.shape[3])
        # xy_coords = torch.flatten(xy_grid, start_dim=3)

        # x: (width*height)
        # y: (width*height)
        x = torch.arange(0, width, dtype=torch.float32).to(self.device)
        y = torch.arange(0, height, dtype=torch.float32).to(self.device)
        x, y = torch.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()

        n_embed = self.position_dim // 2

        div_term = torch.exp(
            torch.arange(0, n_embed, 2) * -(math.log(10000.0) / n_embed)
        ).to(self.device)
        embed_x_sin = torch.sin(x[..., None] * div_term)
        embed_x_cos = torch.cos(x[..., None] * div_term)
        embed_y_sin = torch.sin(y[..., None] * div_term)
        embed_y_cos = torch.cos(y[..., None] * div_term)

        # x: (width*height, position_dim)
        x = torch.cat([embed_x_sin, embed_x_cos, embed_y_sin, embed_y_cos], dim=-1)
        x = x.permute(1, 0)
        x = x.view(-1, width, height)

        return x

    def linear_embedding(self, width, height):
        x = torch.linspace(-1, 1, width, device=self.device) * self.position_scale
        y = torch.linspace(-1, 1, height, device=self.device) * self.position_scale
        x, y = torch.meshgrid((x, y))
        x = torch.stack((x, y), dim=0)
        return x

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions

        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, W, H)
        """
        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.expand(B, L, width, height)

        if self.position_embedding == "sine":
            coords = self.sine_embedding(width, height)
        elif self.position_embedding == "linear":
            coords = self.linear_embedding(width, height)
        else:
            raise NotImplementedError

        # (B, 2, H, W)
        coords = coords[None].expand(B, coords.shape[0], width, height)

        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)

        return x


class BroadcastDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_dim=128,
        image_size=256,
        decoder_depth=4,
        decoder_channels=32,
        decoder_activation="silu",
        decoder_norm="instance",
        position_embedding="sine",
        position_scale=10,
        position_dim=8,
    ):
        super(BroadcastDecoder, self).__init__()
        self.image_size = image_size

        self.broadcast = SpatialBroadcast(
            position_embedding=position_embedding,
            position_scale=position_scale,
            position_dim=position_dim,
        )
        n_dim = hidden_dim
        if position_embedding == "sine":
            n_dim += position_dim
        elif position_embedding == "linear":
            n_dim += 2

        self.decoder = nn.Sequential(
            decoder_block(
                n_dim,
                decoder_channels,
                decoder_activation,
                decoder_norm,
                False,
            ),
            *[
                decoder_block(
                    decoder_channels,
                    decoder_channels,
                    decoder_activation,
                    decoder_norm,
                    False,
                )
                for _ in range(decoder_depth - 1)
            ],
            nn.Conv2d(decoder_channels, in_channels, 1),
        )

    def forward(self, z, width, height):

        # broadcast z to 3-D
        # (B, hidden_dim + 2, w, h)
        z = self.broadcast(z, width, height)

        # (B, C, w, h)
        reconstruction = self.decoder(z)

        return reconstruction
