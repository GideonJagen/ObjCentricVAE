import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl


class objFG(pl.LightningModule):
    def __init__(
        self,
        num_entities,
        beta=0.1,
        latent_dim=12,
        encoder_depth=2,
        decoder_activation="gelu",
        decoder_norm="batch",
        decoder_feature_size=32,
        decoder_num_layers=5,
        decoder_norm_first=False,
        position_embedding="none",
        position_embedding_dim=1,
        position_prediction_scale=2.0,
        position_representation_scale=0.25,
        glimpse_size=32,
        topk_select_method="random",
    ):
        super(objFG, self).__init__()

        self.num_entities = num_entities
        self.beta = beta
        self.latent_dim = latent_dim

        self.decoder_activation = decoder_activation
        self.decoder_norm = decoder_norm
        self.decoder_feature_size = decoder_feature_size
        self.decoder_num_layers = decoder_num_layers
        self.decoder_norm_first = decoder_norm_first

        self.position_embedding = position_embedding
        self.position_embedding_dim = position_embedding_dim * 4
        self.position_prediction_scale = position_prediction_scale
        self.position_representation_scale = position_representation_scale

        self.glimpse_size = glimpse_size

        self.topk_select_method = topk_select_method

        actual_latent_dim = latent_dim * 2 + 1

        decoder_latent_dim = latent_dim

        if position_embedding == "none":
            ...
        elif position_embedding == "radial":
            decoder_latent_dim += 1
        elif position_embedding == "sine":
            decoder_latent_dim += self.position_embedding_dim - 2

        self.kl_importance = 0
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, 2, 1),
        #     nn.GELU(),
        #     nn.GroupNorm(4, 16),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.GELU(),
        #     nn.GroupNorm(4, 16),
        #     nn.Conv2d(16, 32, 3, 2, 1),
        #     nn.GELU(),
        #     nn.GroupNorm(8, 32),
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.GELU(),
        #     nn.GroupNorm(8, 32),
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.GELU(),
        #     nn.GroupNorm(8, 32),
        #     nn.Conv2d(32, actual_latent_dim, 3, padding=1),
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            *[
                nn.Sequential(
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                for _ in range(encoder_depth - 1)
            ],
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, actual_latent_dim, 3, padding=1),
        )

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
                return nn.BatchNorm1d(num_channels)
            elif norm == "group":
                return nn.GroupNorm(num_channels // 4, num_channels)
            elif norm == "instance":
                return nn.InstanceNorm1d(num_channels)
            elif norm == "layer":
                return nn.LayerNorm(num_channels)
            else:
                raise NotImplementedError

        def decoder_block(in_filters, out_filters, act_str, norm_str, norm_first):
            if norm_first:
                return nn.Sequential(
                    nn.Linear(in_filters, out_filters),
                    norm(norm_str, out_filters),
                    activation(act_str),
                )
            else:
                return nn.Sequential(
                    nn.Linear(in_filters, out_filters),
                    activation(act_str),
                    norm(norm_str, out_filters),
                )

        self.decoder_1 = nn.Sequential(
            *[
                decoder_block(
                    decoder_latent_dim,
                    decoder_feature_size,
                    decoder_activation,
                    decoder_norm,
                    decoder_norm_first,
                ),
                *[
                    decoder_block(
                        decoder_feature_size,
                        decoder_feature_size,
                        decoder_activation,
                        decoder_norm,
                        decoder_norm_first,
                    )
                    for _ in range(decoder_num_layers - 1)
                ],
            ]
        )

        self.pixel_decoder = nn.Sequential(
            nn.Linear(decoder_feature_size, 1),
        )

        # self.weight_decoder = nn.Sequential(
        #     nn.Linear(decoder_feature_size, 1),
        # )

    def distribution_map(self, x):

        y = self.encoder(x)

        z_pres = torch.sigmoid(y[:, 0, :, :] * 2)
        z_what = y[:, 1:, :, :]

        mu = z_what[:, : self.latent_dim, :, :]
        logvar = z_what[:, self.latent_dim :, :, :]

        return mu, logvar

    def parametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        parametrization = torch.randn_like(std) * std + mu
        return parametrization

    def split_latent(self, parametrization):
        delta_xy = parametrization[:, :2, :, :].view(parametrization.shape[0], 2, -1)
        parametrization = parametrization[:, 2:, :, :].view(
            parametrization.shape[0], self.latent_dim - 2, -1
        )
        return delta_xy, parametrization

    def select_topk(self, kl_divergence, topk=None):

        if topk is None:
            topk = self.num_entities
        top_kl_divergence, indices = torch.topk(kl_divergence, topk, dim=1)
        indices = indices.detach()
        return indices

    def sample_topk(self, kl_divergence, topk=None):
        if topk is None:
            topk = self.num_entities

        # normalize kl_divergence as a probability distribution
        kl_divergence = F.softmax(kl_divergence, dim=1)

        # replace nan with 0
        kl_divergence = torch.where(
            torch.isnan(kl_divergence), torch.zeros_like(kl_divergence), kl_divergence
        )

        # replace inf with 0
        kl_divergence = torch.where(
            torch.isinf(kl_divergence), torch.zeros_like(kl_divergence), kl_divergence
        )

        # replace negative values with 0
        kl_divergence = torch.where(
            kl_divergence < 0, torch.zeros_like(kl_divergence), kl_divergence
        )

        # renormalize kl_divergence as a probability distribution
        kl_divergence = kl_divergence / torch.sum(kl_divergence, dim=1, keepdim=True)

        # randomly sample topk indices from kl_divergence

        indices = [
            torch.multinomial(kl_divergence[b], topk, replacement=False)
            for b in range(kl_divergence.shape[0])
        ]
        indices = torch.stack(indices)
        indices = indices.detach()
        return indices

    def xy_pos_latents(self, indices, in_size, latent_size):

        x_coord = indices // latent_size[3]
        y_coord = indices % latent_size[3]

        return (
            (x_coord + 0.5) * in_size[2] / latent_size[2],
            (y_coord + 0.5) * in_size[3] / latent_size[3],
        )

    def index_parametrization(self, parametrization, indices):
        return torch.stack(
            [
                torch.stack(
                    [parametrization[batch_idx, :, idx] for idx in indices[batch_idx]]
                )
                for batch_idx in range(parametrization.shape[0])
            ]
        )

    def forward(self, x):
        batch_size = x.shape[0]

        y = self.encoder(x)

        x_mass = torch.abs(x)

        # resize x_mass
        x_mass = F.interpolate(
            x_mass,
            size=(y.shape[2], y.shape[3]),
            mode="bilinear",
            align_corners=True,
        )

        z_pres = torch.sigmoid(y[:, :1] * 2).view(batch_size, -1)
        z_what = y[:, 1:]

        mu = z_what[:, : self.latent_dim]
        logvar = z_what[:, self.latent_dim :]

        x_range = torch.arange(0, x.shape[2], device=x.device, dtype=torch.float32)
        y_range = torch.arange(0, x.shape[3], device=x.device, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x_range, y_range)

        std = torch.exp(0.5 * logvar)

        parametrization = torch.randn_like(std) * std + mu

        delta_xy_pred = parametrization[:, :2].view(batch_size, 2, -1)
        parametrization = parametrization[:, 2:].view(
            batch_size, self.latent_dim - 2, -1
        )

        # Calculate the KL divergence from the prior of gaussian distribution with mean 0 and std 1
        # to the posterior of the gaussian distribution with mean mu and std std
        # KL divergence is calculated as 0.5 * sum(1 + log(std^2) - mu^2 - std^2)

        kl_divergence = -0.5 * torch.mean(1 + logvar - mu**2 - std**2, dim=1)
        kl_divergence = kl_divergence.view(batch_size, -1)
        # find the i,j indices of the max value num_entities elements in the kl_divergence tensor
        # these indices will be used to select the num_entities most important entities

        score = (
            z_pres**0.01
            * kl_divergence**self.kl_importance
            * x_mass.view(batch_size, -1) ** (1 - self.kl_importance)
        )
        if self.topk_select_method == "random":
            indices = self.sample_topk(score)
        elif self.topk_select_method == "max":
            indices = self.select_topk(score)
        else:
            raise NotImplementedError
        # top_kl_divergence, indices = torch.topk(kl_divergence, self.num_entities, dim=1)
        # indices = indices.detach()
        intermediate_feature_map = None

        latents = parametrization[torch.arange(batch_size)[:, None], :, indices]
        pred_delta_xy = delta_xy_pred[torch.arange(batch_size)[:, None], :, indices]
        presence = z_pres[torch.arange(batch_size)[:, None], indices].view(
            batch_size, -1, 1, 1, 1
        )

        # repeat latents to match the size of input image
        # (batch_size, num_entities, latent_dim) ->
        # (batch_size, num_entities, x.shape[2] * x.shape[3], latent_dim)
        latents = latents[:, :, None, :].repeat(1, 1, x.shape[2] * x.shape[3], 1)

        # grid channel is a tensor of size (batch_size, latent_dim, 2, x.shape[2], x.shape[3])
        # it contains the x and y coordinates of each pixel in the image
        grid_channel = torch.stack([x_grid, y_grid], dim=-1).view(1, 1, -1, 2)

        # center the grid channel around the coordinates of the entities
        x_coord = indices // y.shape[3]
        y_coord = indices % y.shape[3]

        # add 0.5 to the coordinates to center the grid channel around the coordinates of the entities
        # scale the coordinates to match the size of the input image
        x_coord = (x_coord + 0.5) * x.shape[2] / y.shape[2]
        y_coord = (y_coord + 0.5) * x.shape[3] / y.shape[3]

        xy_coord = torch.stack([x_coord, y_coord], dim=-1).view(batch_size, -1, 1, 2)
        pred_delta_xy = pred_delta_xy.unsqueeze(2)

        xy = xy_coord - pred_delta_xy * self.position_prediction_scale
        grid_channel = (grid_channel - xy) * self.position_representation_scale

        r_channel = torch.norm(grid_channel, p=2, dim=-1)
        mask = r_channel < (self.glimpse_size * self.position_representation_scale)

        reduced_grid = grid_channel[mask, :]
        reduced_latents = latents[mask, :]

        if self.position_embedding == "none":
            ...
        elif self.position_embedding == "radial":
            r_channel = torch.norm(reduced_grid, p=2, dim=-1, keepdim=True)
            reduced_grid = torch.cat([reduced_grid, r_channel], dim=-1)
        elif self.position_embedding == "sine":
            reduced_grid = self.sine_embedding(reduced_grid)

        # concatenate the grid channel and the latents
        # (batch_size, num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        intermediate_feature_map = torch.cat([reduced_latents, reduced_grid], dim=-1)

        decoded_feature_map = self.decoder_1(intermediate_feature_map)

        pixel_map = self.pixel_decoder(decoded_feature_map)

        # recreate sparse representation
        pixel_map_tensor = torch.zeros(
            batch_size, self.num_entities, x.shape[2] * x.shape[3], device=x.device
        )

        pixel_map_tensor[mask] = pixel_map.view(-1)

        pixel_map = pixel_map_tensor.view(
            batch_size, self.num_entities, 1, x.shape[2], x.shape[3]
        )

        weight_map = (
            mask.view(batch_size, self.num_entities, 1, x.shape[2], x.shape[3])
            * presence
        )

        y = torch.sum(pixel_map * weight_map, dim=1)

        xy = xy.view(batch_size, self.num_entities, 2)
        latents = latents.view(batch_size, self.num_entities, -1)
        presence = presence.view(batch_size, self.num_entities)

        presence_loss = (z_pres - 1) ** 2 / 10

        kl_divergence = kl_divergence + presence_loss

        return [y, indices, kl_divergence, presence, xy, latents]

    def loss_function(self, x, x_hat, kl_divergence, mu, logvar):
        # Reconstruction loss
        error = x_hat - x

        reconstruction_loss = torch.mean(torch.abs(error))

        # KL divergence loss
        kl_divergence_loss = torch.mean(torch.sum(kl_divergence, dim=1), dim=0)

        # MN weighting
        weight_MN = x.shape[1] * x.shape[2] * x.shape[3] / 12

        # Total loss
        loss = reconstruction_loss + kl_divergence_loss * weight_MN * self.beta
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "KLD": kl_divergence_loss,
            "weighted_KLD": kl_divergence_loss * weight_MN * self.beta,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)

    def sine_embedding(self, xy_grid):
        import math

        # xy_grid: (batch_size, num_entities, 2, x.shape[2], x.shape[3])
        # xy_coords: (batch_size, num_entities, 2, x.shape[2] * x.shape[3])
        # xy_coords = torch.flatten(xy_grid, start_dim=3)

        # x: (batch_size, num_entities, x.shape[2] * x.shape[3])
        x = xy_grid[..., 0]

        # y: (batch_size, num_entities, x.shape[2] * x.shape[3])
        y = xy_grid[..., 1]

        n_embed = self.position_embedding_dim // 2

        div_term = torch.exp(
            torch.arange(0, n_embed, 2) * -(math.log(100.0) / n_embed)
        ).to(self.device)

        embed_x_sin = torch.sin(x[..., None] * div_term)
        embed_x_cos = torch.cos(x[..., None] * div_term)
        embed_y_sin = torch.sin(y[..., None] * div_term)
        embed_y_cos = torch.cos(y[..., None] * div_term)

        # x: (batch_size, num_entities, x.shape[2] * x.shape[3], n_embed)
        x = torch.cat([embed_x_sin, embed_x_cos, embed_y_sin, embed_y_cos], dim=-1)

        # x: (batch_size, num_entities, n_embed, x.shape[2] * x.shape[3])
        # x = x.permute(0, 1, 3, 2)

        # x: (batch_size, num_entities, n_embed, x.shape[2], x.shape[3])
        x = x.view(
            *xy_grid.shape[:-1],
            self.position_embedding_dim,
        )

        return x


class MEVAE(pl.LightningModule):
    def __init__(self, num_entities, beta=0.1):
        super().__init__()
        self.model = MultiEntityVariationalAutoEncoder(num_entities, beta)

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, batch_idx):
        x_hat, indices, kl_divergence, xy, mu, logvar = self(x)
        kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]

        kl_divergence = torch.stack(kl_divergence)
        loss_dict = self.model.loss_function(x, x_hat, kl_divergence, mu, logvar)
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]

    def validation_step(self, x, batch_idx):
        x_hat, indices, kl_divergence, xy, mu, logvar = self(x)
        kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]

        kl_divergence = torch.stack(kl_divergence)

        loss_dict = self.model.loss_function(x, x_hat, kl_divergence, mu, logvar)
        # self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]
