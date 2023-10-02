import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from objVAE.MultiheadAttention import MultiheadAttention

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)
# , number_of_heads=1, softmax_tmp = 1,


class MultiEntityVariationalAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        num_entities,
        attention_model=None,
        beta=0.1,
        latent_dim=12,
        attention=True,
        single_decoder=True,
        combine_method="sum",
        object_radius=12,
    ):
        super(MultiEntityVariationalAutoEncoder, self).__init__()

        self.num_entities = num_entities
        self.beta = beta
        self.latent_dim = latent_dim
        self.attention = attention
        self.combine_method = combine_method
        self.object_radius = object_radius
        self.single_decoder = single_decoder

        self.presence_bias = 1
        actual_latent_dim = latent_dim * 2 + 1

        self.time_attention = attention_model
        if self.time_attention == None:
            self.time_attention = MultiheadAttention(num_filters=latent_dim + 1)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, actual_latent_dim, 3, padding=1),
        )

        if single_decoder:
            self.decoder_1 = nn.Sequential(
                nn.Conv2d(self.latent_dim + 1, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1, padding=0),
            )
        else:
            self.decoder_1 = nn.Sequential(
                nn.Conv2d(self.latent_dim + 1, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1, padding=0),
            )

            self.decoder_2 = nn.Sequential(
                nn.Conv2d(32, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1, padding=0),
            )

    def forward(self, x):
        true_batch_size = x.shape[0]
        timesteps = x.shape[1]

        batch_size = true_batch_size * timesteps

        x = x.view(batch_size, x.shape[2], x.shape[3], x.shape[4])

        y = self.encoder(x)

        z_pres = torch.sigmoid(y[:, :1]).view(batch_size, -1)
        z_what = y[:, 1:]

        mu = z_what[:, : self.latent_dim, :, :]
        logvar = z_what[:, self.latent_dim :, :, :]

        x_range = torch.arange(0, x.shape[3], device=x.device, dtype=torch.float32)
        y_range = torch.arange(0, x.shape[2], device=x.device, dtype=torch.float32)
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

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu**2 - std**2, dim=1)
        kl_divergence = kl_divergence.view(batch_size, -1)
        # find the i,j indices of the max value num_entities elements in the kl_divergence tensorimage_size
        # these indices will be used to select the num_entities most important entities

        top_kl_divergence, indices = torch.topk(kl_divergence, self.num_entities, dim=1)
        indices = indices.detach()
        intermediate_feature_map = None

        latents = torch.stack(
            [
                torch.stack(
                    [parametrization[batch_idx, :, idx] for idx in indices[batch_idx]]
                )
                for batch_idx in range(batch_size)
            ]
        )

        pres = z_pres[torch.arange(batch_size)[:, None], indices].view(
            batch_size, -1, 1, 1, 1
        )

        # NEWLY ADDED
        x_coord = indices // y.shape[3]
        y_coord = indices % y.shape[3]

        xy_pred = torch.stack(
            [
                torch.stack(
                    [delta_xy_pred[batch_idx, :, idx] for idx in indices[batch_idx]]
                )
                for batch_idx in range(batch_size)
            ]
        )

        x_coord = (x_coord + 0.5) * x.shape[2] / y.shape[2] - xy_pred[:, :, 0] * 2
        y_coord = (y_coord + 0.5) * x.shape[3] / y.shape[3] - xy_pred[:, :, 1] * 2

        xy_out = torch.stack([x_coord, y_coord], dim=2)

        x_coord_flipped = (x_coord - 64) * -1
        y_coord_flipped = (y_coord - 64) * -1
        xy = torch.stack(
            [
                x_coord,
                y_coord,
                x_coord_flipped,
                y_coord,
                x_coord,
                y_coord_flipped,
                x_coord_flipped,
                y_coord_flipped,
            ],
            dim=2,
        )

        # xy_norm = torch.stack(
        #    [x_coord / x.shape[2] - 0.5, y_coord / x.shape[3] - 0.5], dim=2
        # )

        times = torch.arange(0, timesteps, device=x.device, dtype=torch.int32)
        times = torch.unsqueeze(times.repeat(true_batch_size), -1).repeat(
            1, self.num_entities
        )
        times = times.view(true_batch_size, timesteps, latents.shape[1], -1)

        new_latents = latents.view(true_batch_size, timesteps, latents.shape[1], -1)
        positions = xy.view(true_batch_size, timesteps, xy.shape[1], -1)

        new_latents, attention = self.time_attention(new_latents, positions, times)

        if self.attention:
            latents = new_latents

        # repeat latents to match the size of input image
        # (batch_size, num_entities, latent_dim, 1, 1) -> (batch_size, num_entities, latent_dim, x.shape[2], x.shape[3]
        latents = latents[:, :, :, None, None].repeat(1, 1, 1, x.shape[2], x.shape[3])

        # grid channel is a tensor of size (batch_size, latent_dim, 2, x.shape[2], x.shape[3])
        # it contains the x and y coordinates of each pixel in the image
        grid_channel = torch.stack([x_grid, y_grid])[None, None]
        grid_channel = grid_channel.repeat(batch_size, self.num_entities, 1, 1, 1)

        # center the grid channel around the coordinates of the entities
        x_coord = indices // y.shape[3]
        y_coord = indices % y.shape[3]

        # add 0.5 to the coordinates to center the grid channel around the coordinates of the entities
        # scale the coordinates to match the size of the input image
        x_coord = (x_coord + 0.5) * x.shape[2] / y.shape[2]
        y_coord = (y_coord + 0.5) * x.shape[3] / y.shape[3]
        xy_coord = torch.stack([x_coord, y_coord], dim=2)[..., None, None]
        grid_channel = grid_channel - xy_coord

        # add the predicted delta_xy to the grid channel
        pred_delta_xy = torch.stack(
            [
                torch.stack(
                    [delta_xy_pred[batch_idx, :, idx] for idx in indices[batch_idx]]
                )
                for batch_idx in range(batch_size)
            ]
        )[..., None, None]
        grid_channel = grid_channel + pred_delta_xy * 2

        r_channel = torch.sqrt(torch.sum(grid_channel**2, dim=2, keepdim=True))
        # r_channel = torch.norm(grid_channel, p=2, dim=-1)
        mask = r_channel < self.object_radius

        # concatenate the grid channel and the latents
        # (batch_size, num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        intermediate_feature_map = torch.cat(
            [latents, r_channel / 4, grid_channel / 4], dim=2
        )

        # flatten batch and num_entities dimensions
        # (batch_size * num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        intermediate_feature_map = intermediate_feature_map.view(
            batch_size * self.num_entities, -1, x.shape[2], x.shape[3]
        )

        decoded_feature_map = self.decoder_1(intermediate_feature_map)

        # unflatten batch and num_entities dimensions
        # (batch_size, num_entities, 32, x.shape[2], x.shape[3])
        decoded_feature_map = decoded_feature_map.view(
            batch_size, self.num_entities, -1, x.shape[2], x.shape[3]
        )

        # Multiply each objects feature map with presence
        decoded_feature_map *= pres * mask

        # reduce entity dimension by taking max
        if self.combine_method == "max":
            y = torch.max(decoded_feature_map, dim=1)[0]
        elif self.combine_method == "sum":
            y = torch.sum(decoded_feature_map, dim=1)
        else:
            raise NotImplementedError

        if not self.single_decoder:
            y = self.decoder_2(y)

        y = y.view(true_batch_size, timesteps, x.shape[1], x.shape[2], x.shape[3])

        # presence_loss = z_pres**2 * self.presence_bias

        presence_loss = (z_pres - 1) ** 2 * self.presence_bias

        kl_divergence += presence_loss

        # xy = xy_coord - pred_delta_xy

        return [
            y,
            indices,
            torch.squeeze(pres),
            kl_divergence,
            delta_xy_pred,
            mu,
            logvar,
            attention,
            xy_out,
        ]

    def loss_function(self, x, x_hat, kl_divergence, mu, logvar):
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(x_hat, x)
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


class OldMultiEntityVariationalAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        num_entities,
        attention_model=None,
        beta=0.1,
        latent_dim=12,
        attention=True,
        aggregation="max",
    ):
        super(OldMultiEntityVariationalAutoEncoder, self).__init__()

        self.num_entities = num_entities
        self.beta = beta
        self.latent_dim = latent_dim
        self.attention = attention
        self.aggregation = aggregation

        self.presence_bias = 1
        actual_latent_dim = latent_dim * 2 + 1

        self.time_attention = attention_model
        if self.time_attention == None:
            self.time_attention = MultiheadAttention(num_filters=latent_dim + 1)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, actual_latent_dim, 3, padding=1),
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(self.latent_dim + 1, 32, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, padding=0),
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, padding=0),
        )

    def forward(self, x):
        true_batch_size = x.shape[0]
        timesteps = x.shape[1]

        batch_size = true_batch_size * timesteps

        x = x.view(batch_size, x.shape[2], x.shape[3], x.shape[4])

        y = self.encoder(x)

        z_what = y[:, 1:]

        mu = z_what[:, : self.latent_dim, :, :]
        logvar = z_what[:, self.latent_dim :, :, :]

        x_range = torch.arange(0, x.shape[3], device=x.device, dtype=torch.float32)
        y_range = torch.arange(0, x.shape[2], device=x.device, dtype=torch.float32)
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

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu**2 - std**2, dim=1)
        kl_divergence = kl_divergence.view(batch_size, -1)
        # find the i,j indices of the max value num_entities elements in the kl_divergence tensorimage_size
        # these indices will be used to select the num_entities most important entities

        top_kl_divergence, indices = torch.topk(kl_divergence, self.num_entities, dim=1)
        indices = indices.detach()
        intermediate_feature_map = None

        latents = torch.stack(
            [
                torch.stack(
                    [parametrization[batch_idx, :, idx] for idx in indices[batch_idx]]
                )
                for batch_idx in range(batch_size)
            ]
        )

        # NEWLY ADDED
        x_coord = indices // y.shape[3]
        y_coord = indices % y.shape[3]

        xy_pred = torch.stack(
            [
                torch.stack(
                    [delta_xy_pred[batch_idx, :, idx] for idx in indices[batch_idx]]
                )
                for batch_idx in range(batch_size)
            ]
        )

        x_coord = (x_coord + 0.5) * x.shape[2] / y.shape[2] - xy_pred[:, :, 0] * 2
        y_coord = (y_coord + 0.5) * x.shape[3] / y.shape[3] - xy_pred[:, :, 1] * 2

        # x_coord /= x.shape[2]
        # y_coord /= x.shape[3]

        xy_coords = torch.stack([x_coord, y_coord], dim=2)

        # expanded_latents = torch.cat([latents, xy_coords], dim=2)

        expanded_latents = latents

        times = torch.arange(0, timesteps, device=x.device, dtype=torch.int32)
        times = torch.unsqueeze(times.repeat(true_batch_size), -1).repeat(
            1, self.num_entities
        )

        times = times.view(true_batch_size, timesteps, expanded_latents.shape[1], -1)
        new_latents = expanded_latents.view(
            true_batch_size, timesteps, expanded_latents.shape[1], -1
        )

        positions = xy_coords.view(true_batch_size, timesteps, xy_coords.shape[1], -1)

        new_latents, attention = self.time_attention(new_latents, positions, times)

        # repeat latents to match the size of input image
        # (batch_size, num_entities, latent_dim, 1, 1) -> (batch_size, num_entities, latent_dim, x.shape[2], x.shape[3]
        latents = latents[:, :, :, None, None].repeat(1, 1, 1, x.shape[2], x.shape[3])

        # grid channel is a tensor of size (batch_size, latent_dim, 2, x.shape[2], x.shape[3])
        # it contains the x and y coordinates of each pixel in the image
        grid_channel = torch.stack([x_grid, y_grid])[None, None]
        grid_channel = grid_channel.repeat(batch_size, self.num_entities, 1, 1, 1)

        # center the grid channel around the coordinates of the entities
        x_coord = indices // y.shape[3]
        y_coord = indices % y.shape[3]

        # add 0.5 to the coordinates to center the grid channel around the coordinates of the entities
        # scale the coordinates to match the size of the input image
        x_coord = (x_coord + 0.5) * x.shape[2] / y.shape[2]
        y_coord = (y_coord + 0.5) * x.shape[3] / y.shape[3]
        xy_coord = torch.stack([x_coord, y_coord], dim=2)[..., None, None]
        grid_channel = grid_channel - xy_coord

        # add the predicted delta_xy to the grid channel
        pred_delta_xy = torch.stack(
            [
                torch.stack(
                    [delta_xy_pred[batch_idx, :, idx] for idx in indices[batch_idx]]
                )
                for batch_idx in range(batch_size)
            ]
        )[..., None, None]
        grid_channel = grid_channel + pred_delta_xy * 2

        r_channel = torch.sqrt(torch.sum(grid_channel**2, dim=2, keepdim=True))

        # concatenate the grid channel and the latents
        # (batch_size, num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        intermediate_feature_map = torch.cat(
            [latents, r_channel / 4, grid_channel / 4], dim=2
        )

        # flatten batch and num_entities dimensions
        # (batch_size * num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        intermediate_feature_map = intermediate_feature_map.view(
            batch_size * self.num_entities, -1, x.shape[2], x.shape[3]
        )

        decoded_feature_map = self.decoder_1(intermediate_feature_map)

        # unflatten batch and num_entities dimensions
        # (batch_size, num_entities, 32, x.shape[2], x.shape[3])
        decoded_feature_map = decoded_feature_map.view(
            batch_size, self.num_entities, -1, x.shape[2], x.shape[3]
        )

        # reduce entity dimension by taking max
        if self.aggregation == "sum":
            combined_entities = torch.sum(decoded_feature_map, dim=1)
        elif self.aggregation == "max":
            combined_entities = torch.max(decoded_feature_map, dim=1)[0]
        else:
            raise NotImplementedError

        y = self.decoder_2(combined_entities)

        y = y.view(true_batch_size, timesteps, x.shape[1], x.shape[2], x.shape[3])

        xy = xy_coord - pred_delta_xy

        return [y, indices, kl_divergence, delta_xy_pred, mu, logvar, attention, xy]

    def loss_function(self, x, x_hat, kl_divergence, mu, logvar):
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(x_hat, x)
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


class MEVAE(pl.LightningModule):
    def __init__(
        self,
        num_entities,
        attention_model=None,
        beta=0.1,
        latent_dim=12,
        attention=True,
        combine_method="sum",
        object_radius=12,
        use_old=False,
        single_decoder=True,
    ):
        super().__init__()
        self.use_old = use_old

        if use_old:
            self.model = OldMultiEntityVariationalAutoEncoder(
                num_entities,
                attention_model,
                beta,
                latent_dim=latent_dim,
                attention=attention,
            )
        else:
            self.model = MultiEntityVariationalAutoEncoder(
                num_entities,
                attention_model,
                beta,
                latent_dim=latent_dim,
                attention=attention,
                combine_method=combine_method,
                object_radius=object_radius,
                single_decoder=single_decoder,
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, batch_idx):
        if self.use_old:
            (x_hat, indices, kl_divergence, xy_pred, mu, logvar, attention, xy) = self(
                x
            )
            kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]

            kl_divergence = torch.stack(kl_divergence)
            loss_dict = self.model.loss_function(x, x_hat, kl_divergence, mu, logvar)
            self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
            return loss_dict["loss"]

        else:
            (
                x_hat,
                indices,
                z_pres,
                kl_divergence,
                xy_pred,
                mu,
                logvar,
                attention,
                xy,
            ) = self(x)
            kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]

            kl_divergence = torch.stack(kl_divergence)
            loss_dict = self.model.loss_function(x, x_hat, kl_divergence, mu, logvar)
            self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
            return loss_dict["loss"]

    def validation_step(self, x, batch_idx):
        if self.use_old:
            (
                x_hat,
                indices,
                kl_divergence,
                xy_pred,
                mu,
                logvar,
                attention,
                xy,
            ) = self(x)
            kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]

            kl_divergence = torch.stack(kl_divergence)
            loss_dict = self.model.loss_function(x, x_hat, kl_divergence, mu, logvar)
            self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
            return loss_dict["loss"]
        else:
            (
                x_hat,
                indices,
                z_pres,
                kl_divergence,
                xy_pred,
                mu,
                logvar,
                attention,
                xy,
            ) = self(x)
            kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]

            kl_divergence = torch.stack(kl_divergence)

            loss_dict = self.model.loss_function(x, x_hat, kl_divergence, mu, logvar)
            # self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
            return loss_dict["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)
