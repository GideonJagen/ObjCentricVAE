import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        topk_select_method="max",
        object_radius=12,
        decoder="linear",
        decoder_feature_size=56,
        decoder_num_layers=3,
        encoder_pooling_layers=3,
        encoder_conv_layers=3,
        glimpse_size=32,
    ):
        super(MultiEntityVariationalAutoEncoder, self).__init__()

        self.num_entities = num_entities
        self.beta = beta
        self.latent_dim = latent_dim
        self.attention = attention
        self.combine_method = combine_method
        self.object_radius = object_radius
        self.single_decoder = single_decoder
        self.topk_select_method = topk_select_method
        self.decoder = decoder
        self.glimpse_size = glimpse_size

        self.presence_bias = 1
        self.kl_importance = 0
        self.position_representation_scale = 1
        self.position_prediction_scale = 1

        actual_latent_dim = latent_dim * 2 + 1
        decoder_latent_dim = latent_dim

        self.use_radial = False

        self.time_attention = attention_model
        if self.time_attention is None:
            self.time_attention = MultiheadAttention(num_filters=latent_dim + 1)

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
                for _ in range(encoder_pooling_layers - 1)
            ],
            *[
                nn.Sequential(
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                )
                for _ in range(encoder_conv_layers)
            ],
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, actual_latent_dim, 3, padding=1),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(actual_latent_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                )
                for _ in range(encoder_conv_layers)
            ],
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                )
                for _ in range(encoder_pooling_layers - 1)
            ],
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 1, 3, padding=1),
        )

        def decoder_block(in_filters, out_filters):
            return nn.Sequential(
                nn.Linear(in_filters, out_filters),
                nn.ReLU(),
            )

        self.decoder_linear = nn.Sequential(
            *[
                decoder_block(
                    decoder_latent_dim + 1 if self.use_radial else decoder_latent_dim,
                    decoder_feature_size,
                ),
                *[
                    decoder_block(
                        decoder_feature_size,
                        decoder_feature_size,
                    )
                    for _ in range(decoder_num_layers - 1)
                ],
            ],
            nn.Linear(decoder_feature_size, 1),
        )

    def forward(self, x, training=False):
        true_batch_size = x.shape[0]
        timesteps = x.shape[1]
        batch_size = true_batch_size * timesteps

        x = x.view(batch_size, x.shape[2], x.shape[3], x.shape[4])

        y = self.encoder(x)

        x_mass = torch.abs(x)

        # resize x_mass
        x_mass = F.interpolate(
            x_mass,
            size=(y.shape[2], y.shape[3]),
            mode="bilinear",
            align_corners=True,
        )

        z_pres = torch.sigmoid(y[:, :1]).view(batch_size, -1)
        z_what = y[:, 1:]

        mu = z_what[:, : self.latent_dim, :, :]
        logvar = z_what[:, self.latent_dim :, :, :]

        x_range = torch.arange(0, x.shape[2], device=x.device, dtype=torch.float32)
        y_range = torch.arange(0, x.shape[3], device=x.device, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x_range, y_range)

        std = torch.exp(0.5 * logvar)

        # Reparametrization
        if training:
            parametrization = torch.randn_like(std) * std + mu
        else:
            parametrization = mu

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

        score = kl_divergence**self.kl_importance * x_mass.view(batch_size, -1) ** (
            1 - self.kl_importance
        )
        # **self.kl_importance * x_mass.view(batch_size, -1) ** (
        #    1 - self.kl_importance
        # )
        if self.topk_select_method == "random":
            indices = self.sample_topk(score)
        elif self.topk_select_method == "max":
            indices = self.select_topk(score)
        else:
            raise NotImplementedError

        # indices = indices.detach()
        intermediate_feature_map = None

        latents = parametrization[torch.arange(batch_size)[:, None], :, indices]
        xy_pred = delta_xy_pred[torch.arange(batch_size)[:, None], :, indices]
        pres = z_pres[torch.arange(batch_size)[:, None], indices].view(
            batch_size, -1, 1, 1, 1
        )

        # Create positial embeddings for attention
        # center the grid channel around the coordinates of the entities
        x_coord = indices // y.shape[3]
        y_coord = indices % y.shape[3]

        # add 0.5 to the coordinates to center the grid channel around the coordinates of the entities
        # scale the coordinates to match the size of the input image and add predicted offset
        x_coord = (x_coord + 0.5) * x.shape[2] / y.shape[2] - xy_pred[
            :, :, 0
        ] * self.position_prediction_scale
        y_coord = (y_coord + 0.5) * x.shape[3] / y.shape[3] - xy_pred[
            :, :, 1
        ] * self.position_prediction_scale

        # stack x and y aswell as x and y flipped
        positional_embeddings = torch.stack(
            [
                x_coord,
                y_coord,
                (x_coord - x.shape[2]) * -1,
                (y_coord - x.shape[3]) * -1,
            ],
            dim=2,
        )

        # Add time of every frame to be for masking in attention
        times = torch.arange(0, timesteps, device=x.device, dtype=torch.int32)
        times = torch.unsqueeze(times.repeat(true_batch_size), -1).repeat(
            1, self.num_entities
        )
        times = times.view(true_batch_size, timesteps, latents.shape[1], -1)

        # Self attention between latents and latents one time step in the future and past
        assert not torch.isnan(latents).any()

        new_latents = latents.view(true_batch_size, timesteps, latents.shape[1], -1)
        positional_embeddings = positional_embeddings.view(
            true_batch_size, timesteps, positional_embeddings.shape[1], -1
        )

        new_latents, new_pos, attention = self.time_attention(
            new_latents, positional_embeddings, times
        )

        if self.attention:
            x_coord = new_pos[:, :, 0]
            y_coord = new_pos[:, :, 1]
            latents = new_latents

        # repeat latents to match the size of input image
        # (batch_size, num_entities, latent_dim, 1, 1) -> (batch_size, num_entities, latent_dim, x.shape[2], x.shape[3]
        latents = latents[:, :, None, :].repeat(1, 1, x.shape[2] * x.shape[3], 1)

        # grid channel is a tensor of size (batch_size, latent_dim, 2, x.shape[2], x.shape[3])
        # it contains the x and y coordinates of each pixel in the image
        grid_channel = torch.stack([x_grid, y_grid], dim=-1).view(1, 1, -1, 2)

        xy_coord = torch.stack([x_coord, y_coord], dim=-1).view(batch_size, -1, 1, 2)

        grid_channel = (grid_channel - xy_coord) * self.position_representation_scale

        r_channel = torch.norm(grid_channel, p=2, dim=-1)
        mask = r_channel < (self.glimpse_size * self.position_representation_scale)

        reduced_grid = grid_channel[mask, :]
        reduced_latents = latents[mask, :]

        # if self.position_embedding == "none":
        #    ...
        # elif self.position_embedding == "radial":
        if self.use_radial:
            r_channel = torch.norm(reduced_grid, p=2, dim=-1, keepdim=True)
            reduced_grid = torch.cat([reduced_grid, r_channel], dim=-1)
        # elif self.position_embedding == "sine":
        # reduced_grid = self.sine_embedding(reduced_grid)

        # concatenate the grid channel and the latents
        # (batch_size, num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        intermediate_feature_map = torch.cat([reduced_latents, reduced_grid], dim=-1)

        # flatten batch and num_entities dimensions
        # (batch_size * num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        # intermediate_feature_map = intermediate_feature_map.view(
        #    batch_size * self.num_entities, -1, x.shape[2], x.shape[3]
        # )

        if self.decoder == "linear":
            pixel_map = self.decoder_linear(intermediate_feature_map)
        elif self.decoder == "conv":
            pixel_map = self.decoder_conv(intermediate_feature_map)
        else:
            raise NotImplementedError

        pixel_map_tensor = torch.zeros(
            batch_size, self.num_entities, x.shape[2] * x.shape[3], device=x.device
        )

        pixel_map_tensor[mask] = pixel_map.view(-1)

        pixel_map = pixel_map_tensor.view(
            batch_size, self.num_entities, 1, x.shape[2], x.shape[3]
        )

        weight_map = (
            mask.view(batch_size, self.num_entities, 1, x.shape[2], x.shape[3]) * pres
        )

        """
        # unflatten batch and num_entities dimensions
        # (batch_size, num_entities, 32, x.shape[2], x.shape[3])
        decoded_feature_map = decoded_feature_map.view(
            batch_size, self.num_entities, -1, x.shape[2], x.shape[3]
        )
        """

        # Multiply each objects feature map with presence
        decoded_feature_map = pixel_map * weight_map

        # reduce entity dimension by combining feature maps by layer with ordering based on pres value
        if self.combine_method == "max":
            y = torch.max(decoded_feature_map, dim=1)[0]
        elif self.combine_method == "min":
            y = torch.min(decoded_feature_map, dim=1)[0]
        elif self.combine_method == "sum":
            y = torch.sum(decoded_feature_map, dim=1)
        elif self.combine_method == "ordered":
            # sort feature maps by pres value in descending order
            sorted_indices = torch.argsort(pres, dim=1, descending=True)
            sorted_feature_maps = torch.gather(
                decoded_feature_map,
                1,
                sorted_indices.unsqueeze(-1).repeat(
                    1, 1, decoded_feature_map.shape[-2], decoded_feature_map.shape[-1]
                ),
            )
            # combine feature maps by layer
            y = torch.cat(sorted_feature_maps, dim=1)
        else:
            raise NotImplementedError

        if not self.single_decoder:
            y = self.decoder_2(y)

        y = y.view(true_batch_size, timesteps, x.shape[1], x.shape[2], x.shape[3])

        presence_loss = (z_pres - 1) ** 2

        return [
            y,
            indices,
            torch.squeeze(pres),
            kl_divergence,
            presence_loss,
            delta_xy_pred,
            mu,
            logvar,
            attention,
            torch.squeeze(xy_coord),
        ]

    def loss_function(self, x, x_hat, kl_divergence, presence_loss, mu, logvar):
        # reconstruction_loss = torch.mean(torch.abs(error))
        recon_loss = F.mse_loss(x_hat, x)

        # KL divergence & presence loss
        kl_divergence_loss = kl_divergence.mean() * self.beta
        presence_loss_mean = presence_loss.mean() * self.presence_bias
        # MN weighting
        # weight_MN = x.shape[1] * x.shape[2] * x.shape[3] / 12

        # Total loss
        loss = recon_loss + presence_loss_mean + kl_divergence_loss

        return {
            "loss": loss,
            "reconstruction_loss": recon_loss,
            "KLD": kl_divergence_loss,
            "weighted_KLD": kl_divergence_loss * self.beta,
            "presence_loss": presence_loss_mean,
            "weighted_presence_loss": presence_loss_mean * self.presence_bias,
        }

    def select_topk(self, score, topk=None):
        """Select latent vectors based on score"""
        if topk is None:
            topk = self.num_entities
        score, indices = torch.topk(score, topk, dim=1)
        indices = indices.detach()
        return indices

    def sample_topk(self, score, topk=None):
        """Randomly select latent vectors based on score"""
        if topk is None:
            topk = self.num_entities

        # normalize kl_divergence as a probability distribution
        score = F.softmax(score, dim=1)

        # replace nan with 0
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)

        # replace inf with 0
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)

        # replace negative values with 0
        score = torch.where(score < 0, torch.zeros_like(score), score)

        # renormalize kl_divergence as a probability distribution
        score = score / torch.sum(score, dim=1, keepdim=True)

        # randomly sample topk indices from kl_divergence

        indices = [
            torch.multinomial(score[b], topk, replacement=False)
            for b in range(score.shape[0])
        ]
        indices = torch.stack(indices)
        indices = indices.detach()
        return indices

    def extract_latents(self, x):
        true_batch_size = x.shape[0]
        timesteps = x.shape[1]
        batch_size = true_batch_size * timesteps

        x = x.view(batch_size, x.shape[2], x.shape[3], x.shape[4])

        y = self.encoder(x)

        x_mass = torch.abs(x)

        # resize x_mass
        x_mass = F.interpolate(
            x_mass,
            size=(y.shape[2], y.shape[3]),
            mode="bilinear",
            align_corners=True,
        )

        z_pres = torch.sigmoid(y[:, :1]).view(batch_size, -1)
        z_what = y[:, 1:]

        mu = z_what[:, : self.latent_dim, :, :]
        logvar = z_what[:, self.latent_dim :, :, :]

        x_range = torch.arange(0, x.shape[2], device=x.device, dtype=torch.float32)
        y_range = torch.arange(0, x.shape[3], device=x.device, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x_range, y_range)

        std = torch.exp(0.5 * logvar)

        # Reparametrization
        # parametrization = torch.randn_like(std) * std + mu
        # Use Mu as parametrization to remove randomness in the latents
        parametrization = mu

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

        score = kl_divergence**self.kl_importance * x_mass.view(batch_size, -1) ** (
            1 - self.kl_importance
        )
        # **self.kl_importance * x_mass.view(batch_size, -1) ** (
        #    1 - self.kl_importance
        # )
        if self.topk_select_method == "random":
            indices = self.sample_topk(score)
        elif self.topk_select_method == "max":
            indices = self.select_topk(score)
        else:
            raise NotImplementedError

        # indices = indices.detach()
        latents = parametrization[torch.arange(batch_size)[:, None], :, indices]
        xy_pred = delta_xy_pred[torch.arange(batch_size)[:, None], :, indices]
        pres = z_pres[torch.arange(batch_size)[:, None], indices].view(
            batch_size, -1, 1, 1, 1
        )

        # Create positial embeddings for attention
        # center the grid channel around the coordinates of the entities
        x_coord = indices // y.shape[3]
        y_coord = indices % y.shape[3]

        # add 0.5 to the coordinates to center the grid channel around the coordinates of the entities
        # scale the coordinates to match the size of the input image and add predicted offset
        x_coord = (x_coord + 0.5) * x.shape[2] / y.shape[2] - xy_pred[
            :, :, 0
        ] * self.position_prediction_scale
        y_coord = (y_coord + 0.5) * x.shape[3] / y.shape[3] - xy_pred[
            :, :, 1
        ] * self.position_prediction_scale

        # stack x and y aswell as x and y flipped
        positional_embeddings = torch.stack(
            [
                x_coord,
                y_coord,
                (x_coord - x.shape[2]) * -1,
                (y_coord - x.shape[3]) * -1,
            ],
            dim=2,
        )

        # Add time of every frame to be for masking in attention
        times = torch.arange(0, timesteps, device=x.device, dtype=torch.int32)
        times = torch.unsqueeze(times.repeat(true_batch_size), -1).repeat(
            1, self.num_entities
        )
        times = times.view(true_batch_size, timesteps, latents.shape[1], -1)

        # Self attention between latents and latents one time step in the future and past
        new_latents = latents.view(true_batch_size, timesteps, latents.shape[1], -1)
        positional_embeddings = positional_embeddings.view(
            true_batch_size, timesteps, positional_embeddings.shape[1], -1
        )

        new_latents, new_pos, attention = self.time_attention(
            new_latents, positional_embeddings, times
        )

        xy_coord = torch.stack([x_coord, y_coord], dim=-1).view(batch_size, -1, 1, 2)

        return latents, new_latents, pres, torch.squeeze(xy_coord)

    def decode_from_latents(self, latents, image_shape, grid_channel):
        batch_size = latents.shape[0]

        # repeat latents to match the size of input image
        # (batch_size, num_entities, latent_dim, 1, 1) -> (batch_size, num_entities, latent_dim, x.shape[2], x.shape[3]
        latents = latents[:, :, None, :].repeat(
            1, 1, image_shape[0] * image_shape[1], 1
        )

        # grid channel is a tensor of size (batch_size, latent_dim, 2, x.shape[2], x.shape[3])
        # it contains the x and y coordinates of each pixel in the image
        # grid_channel = torch.stack([x_grid, y_grid], dim=-1).view(1, 1, -1, 2)

        # xy_coord = torch.stack([x_coord, y_coord], dim=-1).view(batch_size, -1, 1, 2)

        # grid_channel = (grid_channel - xy_coord) * self.position_representation_scale

        r_channel = torch.norm(grid_channel, p=2, dim=-1)
        mask = r_channel < (self.glimpse_size * self.position_representation_scale)

        reduced_grid = grid_channel[mask, :]
        reduced_latents = latents[mask, :]

        # if self.position_embedding == "none":
        #    ...
        # elif self.position_embedding == "radial":
        if self.use_radial:
            r_channel = torch.norm(reduced_grid, p=2, dim=-1, keepdim=True)
            reduced_grid = torch.cat([reduced_grid, r_channel], dim=-1)
        # elif self.position_embedding == "sine":
        # reduced_grid = self.sine_embedding(reduced_grid)

        # concatenate the grid channel and the latents
        # (batch_size, num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        intermediate_feature_map = torch.cat([reduced_latents, reduced_grid], dim=-1)

        # flatten batch and num_entities dimensions
        # (batch_size * num_entities, latent_dim + 2, x.shape[2], x.shape[3])
        # intermediate_feature_map = intermediate_feature_map.view(
        #    batch_size * self.num_entities, -1, x.shape[2], x.shape[3]
        # )

        if self.decoder == "linear":
            pixel_map = self.decoder_linear(intermediate_feature_map)
        elif self.decoder == "conv":
            pixel_map = self.decoder_conv(intermediate_feature_map)
        else:
            raise NotImplementedError

        pixel_map_tensor = torch.zeros(
            batch_size,
            pixel_map.shape[1],
            image_shape[0] * image_shape[1],
            device=latents.device,
        )

        pixel_map_tensor[mask] = pixel_map.view(-1)

        pixel_map = pixel_map_tensor.view(
            batch_size, pixel_map.shape[1], 1, image_shape[0], image_shape[1]
        )

        weight_map = mask.view(
            batch_size, pixel_map.shape[1], 1, image_shape[0], image_shape[1]
        )

        """
        # unflatten batch and num_entities dimensions
        # (batch_size, num_entities, 32, x.shape[2], x.shape[3])
        decoded_feature_map = decoded_feature_map.view(
            batch_size, self.num_entities, -1, x.shape[2], x.shape[3]
        )
        """

        # Multiply each objects feature map with presence
        decoded_feature_map = pixel_map * weight_map

        # reduce entity dimension by taking max
        if self.combine_method == "max":
            y = torch.max(decoded_feature_map, dim=1)[0]
        elif self.combine_method == "min":
            y = torch.min(decoded_feature_map, dim=1)[0]
        elif self.combine_method == "sum":
            y = torch.sum(decoded_feature_map, dim=1)
        else:
            raise NotImplementedError

        if not self.single_decoder:
            y = self.decoder_2(y)

        y = y.view(batch_size, image_shape[0], image_shape[1])

        return y


class MEVAE(pl.LightningModule):
    def __init__(
        self,
        num_entities,
        attention_model=None,
        beta=0.1,
        latent_dim=12,
        attention=True,
        combine_method="sum",
        topk_select_method="max",
        object_radius=12,
        single_decoder=True,
        decoder="linear",
        decoder_feature_size=56,
        decoder_num_layers=3,
        encoder_pooling_layers=3,
        encoder_conv_layers=3,
        glimpse_size=32,
    ):
        super().__init__()

        self.num_entities = num_entities

        self.model = MultiEntityVariationalAutoEncoder(
            num_entities,
            attention_model,
            beta,
            latent_dim=latent_dim,
            attention=attention,
            combine_method=combine_method,
            topk_select_method=topk_select_method,
            object_radius=object_radius,
            single_decoder=single_decoder,
            decoder=decoder,
            decoder_feature_size=decoder_feature_size,
            decoder_num_layers=decoder_num_layers,
            encoder_pooling_layers=encoder_pooling_layers,
            encoder_conv_layers=encoder_conv_layers,
            glimpse_size=glimpse_size,
        )

    def forward(self, x, training=False):
        return self.model(x, training=training)

    def training_step(self, x, batch_idx):
        (
            x_hat,
            indices,
            z_pres,
            kl_divergence,
            presence_loss,
            xy_pred,
            mu,
            logvar,
            attention,
            xy,
        ) = self(x, training=True)
        # kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]
        presence_loss = [pl[idx] for pl, idx in zip(presence_loss, indices)]

        # kl_divergence = torch.stack(kl_divergence)
        presence_loss = torch.stack(presence_loss)
        loss_dict = self.model.loss_function(
            x, x_hat, kl_divergence, presence_loss, mu, logvar
        )
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]

    def validation_step(self, x, batch_idx):
        (
            x_hat,
            indices,
            z_pres,
            kl_divergence,
            presence_loss,
            xy_pred,
            mu,
            logvar,
            attention,
            xy,
        ) = self(x, training=False)
        # kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]
        presence_loss = [pl[idx] for pl, idx in zip(presence_loss, indices)]
        presence_loss = torch.stack(presence_loss)
        # kl_divergence = torch.stack(kl_divergence)

        loss_dict = self.model.loss_function(
            x, x_hat, kl_divergence, presence_loss, mu, logvar
        )
        # self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

    # Other methods#
    def _combine_nodes(self, attention_to_next, combine_map_v, remove_map_v, i_combine):
        new_attention = []
        for j, row in enumerate(combine_map_v[i_combine]):
            attention_to_next_dt = attention_to_next.detach().cpu().numpy()
            new_attention_row = np.transpose(np.transpose(attention_to_next_dt) * row)
            new_attention_row = np.sum(new_attention_row, axis=0) / np.sum(row)
            if remove_map_v[i_combine][j]:
                new_attention_row *= 0
            new_attention.append(new_attention_row)
        new_attention = np.array(new_attention)

        updated_attention = new_attention

        new_attention = []
        for j, row in enumerate(combine_map_v[i_combine + 1]):
            attention_to_next_dt = updated_attention
            new_attention_row = attention_to_next_dt * row
            new_attention_row = np.sum(new_attention_row, axis=1) / np.sum(row)
            if remove_map_v[i_combine + 1][j]:
                new_attention_row *= 0
            new_attention.append(new_attention_row)
        new_attention = np.array(np.transpose(new_attention))
        return new_attention

    def _distance(self, x1, x2, y1, y2):
        dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
        return dist

    def extract_obj_and_tra(
        self, sequence, combine_radius=4, presence_floor=0.5, tra_floor=0.99
    ):
        sequence = torch.unsqueeze(sequence, dim=0)
        sequence = sequence.to(self.model.device)
        (
            recon,
            indices,
            pres,
            kl_divergence,
            xy_pred,
            mu,
            logvar,
            attention,
            xy,
        ) = self.model(sequence)

        # x = x.detach().cpu().numpy()
        recon = recon.detach().cpu().numpy()
        pres = pres.detach().cpu().numpy()
        xy = xy.detach().cpu().numpy()

        combine_map_v = []
        remove_map_v = []
        xp_v = []
        yp_v = []
        for i in range(xy.shape[0]):
            xp = xy[i, :, 0]
            yp = xy[i, :, 1]

            xp_v.append(xp)
            yp_v.append(yp)

            distances = []
            for j in range(xp.shape[0]):
                distance_r = []
                for k in range(xp.shape[0]):
                    if j == k:
                        distance_r.append(0)
                        continue
                    dist = self._distance(xp[j], xp[k], yp[j], yp[k])
                    distance_r.append(dist)
                distances.append(distance_r)
            distances = np.array(distances)

            # Create the combin map
            row_indices, col_indices = np.indices(distances.shape)
            matrix = np.zeros(distances.shape)
            matrix[row_indices - col_indices <= 0] = 1

            combine_map = (
                np.where(np.array(distances) < combine_radius, 1, 0).astype(bool)
                & matrix.astype(bool)
            ).astype(np.int32)

            pres_mask = pres[i] <= presence_floor

            remove_map = (
                np.sum(combine_map - np.eye(combine_map.shape[0]), axis=0) + pres_mask
            )

            combine_map_v.append(combine_map)
            remove_map_v.append(remove_map)

        tra_dict_new = {}
        final_tra_list = []
        for i_frame, frame in enumerate(xy):
            tra_dict = tra_dict_new.copy()

            if i_frame == 0:
                for i_obj, obj in enumerate(frame):
                    if not remove_map_v[i_frame][i_obj]:
                        tra_dict_new[i_obj] = self._new_obj_dict(obj, i_frame)
                continue

            # Calculate attention metrics
            binary_attention = self._get_connections(
                attention,
                i_frame,
                self.num_entities,
                combine_map_v,
                remove_map_v,
                tra_floor,
            )
            binary_attention_t = np.transpose(binary_attention)

            visited = set()
            added = set()
            for i_obj, obj in enumerate(frame):
                connections = np.where(binary_attention_t[i_obj] == 1)[0]
                if remove_map_v[i_frame][i_obj] or not connections.size > 0:
                    continue
                obj_dict = tra_dict.get(connections[0])
                if obj_dict:
                    obj_dict["x"].append(obj[0])
                    obj_dict["y"].append(obj[1])
                    obj_dict["code"].append(-1)
                    if not connections[0] in added:
                        del tra_dict_new[connections[0]]
                    tra_dict_new[i_obj] = obj_dict
                    visited.add(connections[0])
                    added.add(i_obj)
                else:
                    tra_dict_new[i_obj] = self._new_obj_dict(obj, i_frame)

            ended = tra_dict.keys() - visited
            for end in ended:
                obj_dict = tra_dict.pop(end)
                obj_dict["end"] = i_frame - 1
                del tra_dict_new[end]
                final_tra_list.append(obj_dict)

        obj_dicts = list(tra_dict_new.values())
        final_tra_list.extend(obj_dicts)
        return final_tra_list

    def _new_obj_dict(self, obj, frame):
        return {"x": [obj[0]], "y": [obj[1]], "code": [-1], "start": frame}

    def _get_connections(
        self, attention, i_frame, num_entities, combine_map, remove_map, tra_floor
    ):
        attention_to_next = attention[
            0,
            (i_frame - 1) * num_entities : i_frame * num_entities,
            i_frame * num_entities : (i_frame + 1) * num_entities,
        ]
        new_attention = self._combine_nodes(
            attention_to_next, combine_map, remove_map, i_frame - 1
        )

        max_indices = np.argmax(new_attention, axis=1)
        binary_attention = np.zeros_like(new_attention)
        binary_attention[np.arange(new_attention.shape[0]), max_indices] = 1
        binary_attention *= np.where(new_attention > 0, 1, 0)
        binary_attention *= np.where(new_attention >= tra_floor, 1, 0)

        return binary_attention
