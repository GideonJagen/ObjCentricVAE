import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F

from objVAE.fg import objFG
from objVAE.bg import objBG


class objVAE(pl.LightningModule):
    def __init__(
        self,
        fg_model=None,
        bg_model=None,
        num_entities=16,
        num_fg_iterations=2,
        beta=0.1,
        learning_rate=1e-4,
        lossf="mse",
    ):
        super(objVAE, self).__init__()
        if fg_model is None:
            fg_model = objFG(num_entities=num_entities)
        if bg_model is None:
            bg_model = objBG()
        self.fg_model = fg_model
        self.bg_model = bg_model
        self.beta = beta
        self.learning_rate = learning_rate
        self.lossf = lossf
        self.num_fg_iterations = num_fg_iterations

    def forward(self, x):
        # (B, C, H, W), (B,)
        bg, z_bg, kl_bg = self.bg_model(x)

        # (B, C, H, W), (B, K, hidden_dim), (B, K, 3), (B, K)
        # for iteration in range(self.num_fg_iterations):
        fg, indices, kl_divergence, presence, xy, z_fg = self.fg_model(x - bg)

        fgs = [fg]

        kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]
        for iteration in range(self.num_fg_iterations - 1):
            _fg, _indices, _kl_divergence, _presence, _xy, _z_fg = self.fg_model(
                x - bg - fg
            )
            fg = fg + _fg
            kl_divergence = kl_divergence + [
                kld[idx] for kld, idx in zip(_kl_divergence, _indices)
            ]
            presence = torch.cat([presence, _presence], dim=1)
            xy = torch.cat([xy, _xy], dim=1)
            z_fg = torch.cat([z_fg, _z_fg], dim=1)

            fgs.append(_fg)

        kl_fg = torch.stack(kl_divergence)

        reconstruction = bg + fg

        return reconstruction, presence, bg, fgs, xy, z_fg, kl_bg, kl_fg

    def reconstruct(self, x):
        return self(x)[0]

    def training_step(self, batch, batch_idx):
        x = batch

        reconstruction, bg, fgs, z_what, z_where, z_bg, kl_bg, kl_fg = self(x)

        kl_bg = kl_bg.mean()
        kl_fg = kl_fg.mean()

        kl = kl_fg + kl_bg

        recon_loss = F.mse_loss(reconstruction, x)
        if self.lossf == "mse":
            rloss = recon_loss
        elif self.lossf == "mae":
            rloss = F.l1_loss(reconstruction, x)

        loss = rloss + self.beta * kl

        self.log_dict(
            {
                "loss": loss,
                "reconstruction": recon_loss,
                "kl": self.beta * kl,
                "kl_bg": kl_bg,
                "kl_fg": kl_fg,
            },
            on_step=True,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class objVAEBG(pl.LightningModule):
    def __init__(
        self,
        fg_model=None,
        bg_model=None,
        num_entities=16,
        beta=0.1,
        learning_rate=1e-4,
        lossf="mse",
    ):
        super(objVAEBG, self).__init__()
        # if fg_model is None:
        #     fg_model = objFG(num_entities=num_entities)
        if bg_model is None:
            bg_model = objBG()
        self.fg_model = fg_model
        self.bg_model = bg_model
        self.beta = beta
        self.learning_rate = learning_rate
        self.lossf = lossf

    def forward(self, x):
        # (B, C, H, W), (B,)
        bg, z_bg, kl_bg = self.bg_model(x)
        # z_bg = kl_bg = bg = torch.tensor(0)

        # (B, C, H, W), (B, K, hidden_dim), (B, K, 3), (B, K)
        # fg, indices, kl_divergence, xy, mu, weightmap = self.fg_model(x)
        # kl_divergence = [kld[idx] for kld, idx in zip(kl_divergence, indices)]
        # kl_fg = torch.stack(kl_divergence)

        # loss_dict = self.model.loss_function(x, fg, kl_divergence, mu, logvar)
        # self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
        # return loss_dict["loss"]

        # Treat fg as residual. For microscopy, this is a good assumption.
        # reconstruction = bg + fg
        reconstruction = bg

        return reconstruction, kl_bg

    def reconstruct(self, x):
        return self(x)[0]

    def training_step(self, batch, batch_idx):
        x = batch

        reconstruction, kl_bg = self(x)

        # kl = kl_bg.mean() * 0.0 + kl_fg.sum(dim=1).mean()
        kl = kl_bg.mean()

        recon_loss = F.mse_loss(reconstruction, x)
        if self.lossf == "mse":
            rloss = recon_loss
        elif self.lossf == "mae":
            rloss = F.l1_loss(reconstruction, x)
        weight_MN = x.shape[1] * x.shape[2] * x.shape[3] / 12

        loss = rloss + self.beta * kl * weight_MN

        self.log_dict(
            {
                "loss": loss,
                "reconstruction": recon_loss,
                "kl": kl,
                "kl_bg": kl_bg.mean(),
                # "kl_fg": kl_fg.mean(),
            },
            on_step=True,
            on_epoch=True,
        )

        return loss

    test_step = training_step
    validation_step = training_step

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
