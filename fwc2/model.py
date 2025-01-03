import numpy
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.uniform import Uniform
import lightning.pytorch as pl
import torchmetrics
from torchmetrics.classification import Accuracy
import pytorch_metric_learning.losses as losses

from fwc2.loss import NTXent, FWC2Loss
from fwc2.utils import find_net_arch

from typing import Optional


class RandomFeatureCorruption:
    def __init__(self,
                 corruption_rate: float = 0.6,
                 corrupt_both_views: bool = True,
                 features_low: numpy.ndarray = None,
                 features_high: numpy.ndarray = None):
        self.corruption_rate = corruption_rate
        self.corrupt_both_views = corrupt_both_views
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))

    def __call__(self, x):
        bs, _ = x.size()

        mask = torch.rand_like(x, device=x.device) <= self.corruption_rate
        x_random = self.marginals.sample(torch.Size((bs,))).to(x.device)

        xc1 = torch.where(mask, x_random, x)

        if self.corrupt_both_views:
            xc2 = torch.where(~mask, x_random, x)
            return xc1, xc2

        return xc1, x


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.0) -> None:
        layers = []
        in_dim = input_dim
        for i in range(len(hidden_dims)):
            hidden_dim = hidden_dims[i]
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class MLPv2(torch.nn.Sequential):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.0) -> None:
        layers = []
        in_dim = input_dim
        for i in range(len(hidden_dims)):
            hidden_dim = hidden_dims[i]
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class LinearLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, batch_norm: bool = False):
        super().__init__()
        self.size_in = input_size
        self.size_out = output_size
        if batch_norm:
            self.model = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
            )
        else:
            self.model = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

    def forward(self, x: torch.tensor):
        return self.model(x)


class LazyMLP(nn.Module):
    def __init__(self, n_layers: int, dim_hidden: int, batch_norm: bool = False):
        super().__init__()
        self.n_layers = n_layers
        self.dim_hidden = dim_hidden
        if batch_norm:
            lazy_block = nn.Sequential(
                nn.LazyLinear(dim_hidden),
                nn.BatchNorm1d(dim_hidden),
                nn.ReLU(),
            )
        else:
            lazy_block = nn.Sequential(nn.LazyLinear(dim_hidden), nn.ReLU())

        self.model = nn.Sequential(
            lazy_block,
            *[LinearLayer(dim_hidden, dim_hidden, batch_norm) for _ in range(n_layers - 1)],
        )

    def forward(self, x: torch.tensor):
        return self.model(x)


class LazyMLP2(nn.Module):
    def __init__(self, hidden_layers: list = [64, 128, 64, 32], activation_fn: nn.Module = nn.ReLU,
                 batch_norm: bool = False):
        super().__init__()
        self.hidden_layers = hidden_layers
        if batch_norm:
            lazy_block = nn.Sequential(
                nn.LazyLinear(hidden_layers[0]),
                nn.BatchNorm1d(hidden_layers[0]),
                activation_fn(),
            )
        else:
            lazy_block = nn.Sequential(nn.LazyLinear(hidden_layers[0]), activation_fn())

        layers = []
        in_dim = hidden_layers[0]
        for hdim in hidden_layers[1:]:
            layers.append(LinearLayer(in_dim, hdim, batch_norm))
            in_dim = hdim

        self.model = nn.Sequential(
            lazy_block,
            *layers,
        )

    def forward(self, x: torch.tensor):
        return self.model(x)


class S2SDEncoder(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            projector: nn.Module,
            loss_fn: nn.Module = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=1.0)),
            corrupt_fn=None,
            lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['corrupt_fn', 'loss_fn', 'encoder', 'projector'])

        self.encoder = encoder
        self.projector = projector
        self.loss_fn = loss_fn
        self.corrupt_fn = corrupt_fn

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def _step(self, x: Tensor):
        x1, x2 = self.corrupt_fn(x)
        h1, h2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))

        return self.loss_fn(h1, h2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch

        loss = self._step(x)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        loss = self._step(x)

        self.log('val_loss', loss, prog_bar=True)

    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class S2SDClassifier(pl.LightningModule):
    def __init__(
            self,
            encoder_ckpt: str,
            dim_hidden: int = 256,
            n_layers: int = 2,
            num_classes: int = 5,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            score_fn: torchmetrics.Metric = torchmetrics.Accuracy(task="multiclass", num_classes=5),
            lr: float = 1e-3
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.encoder = S2SDEncoder.load_from_checkpoint(encoder_ckpt)
        self.loss_fn = loss_fn
        self.score_func = score_fn

        self.classifier = nn.Sequential(
            LazyMLP(n_layers=n_layers, dim_hidden=dim_hidden), nn.Linear(dim_hidden, num_classes)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder.encoder(x)
        return self.classifier(h)

    def _step(self, x: Tensor, y: Tensor):
        predictions = self(x)

        loss = self.loss_fn(predictions, y)
        score = self.score_func(predictions, y)

        return loss, score

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss, score = self._step(x, y)

        metrics = {"train_loss": loss, "train_acc": score}
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        loss, score = self._step(x, y)

        metrics = {"val_loss": loss, "val_acc": score}
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss


# ***************
class FWC2AEv3(pl.LightningModule):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            encoder_layers: int,
            lr: float = 1e-3,
            dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        encoder_arch = find_net_arch(hidden_dim, encoder_layers, factor=2)
        decoder_arch = encoder_arch.reverse()

        print(f'encoder layers = {encoder_arch}')
        print(f'decoder layers = {decoder_arch}')

        self.encoder = MLP(in_dim, encoder_arch, dropout)
        self.decoder = MLP(hidden_dim, decoder_arch, dropout)
        self.loss_fn = nn.MSELoss()

    def _step(self, x: Tensor):
        x_hat = self.decoder(self.encoder(x))

        return self.loss_fn(x, x_hat)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch

        loss = self._step(x)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        loss = self._step(x)

        self.log('val_loss', loss, prog_bar=True)

    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)

# *********************************
# class MaskGenerator(nn.Module):
#     """Module for generating Bernoulli mask."""
#
#     def __init__(self, p: float):
#         super().__init__()
#         self.p = p
#
#     def forward(self, x: torch.tensor):
#         """Generate Bernoulli mask."""
#         p_mat = torch.ones_like(x) * self.p
#         return torch.bernoulli(p_mat)
#
#
# class PretextGenerator(nn.Module):
#     """Module for generating training pretext."""
#
#     def __init__(self):
#         super().__init__()
#
#     @staticmethod
#     def shuffle(x: torch.tensor):
#         """Shuffle each column in a tensor."""
#         m, n = x.shape
#         x_bar = torch.zeros_like(x)
#         for i in range(n):
#             idx = torch.randperm(m)
#             x_bar[:, i] += x[idx, i]
#         return x_bar
#
#     def forward(self, x: torch.tensor, mask: torch.tensor):
#         """Generate corrupted features and corresponding mask."""
#         shuffled = self.shuffle(x)
#         corrupt_x = x * (1.0 - mask) + shuffled * mask
#         return corrupt_x
#
#
# class LinearLayer(nn.Module):
#     """
#     Module to create a sequential block consisting of:
#
#         1. Linear layer
#         2. (optional) Batch normalization layer
#         3. ReLu activation layer
#     """
#
#     def __init__(self, input_size: int, output_size: int, batch_norm: bool = False):
#         super().__init__()
#         self.size_in = input_size
#         self.size_out = output_size
#         if batch_norm:
#             self.model = nn.Sequential(
#                 nn.Linear(input_size, output_size),
#                 nn.BatchNorm1d(output_size),
#                 nn.ReLU(),
#             )
#         else:
#             self.model = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
#
#     def forward(self, x: torch.tensor):
#         """Run inputs through linear block."""
#         return self.model(x)
#
#
# class LazyMLP(nn.Module):
#     def __init__(self, n_layers: int, dim_hidden: int, batch_norm: bool = False):
#         super().__init__()
#         self.n_layers = n_layers
#         self.dim_hidden = dim_hidden
#         if batch_norm:
#             lazy_block = nn.Sequential(
#                 nn.LazyLinear(dim_hidden),
#                 nn.BatchNorm1d(dim_hidden),
#                 nn.ReLU(),
#             )
#         else:
#             lazy_block = nn.Sequential(nn.LazyLinear(dim_hidden), nn.ReLU())
#
#         self.model = nn.Sequential(
#             lazy_block,
#             *[LinearLayer(dim_hidden, dim_hidden, batch_norm) for _ in range(n_layers - 1)],
#         )
#
#     def forward(self, x: torch.tensor):
#         """Run inputs through linear block."""
#         return self.model(x)
#
#
# class SCARFEncoder(pl.LightningModule):
#     def __init__(
#         self,
#         dim_hidden: int = 256,
#         n_encoder_layers: int = 4,
#         n_projection_layers: int = 2,
#         p_mask: float = 0.6,
#         loss_func: nn.Module = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=1.0)),
#         optim: Optional[torch.optim.Optimizer] = None,
#         scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
#         batch_norm: bool = False,
#     ):
#         super().__init__()
#
#         self.loss_func = loss_func
#         self.optim = optim
#         self.scheduler = scheduler
#
#         self.save_hyperparameters(ignore=["loss_func"])
#
#         self.get_mask = MaskGenerator(p=p_mask)
#         self.corrupt = PretextGenerator()
#
#         self.encoder = LazyMLP(
#             n_layers=n_encoder_layers, dim_hidden=dim_hidden, batch_norm=batch_norm
#         )
#         self.projection = LazyMLP(
#             n_layers=n_projection_layers, dim_hidden=dim_hidden, batch_norm=batch_norm
#         )
#
#     def configure_optimizers(self):
#         optim = self.optim(self.parameters()) if self.optim else torch.optim.Adam(self.parameters())
#         if self.scheduler:
#             scheduler = self.scheduler(optim)
#             return {
#                 "optimizer": optim,
#                 "lr_scheduler": {
#                     "scheduler": scheduler,
#                     "monitor": "train-loss",
#                     "interval": "epoch",
#                 },
#             }
#         return optim
#
#     def forward(self, x) -> torch.Tensor:
#         return self.encoder(x)
#
#     def encode(self, x) -> torch.Tensor:
#         self.encoder.eval()
#         with torch.no_grad():
#             return self.encoder(x)
#
#     def training_step(self, batch, idx):
#         x = batch[0]
#         mask = self.get_mask(x)
#         x_corrupt = self.corrupt(x, mask)
#         enc_x, enc_corrupt = self(x), self(x_corrupt)
#         proj_x, proj_corrupt = self.projection(enc_x), self.projection(enc_corrupt)
#         loss = self.loss_func(proj_corrupt, proj_x)
#
#         self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
#         metrics = {"train-loss": loss}
#         self.log_dict(
#             metrics,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#         )
#         return loss
#
#     def validation_step(self, batch, idx):
#         x_corrupt, x = batch
#         enc_x, enc_corrupt = self(x), self(x_corrupt)
#         proj_x, proj_corrupt = self.projection(enc_x), self.projection(enc_corrupt)
#         loss = self.loss_func(proj_corrupt, proj_x)
#
#         self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
#         metrics = {"valid-loss": loss}
#         self.log_dict(
#             metrics,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#         )
#         return loss
#
#
# class SCARFLearner(pl.LightningModule):
#     def __init__(
#         self,
#         encoder_ckpt: str,
#         dim_hidden: int = 256,
#         n_layers: int = 2,
#         num_classes: int = 7,
#         loss_func: nn.Module = nn.CrossEntropyLoss(),
#         score_func: torchmetrics.Metric = Accuracy(task="multiclass", num_classes=7),
#         optim: Optional[torch.optim.Optimizer] = None,
#         scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
#     ):
#         super().__init__(loss_func=loss_func, optim=optim, scheduler=scheduler)
#
#         self.encoder = SCARFEncoder.load_from_checkpoint(encoder_ckpt)
#         self.score_func = score_func
#
#         self.classifier = nn.Sequential(
#             LazyMLP(n_layers=n_layers, dim_hidden=dim_hidden), nn.Linear(dim_hidden, num_classes)
#         )
#
#     def forward(self, x) -> torch.Tensor:
#         embd = self.encoder(x)
#         return self.classifier(embd)
#
#     def training_step(self, batch, idx):
#         x, y = batch
#         preds = self(x)
#         loss = self.loss_func(preds, y)
#         score = self.score_func(preds, y)
#
#         self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
#         metrics = {"train-loss": loss, "train-acc": score}
#         self.log_dict(
#             metrics,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=False,
#             logger=True,
#         )
#         return loss
#
#     def validation_step(self, batch, idx):
#         x, y = batch
#         preds = self(x)
#         loss = self.loss_func(preds, y)
#         score = self.score_func(preds, y)
#
#         self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
#         metrics = {"valid-loss": loss, "valid-acc": score}
#         self.log_dict(
#             metrics,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=False,
#             logger=True,
#         )
#         return loss
