import numpy
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.uniform import Uniform
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy

from fwc2.loss import NTXent, FWC2Loss


class RandomFeatureCorruption:
    def __init__(self, corruption_rate, features_low, features_high):
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_rate = corruption_rate

    def __call__(self, x):
        bs, _ = x.size()

        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = self.marginals.sample(torch.Size((bs,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        return x_corrupted


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


class FWC2(nn.Module):
    def __init__(
            self,
            input_dim: int,
            features_low: int,
            features_high: int,
            dims_hidden_encoder: list,
            dims_hidden_head: list,
            corruption_rate: float = 0.6,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MLP(input_dim, dims_hidden_encoder, dropout)
        self.head = MLP(dims_hidden_encoder[-1], dims_hidden_head, dropout)

        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_rate = corruption_rate

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _ = x.size()

        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        embeddings = self.head(self.encoder(x))
        embeddings_corrupted = self.head(self.encoder(x_corrupted))

        return embeddings, embeddings_corrupted

    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class FWC2V2(nn.Module):
    def __init__(
            self,
            input_dim: int,
            features_low: int,
            features_high: int,
            dims_hidden_encoder: list,
            dims_hidden_head: list,
            corruption_rate: float = 0.6,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = MLP(input_dim, dims_hidden_encoder, dropout)
        self.head = MLP(dims_hidden_encoder[-1], dims_hidden_head, dropout)

        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_rate = corruption_rate

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _ = x.size()

        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)

        x_pos = torch.where(corruption_mask, x_random, x)
        x_anchor = torch.where(~corruption_mask, x_random, x)

        emb_pos = self.head(self.encoder(x_pos))
        emb_anchor = self.head(self.encoder(x_anchor))

        return emb_pos, emb_anchor

    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class FWC2PreTrainer(pl.LightningModule):
    def __init__(
            self,
            encoder,
            projector,
            corrupt_fn,
            loss_fn,
            config,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.corrupt_fn = corrupt_fn
        self.loss_fn = loss_fn
        self.config = config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    def _step(self, x: Tensor):
        h_i, h_j = self.projector(self.encoder(x)), self.projector(self.encoder(self.corrupt_fn(x)))
        loss = self.loss_fn(h_i, h_j)

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch

        loss = self._step(x)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        loss = self._step(x)

        self.log('val_loss', loss)


class FWC2FineTuner(pl.LightningModule):
    def __init__(
            self,
            encoder,
            predictor,
            loss_fn,
            config,
            num_classes=5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.config = config
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        # freezing encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    def forward(self, x: Tensor) -> Tensor:
        # with torch.no_grad:
        z = self.encoder(x)
        return self.predictor(z)

    def _step(self, batch):
        x, y = batch

        scores = self(x)
        loss = self.loss_fn(scores, y)

        return scores, loss, y

    def training_step(self, batch, batch_idx):
        scores, loss, y = self._step(batch)

        self.log_dict({'train_loss': loss, 'train_acc': self.train_accuracy(scores, y)}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        scores, loss, y = self._step(batch)

        self.log_dict({'val_loss': loss, 'val_acc': self.val_accuracy(scores, y)}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        scores, loss, y = self._step(batch)

        self.log_dict({'test_loss': loss, 'test_acc': self.test_accuracy(scores, y)}, prog_bar=True)


class FWC2v3(pl.LightningModule):
    def __init__(
            self,
            in_dim: int,
            encoder_dims: list,
            projector_dims: list,
            decoder_dims: list,
            corrupt_rate: float = 0.6,
            features_low: numpy.ndarray = None,
            features_high: numpy.ndarray = None,
            tau: float = 1.0,
            lr: float = 1e-3,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.encoder = MLP(in_dim, encoder_dims, dropout)
        self.projector = MLP(encoder_dims[-1], projector_dims, dropout)
        self.decoder = MLP(encoder_dims[-1], decoder_dims, dropout)
        self.corrupt_fn = RandomFeatureCorruption(corrupt_rate, features_low, features_high)
        self.loss_fn = FWC2Loss(tau)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _step(self, x: Tensor):
        z = self.encoder(x)
        h, h_p = self.projector(z), self.projector(self.encoder(self.corrupt_fn(x)))
        x_hat = self.decoder(z)
        loss = self.loss_fn(h, h_p, x, x_hat)

        return loss

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