import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.distributions.uniform import Uniform
import pytorch_lightning as pl


class RandomFeatureCorruption:
    def __init__(self,corruption_rate: float = 0.0, features_min: float = 0.0,features_max: float = 1.0):
        self.marginals = Uniform(torch.Tensor(features_min), torch.Tensor(features_max))
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

    def forward(self, x: Tensor) -> Tensor:
        embeddings, embeddings_corrupted = self.model(x)
        return embeddings, embeddings_corrupted

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        z_i, z_j = self.projector(self.encoder(x)), self.projector(self.encoder(self.corrupt_fn(x)))
        loss = self.loss_fn(z_i, z_j)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        z_i, z_j = self.projector(self.encoder(x)), self.projector(self.encoder(self.corrupt_fn(x)))
        loss = self.loss_fn(z_i, z_j)

        self.log('val_loss', loss)
