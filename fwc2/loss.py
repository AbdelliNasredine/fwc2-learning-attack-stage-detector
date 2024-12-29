import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NTXent(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class FWC2Loss(nn.Module):
    def __init__(
            self,
            tau: float = 1.0,
            mode: str = 'combined',
    ) -> None:
        super().__init__()
        self.contrast_loss = NTXent(temperature=tau)
        self.reconstruct_loss = nn.MSELoss()
        self.mode = mode

    def forward(self, z_i: Tensor, z_j: Tensor, x: Tensor, x_hat: Tensor) -> Tensor:
        if self.mode == 'combined':
            return self.contrast_loss(z_i, z_j) * self.reconstruct_loss(x, x_hat)
        elif self.mode == 'mse':
            return self.reconstruct_loss(x, x_hat)
        else:
            return self.contrast_loss(z_i, z_j)


class FWC2V4Loss(nn.Module):
    def __init__(
            self,
            alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.cos = nn.CosineSimilarity()

    def forward(
            self,
            z1: Tensor,
            z2: Tensor,
            x1: Tensor,
            x1_hat: Tensor,
            x2: Tensor,
            x2_hat: Tensor,
    ) -> Tensor:
        loss = self.mse_loss(x1, x1_hat) + self.mse_loss(x2, x2_hat) - self.alpha * self.cos(z1, z2)
