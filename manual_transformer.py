import torch
from math import sqrt
from torch import nn

D_MODEL = 512


class manual_SDPA(nn.Module):
    def __init__(self):
        """
        Args:

        """

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q (tensor): The queries, of dimension (batch_size, d_k).
            k (tensor): The keys, of dimension (d_k).
            v (tensor): The values, of dimension (d_v).
        """
        q_mul_k = torch.bmm(q, k.transpose(-2, -1))
        scaled = torch.div(q_mul_k, sqrt(k.shape[-1]))
        e_powered = torch.exp(scaled)
        softmaxed = torch.div(e_powered, torch.sum(e_powered).unsqueeze(-1))
        return torch.bmm(softmaxed, v)


class manual_MHA(nn.Module):
    def __init__(self, num_heads: int):
        """
        Args:
            num_heads (int): Number of heads to concatenate together at end.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO: Implement Multi-Headed Attention with heads
        pass

class manual_encoder(nn.Module):
    def __init__(self, d_model: int):
        """
        Args:
            d_model (int): Dimension of embeddings and tensors passed between
                internal blocks of the encoder and decoder.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class manual_decoder(nn.Module):
    def __init__(self, ):
        """
        Args:

        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

if __name__ == "__main__":
    sdpa = manual_SDPA()

