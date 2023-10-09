import torch
from math import sqrt
from torch import nn

D_MODEL = 512


def manual_SDPA(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
        q (tensor): The queries, of dimension (*, d_k, D_MODEL).
        k (tensor): The keys, of dimension (*, d_k, D_MODEL).
        v (tensor): The values, of dimension (*, d_v, D_MODEL).
    """
    q_mul_k = torch.matmul(q, k.transpose(-2, -1))
    scaled = torch.div(q_mul_k, sqrt(k.shape[-1]))
    e_powered = torch.exp(scaled)
    softmaxed = torch.div(e_powered, torch.sum(e_powered).unsqueeze(-1))
    return torch.matmul(softmaxed, v)


class manual_MHA(nn.Module):
    def __init__(self, num_heads: int):
        """
        Args:
            num_heads (int): Number of heads to concatenate together at end.
        """
        super().__init__()
        self.v_lin_projs = (nn.Linear(in_features=D_MODEL,
                                      out_features=D_MODEL/num_heads)
                            for head in range(num_heads))
        self.k_lin_projs = (nn.Linear(in_features=D_MODEL,
                                      out_features=D_MODEL/num_heads)
                            for head in range(num_heads))
        self.q_lin_projs = (nn.Linear(in_features=D_MODEL,
                                      out_features=D_MODEL/num_heads)
                            for head in range(num_heads))


    def forward(self, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v (tensor): The values, of dimension (batch_size, d_v, D_MODEL).
            k (tensor): The keys, of dimension (batch_size, d_k, D_MODEL).
            q (tensor): The queries, of dimension (batch_size, d_k, D_MODEL).
        """
        #TODO: For each v, k, q in batches, run v, k, q through all lin_projs

        pass

class manual_encoder(nn.Module):
    def __init__(self, d_model: int):
        """
        Args:
            d_model (int): Dimension of embeddings and tensors passed between
                internal blocks of the encoder and decoder.
        """
        super().__init__()

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

