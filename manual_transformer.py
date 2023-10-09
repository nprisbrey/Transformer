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
        self.num_heads = num_heads
        assert D_MODEL % num_heads == 0, \
            f"D_MODEL / num_heads must divide evenly, not {D_MODEL} / {num_heads}!"
        self.d_k = int(D_MODEL / num_heads)
        self.d_v = int(D_MODEL / num_heads)
        self.q_lin_projs = nn.Parameter(torch.randn(num_heads,
                                                    D_MODEL,
                                                    self.d_k))
        self.k_lin_projs = nn.Parameter(torch.randn(num_heads,
                                                    D_MODEL,
                                                    self.d_k))
        self.v_lin_projs = nn.Parameter(torch.randn(num_heads,
                                                    D_MODEL,
                                                    self.d_v))
        self.concat_lin_proj = nn.Parameter(torch.randn(num_heads * self.d_v,
                                                        D_MODEL))

    def forward(self, embeded_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeded_seq (tensor): A matrix of the embeded sequence, of
                dimension (seq_len, D_MODEL).
        """
        # Create matrix with a copy of the input sequence for each head
        expanded_seq = embeded_seq.expand(self.num_heads, -1, -1)

        # Results have dimensions of (num_heads, seq_len, d_k or d_v)
        qs = torch.matmul(expanded_seq, self.q_lin_projs)
        ks = torch.matmul(expanded_seq, self.k_lin_projs)
        vs = torch.matmul(expanded_seq, self.v_lin_projs)

        print(f"qs: {qs.shape}")
        print(f"ks: {ks.shape}")
        print(f"vs: {vs.shape}")

        # Calculate attentions
        head_attns = manual_SDPA(qs, ks, vs)

        print(f"head_attns: {head_attns.shape}")

        # Concat head outputs to dimensions of (seq_len, self.num_heads * self.d_v)
        concat_heads = torch.cat([head_attn.squeeze(dim=0) for head_attn in torch.split(head_attns, 1, dim=0)], dim=1)

        return torch.matmul(concat_heads, self.concat_lin_proj)

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

