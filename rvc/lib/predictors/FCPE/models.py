import torch
import torch.nn as nn
import torch.nn.functional as F

# import weight_norm from different version of pytorch
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

from .model_conformer_naive import ConformerNaiveEncoder


class CFNaiveMelPE(nn.Module):
    """Conformer-based mel-to-pitch encoder for FCPE."""

    def __init__(self,
                 input_channels: int,
                 out_dims: int,
                 hidden_dims: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 f0_max: float = 1975.5,
                 f0_min: float = 32.70,
                 use_fa_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.,
                 use_harmonic_emb: bool = False,
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.use_fa_norm = use_fa_norm
        self.residual_dropout = 0.1  # unused, kept for checkpoint compatibility
        self.attention_dropout = 0.1  # unused, kept for checkpoint compatibility

        if use_harmonic_emb:
            self.harmonic_emb = nn.Embedding(9, hidden_dims)
        else:
            self.harmonic_emb = None

        self.input_stack = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dims, 3, 1, 1),
            nn.GroupNorm(4, hidden_dims),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1)
        )
        self.net = ConformerNaiveEncoder(
            num_layers=n_layers,
            num_heads=n_heads,
            dim_model=hidden_dims,
            use_norm=use_fa_norm,
            conv_only=conv_only,
            conv_dropout=conv_dropout,
            atten_dropout=atten_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(
            nn.Linear(hidden_dims, out_dims)
        )
        self.cent_table_b = torch.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0],
                                           self.f0_to_cent(torch.Tensor([f0_max]))[0],
                                           out_dims).detach()
        self.register_buffer("cent_table", self.cent_table_b)
        self.gaussian_blurred_cent_mask_b = (1200. * torch.log2(torch.Tensor([self.f0_max / 10.])))[0].detach()
        self.register_buffer("gaussian_blurred_cent_mask", self.gaussian_blurred_cent_mask_b)

    def forward(self, x: torch.Tensor, _h_emb=None) -> torch.Tensor:
        """_h_emb selects the harmonic embedding index (0/1/2); training only."""
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        if self.harmonic_emb is not None:
            if _h_emb is None:
                x = x + self.harmonic_emb(torch.LongTensor([0]).to(x.device))
            else:
                x = x + self.harmonic_emb(torch.LongTensor([int(_h_emb)]).to(x.device))
        x = self.net(x)
        x = self.norm(x)
        x = self.output_proj(x)
        x = torch.sigmoid(x)
        return x  # latent (B, T, out_dims)

    @torch.no_grad()
    def latent2cents_decoder(self,
                             y: torch.Tensor,
                             threshold: float = 0.05,
                             mask: bool = True
                             ) -> torch.Tensor:
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    @torch.no_grad()
    def latent2cents_local_decoder(self,
                                   y: torch.Tensor,
                                   threshold: float = 0.05,
                                   mask: bool = True
                                   ) -> torch.Tensor:
        """Convert latent to cents using a local argmax neighbourhood."""
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
        local_argmax_index[local_argmax_index < 0] = 0
        local_argmax_index[local_argmax_index >= self.out_dims] = self.out_dims - 1
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    @torch.no_grad()
    def gaussian_blurred_cent2latent(self, cents):  # cents: [B,N,1]
        mask = (cents > 0.1) & (cents < self.gaussian_blurred_cent_mask)
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()

    @torch.no_grad()
    def infer(self,
              mel: torch.Tensor,
              decoder: str = "local_argmax",  # "argmax" or "local_argmax"
              threshold: float = 0.05,
              ) -> torch.Tensor:
        latent = self.forward(mel)
        if decoder == "argmax":
            cents = self.latent2cents_decoder(latent, threshold=threshold)
        elif decoder == "local_argmax":
            cents = self.latent2cents_local_decoder(latent, threshold=threshold)
        else:
            raise ValueError(f"  [x] Unknown decoder type {decoder}.")
        f0 = self.cent_to_f0(cents)
        return f0  # (B, T, 1)

    def train_and_loss(self, mel, gt_f0, loss_scale=10):
        if mel.shape[-2] != gt_f0.shape[-2]:
            _len = min(mel.shape[-2], gt_f0.shape[-2])
            mel = mel[:, :_len, :]
            gt_f0 = gt_f0[:, :_len, :]
        gt_cent_f0 = self.f0_to_cent(gt_f0)
        x_gt = self.gaussian_blurred_cent2latent(gt_cent_f0)
        if self.harmonic_emb is not None:
            x = self.forward(mel, _h_emb=0)
            x_half = self.forward(mel, _h_emb=1)
            x_gt_half = self.gaussian_blurred_cent2latent(gt_cent_f0 / 2)
            x_gt_double = self.gaussian_blurred_cent2latent(gt_cent_f0 * 2)
            x_double = self.forward(mel, _h_emb=2)
            loss = F.binary_cross_entropy(x, x_gt)
            loss_half = F.binary_cross_entropy(x_half, x_gt_half)
            loss_double = F.binary_cross_entropy(x_double, x_gt_double)
            loss = loss + (loss_half + loss_double) / 2
            loss = loss * loss_scale
        else:
            x = self.forward(mel)
            loss = F.binary_cross_entropy(x, x_gt) * loss_scale
        return loss

    @torch.no_grad()
    def cent_to_f0(self, cent: torch.Tensor) -> torch.Tensor:
        f0 = 10. * 2 ** (cent / 1200.)
        return f0

    @torch.no_grad()
    def f0_to_cent(self, f0: torch.Tensor) -> torch.Tensor:
        cent = 1200. * torch.log2(f0 / 10.)
        return cent
