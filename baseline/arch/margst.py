"""MargST: marginal spatio-temporal forecasting model.

Clean reference implementation. The model keeps the final architecture only --
dual-marginal context with MLP encoders and scalar fusion -- without the
ablation switches (encoder type, aggregation mode, fusion variants, STAR core).

Pipeline
--------
    x  (B, L, N, C)
      -> input encoding + TOD/DOW/adaptive embeddings      (B, L, N, D)
      -> U_T = TemporalEncoder(mean_n x)                   (B, L, 1, D)
         U_S = SpatialEncoder(mean_t x)                    (B, 1, N, D)
      -> C = U_T * U_S                                     (B, L, N, D)
      -> H = softplus(alpha) * C + softplus(beta) * x      (B, L, N, D)
      -> merge time into channels, residual MLP stack
      -> output projection                                 (B, L_out, N, C_out)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

__all__ = ["MargST", "MarginalEncoder", "DataEncoding"]


class MarginalEncoder(nn.Module):
    """Pre-LN residual MLP blocks, shared by the temporal and spatial marginal.

    The encoder is applied to a single sequence per batch element, so it carries
    no cross-token interaction: it transforms channels only. That is what makes
    marginal encoding cheap compared with encoding the full grid.

    Args:
        d_model (int): Channel width of the encoder input/output.
        mlp_ratio (float): Hidden expansion ratio inside each block.
        num_layers (int): Number of residual blocks.
        dropout (float): Dropout inside the MLP blocks.
    """

    def __init__(self, d_model: int, mlp_ratio: float = 2.0,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.layers = nn.ModuleList([
            Mlp(in_features=d_model, hidden_features=hidden,
                act_layer=nn.GELU, drop=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return x


class DataEncoding(nn.Module):
    """Multi-branch input encoding over the time axis.

    Concatenates the raw values with several context views of the same window and
    projects them back to `hid_dim`:

        raw value | multi-scale recent means | latest value | first-order diff

    Args:
        in_dim (int): Number of input channels (1 for a single traffic series).
        hid_dim (int): Output width.
        fusion_windows (tuple): Windows of the recent-mean branches.
        use_diff (bool): Whether to append the first-order difference branch.
    """

    def __init__(self, in_dim: int, hid_dim: int,
                 fusion_windows=(3, 6, 12), use_diff: bool = True):
        super().__init__()
        self.fusion_windows = tuple(fusion_windows)
        self.use_diff = bool(use_diff)
        # raw + latest + means (+ diff)
        self.num_fusion_branches = 2 + len(self.fusion_windows) + int(self.use_diff)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(self.num_fusion_branches * in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def _repeat_recent_mean(self, x: torch.Tensor, window: int) -> torch.Tensor:
        seq_len = x.shape[1]
        window = min(window, seq_len)
        recent_mean = x[:, -window:, :, :].mean(dim=1, keepdim=True)
        return recent_mean.expand(-1, seq_len, -1, -1)

    def _first_order_diff(self, x: torch.Tensor) -> torch.Tensor:
        first = torch.zeros_like(x[:, :1, :, :])
        return torch.cat([first, x[:, 1:, :, :] - x[:, :-1, :, :]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        branches = [x]
        branches += [self._repeat_recent_mean(x, w) for w in self.fusion_windows]
        branches.append(x[:, -1:, :, :].expand(-1, seq_len, -1, -1))
        if self.use_diff:
            branches.append(self._first_order_diff(x))
        data = torch.cat(branches, dim=-1)
        return self.linear2(self.activation(self.linear1(data)))


class MargST(nn.Module):
    """Marginal spatio-temporal model with a BasicTS-compatible forward.

    Args:
        num_nodes (int): Number of sensors.
        input_steps (int): History length.
        output_steps (int): Prediction length.
        steps_per_day (int): Time steps per day (resolution of the TOD table).
        input_dim (int): Traffic feature channels in `history_data`.
        output_dim (int): Output channels.
        input_embedding_dim (int): Width of the input encoding branch.
        tod_embedding_dim (int): Time-of-day embedding width (0 disables it).
        dow_embedding_dim (int): Day-of-week embedding width (0 disables it).
        adaptive_embedding_dim (int): Width of the node-time adaptive embedding.
        mlp_layers (int): Depth of both marginal encoders.
        mlp_ratio (float): Hidden expansion ratio inside every MLP.
        num_layers (int): Depth of the final residual MLP stack.
        dropout_rate (float): Dropout in the encoders and the final stack.
        dropout_adaptive (float): Dropout applied to the adaptive embedding.
        fusion_windows (tuple): Windows used by the input encoding.
        use_diff (bool): Whether the input encoding keeps the difference branch.
    """

    def __init__(self, num_nodes: int, input_steps: int = 12, output_steps: int = 12,
                 steps_per_day: int = 288, input_dim: int = 1, output_dim: int = 1,
                 input_embedding_dim: int = 24, tod_embedding_dim: int = 24,
                 dow_embedding_dim: int = 24, adaptive_embedding_dim: int = 24,
                 mlp_layers: int = 2, mlp_ratio: float = 2.0, num_layers: int = 4,
                 dropout_rate: float = 0.1, dropout_adaptive: float = 0.5,
                 fusion_windows=(3, 6, 12), use_diff: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (input_embedding_dim + tod_embedding_dim
                          + dow_embedding_dim + adaptive_embedding_dim)

        # input side
        self.input_dataEnco = DataEncoding(input_dim, input_embedding_dim,
                                           fusion_windows=fusion_windows,
                                           use_diff=use_diff)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(input_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # marginal context
        self.temporal_encoder = MarginalEncoder(self.model_dim, mlp_ratio,
                                                mlp_layers, dropout_rate)
        self.spatial_encoder = MarginalEncoder(self.model_dim, mlp_ratio,
                                               mlp_layers, dropout_rate)
        # scalars init to softplus(0)=0.693 and softplus(1)=1.313
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        self.beta_raw = nn.Parameter(torch.ones(()))

        # prediction head
        self.dropout = nn.Dropout(dropout_adaptive)
        self.encoder_projection = nn.Linear(input_steps * self.model_dim, self.model_dim)
        self.encoder_layers = nn.ModuleList([
            Mlp(in_features=self.model_dim,
                hidden_features=int(self.model_dim * mlp_ratio),
                act_layer=nn.ReLU, drop=dropout_rate)
            for _ in range(num_layers)
        ])
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(self.model_dim) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(self.model_dim, output_steps * output_dim)

    # ------------------------------------------------------------------ input
    def _create_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Build the (B, L, N, D) representation from history_data."""
        batch_size = x.shape[0]
        features = [self.input_dataEnco(x[..., :self.input_dim])]

        if self.tod_embedding_dim > 0:
            tod = (x[..., 1] * self.steps_per_day).long()
            features.append(self.tod_embedding(tod))
        if self.dow_embedding_dim > 0:
            features.append(self.dow_embedding(x[..., 2].long()))
        if self.adaptive_embedding_dim > 0:
            adaptive = self.adaptive_embedding.expand(
                batch_size, *self.adaptive_embedding.shape
            )
            features.append(self.dropout(adaptive))
        return torch.cat(features, dim=-1)

    # ------------------------------------------------------------------ model
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the history and fuse the dual-marginal context into it."""
        embedded = self._create_embeddings(x)

        # dual marginals: one sequence per batch element on each axis
        u_t = self.temporal_encoder(embedded.mean(dim=2)).unsqueeze(2)   # (B, L, 1, D)
        u_s = self.spatial_encoder(embedded.mean(dim=1)).unsqueeze(1)    # (B, 1, N, D)

        context = u_t * u_s                                              # (B, L, N, D)
        alpha = F.softplus(self.alpha_raw)
        beta = F.softplus(self.beta_raw)
        return alpha * context + beta * embedded

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = None, epoch: int = None, train: bool = True,
                **kwargs) -> torch.Tensor:
        """BasicTS entry point.

        Args:
            history_data: [B, L, N, C] with C following `FORWARD_FEATURES`
                (traffic, time-of-day, day-of-week).

        Returns:
            [B, L_out, N, output_dim]
        """
        # BasicTS stores day-of-week in [0, 1); the embedding table needs 0..6.
        history_data = history_data.clone()
        history_data[..., 2] = (history_data[..., 2] * 7.0).round()

        fused = self.encode(history_data)
        batch_size = fused.shape[0]

        merged = fused.transpose(1, 2).flatten(-2)
        hidden = self.encoder_projection(merged)
        for norm, layer in zip(self.encoder_norms, self.encoder_layers):
            hidden = hidden + layer(norm(hidden))

        output = self.output_projection(hidden).view(
            batch_size, self.num_nodes, self.output_steps, self.output_dim
        )
        return output.transpose(1, 2)
