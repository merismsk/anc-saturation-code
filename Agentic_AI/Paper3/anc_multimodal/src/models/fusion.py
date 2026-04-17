from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .controller import MLPController, CNN1DController, GRUController


class ConcatFusion(nn.Module):
    """Baseline fusion: simple concatenation of audio and accelerometer features."""

    def __init__(self, audio_dim: int, accel_dim: int):
        super().__init__()
        self.output_dim = audio_dim + accel_dim

    def forward(self, audio_feat: torch.Tensor, accel_feat: torch.Tensor):
        """
        Args:
            audio_feat: (batch, audio_dim)
            accel_feat: (batch, accel_dim)
        Returns:
            fused: (batch, audio_dim + accel_dim)
            attn_weights: None (no attention in this module)
        """
        return torch.cat([audio_feat, accel_feat], dim=-1), None


class LearnableFusion(nn.Module):
    """Gated fusion with learnable projection and gating mechanism.

    Projects both modalities to a shared embedding space, then uses
    a sigmoid gate to adaptively weight the contributions.
    """

    def __init__(self, audio_dim: int, accel_dim: int, embed_dim: int = 64):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.accel_proj = nn.Linear(accel_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.output_dim = embed_dim

    def forward(self, audio_feat: torch.Tensor, accel_feat: torch.Tensor):
        a = self.audio_proj(audio_feat)
        v = self.accel_proj(accel_feat)
        g = self.gate(torch.cat([a, v], dim=-1))
        fused = g * a + (1 - g) * v
        # Return gate values as pseudo-attention weights for analysis
        return fused, g.mean(dim=-1, keepdim=True)


class AttentionFusion(nn.Module):
    """Cross-modal attention fusion (novel contribution).

    Projects each modality to a token, then applies multi-head self-attention
    over the two tokens. Returns attention weights showing which modality
    the model attends to under different conditions.

    With ``num_layers == 1`` (default), behavior matches the original
    single ``MultiheadAttention`` block. With ``num_layers > 1``, uses a
    stacked ``TransformerEncoder`` over the two tokens (weights not exposed;
    ``attn_weights`` is None).
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.audio_proj = nn.Linear(input_dim, embed_dim)
        self.accel_proj = nn.Linear(input_dim, embed_dim)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        if num_layers == 1:
            self.attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=0.0, batch_first=True
            )
            self.encoder = None
        else:
            self.attention = None
            d_ff = ffn_dim if ffn_dim is not None else 4 * embed_dim
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.output_dim = embed_dim

    def forward(self, audio_feat: torch.Tensor, accel_feat: torch.Tensor):
        """
        Args:
            audio_feat: (batch, input_dim)
            accel_feat: (batch, input_dim)
        Returns:
            fused: (batch, embed_dim)
            attn_weights: (batch, num_heads, 2, 2) for ``num_layers==1``;
 otherwise ``None`` (PyTorch encoder does not return weights).
        """
        a = self.audio_proj(audio_feat).unsqueeze(1)  # (B, 1, D)
        v = self.accel_proj(accel_feat).unsqueeze(1)   # (B, 1, D)
        tokens = torch.cat([a, v], dim=1)               # (B, 2, D)

        if self.num_layers == 1:
            attn_out, attn_weights = self.attention(
                tokens, tokens, tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_out = self.norm(attn_out + tokens)  # Residual + LayerNorm
            fused = attn_out.mean(dim=1)             # (B, D) mean-pool over tokens
            return self.out_proj(fused), attn_weights

        enc_out = self.encoder(tokens)
        fused = enc_out.mean(dim=1)
        return self.out_proj(fused), None


class FusedANCModel(nn.Module):
    """Complete ANC model combining fusion module + controller.

    Supports: mic-only, accel-only, and multi-modal (with fusion) modes.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']
        filter_order = model_cfg['filter_order']
        ctrl_cfg = model_cfg['controller']
        fusion_cfg = model_cfg.get('fusion', {})

        model_type = model_cfg['type']
        self.model_type = model_type
        self.filter_order = filter_order

        # Determine input configuration
        if model_type in ('dl_miconly', 'dl_accelonly'):
            self.fusion = None
            input_dim = filter_order
            in_channels = 1
        else:
            # Multi-modal with fusion
            method = fusion_cfg.get('method', 'concat')
            embed_dim = fusion_cfg.get('embed_dim', 64)
            num_heads = fusion_cfg.get('num_heads', 4)

            if method == 'concat':
                self.fusion = ConcatFusion(filter_order, filter_order)
            elif method == 'learnable':
                self.fusion = LearnableFusion(filter_order, filter_order, embed_dim)
            elif method == 'attention':
                num_layers = fusion_cfg.get('num_layers', 1)
                fusion_dropout = fusion_cfg.get('dropout', 0.1)
                ffn_dim = fusion_cfg.get('ffn_dim', None)
                self.fusion = AttentionFusion(
                    filter_order, embed_dim, num_heads,
                    num_layers=num_layers, dropout=fusion_dropout, ffn_dim=ffn_dim,
                )
            else:
                raise ValueError(f"Unknown fusion method: {method}")

            input_dim = self.fusion.output_dim
            in_channels = 1  # After fusion, it's a single feature vector

        # Build controller
        ctrl_type = ctrl_cfg.get('type', 'cnn')
        hidden_dims = ctrl_cfg.get('hidden_dims', [128, 64])
        cnn_channels = ctrl_cfg.get('cnn_channels', [32, 64])

        gru_hidden = ctrl_cfg.get('gru_hidden', 64)
        gru_layers = ctrl_cfg.get('gru_layers', 2)

        if ctrl_type == 'mlp':
            self.controller = MLPController(input_dim, hidden_dims)
        elif ctrl_type == 'cnn':
            if self.fusion is None:
                self.controller = CNN1DController(1, filter_order, cnn_channels)
            else:
                self.controller = MLPController(input_dim, hidden_dims)
        elif ctrl_type == 'gru':
            # GRU processes the fused feature as a sequence
            self.controller = GRUController(
                input_dim=1, hidden_dim=gru_hidden,
                num_layers=gru_layers, output_dim=1,
            )
            # For GRU, input_dim=1 because we feed the fused vector as a sequence
            # (B, embed_dim) -> (B, embed_dim, 1) -> GRU processes sequentially
        else:
            raise ValueError(f"Unknown controller type: {ctrl_type}")

    def forward(self, audio_ref: torch.Tensor, accel_ref: torch.Tensor = None):
        """
        Args:
            audio_ref: (batch, filter_order) mic reference window.
            accel_ref: (batch, filter_order) accelerometer reference window.
        Returns:
            y: (batch, 1) anti-noise prediction.
            attn_weights: attention weights if available, else None.
        """
        attn_weights = None

        if self.model_type == 'dl_miconly':
            if isinstance(self.controller, CNN1DController):
                x = audio_ref.unsqueeze(1)  # (B, 1, L)
                y = self.controller(x)
            else:
                y = self.controller(audio_ref)
        elif self.model_type == 'dl_accelonly':
            if isinstance(self.controller, CNN1DController):
                x = accel_ref.unsqueeze(1)
                y = self.controller(x)
            else:
                y = self.controller(accel_ref)
        else:
            # Multi-modal with fusion
            fused, attn_weights = self.fusion(audio_ref, accel_ref)
            y = self.controller(fused)

        return y, attn_weights
