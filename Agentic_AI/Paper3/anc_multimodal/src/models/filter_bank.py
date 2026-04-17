"""Filter-bank based ANC architectures.

Provides:
- FilterBank: K pre-trained FIR filters applied to reference windows
- FilterBankSelector: Gating network that produces soft weights over K filters
- FilterBankANCModel: Multi-modal attention fusion + filter-bank selection (PROPOSED)
- SFANCBaseline: Published SFANC-style CNN classifier + filter bank (BASELINE)
- pretrain_filter_bank: Initialize filter bank from K-means on FxLMS solutions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AttentionFusion, ConcatFusion, LearnableFusion


# ── Filter Bank ──────────────────────────────────────────────────────────────

class FilterBank(nn.Module):
    """Bank of K FIR filters applied to reference windows.

    Each filter is a set of filter_order coefficients. Given a reference
    window x of shape (B, filter_order), produces K filter outputs via
    dot product: y_k = filters[k] . x for each k.
    """

    def __init__(self, K: int, filter_order: int, trainable: bool = False):
        super().__init__()
        self.K = K
        self.filter_order = filter_order
        # Initialize with small random values (will be replaced by centroids)
        init = torch.randn(K, filter_order) * 0.01
        if trainable:
            self.filters = nn.Parameter(init)
        else:
            self.register_buffer('filters', init)

    def load_centroids(self, centroids: torch.Tensor):
        """Load pre-trained filter centroids from K-means."""
        assert centroids.shape == (self.K, self.filter_order), \
            f"Expected ({self.K}, {self.filter_order}), got {centroids.shape}"
        with torch.no_grad():
            if isinstance(self.filters, nn.Parameter):
                self.filters.copy_(centroids)
            else:
                self.filters = centroids.to(self.filters.device)

    def apply(self, x_ref: torch.Tensor) -> torch.Tensor:
        """Apply all K filters to reference window.

        Args:
            x_ref: (B, filter_order) reference signal window.
        Returns:
            (B, K) filter outputs — one scalar per filter per sample.
        """
        return torch.einsum('bf,kf->bk', x_ref, self.filters)


# ── Filter Bank Selector (Gating Network) ───────────────────────────────────

class FilterBankSelector(nn.Module):
    """Gating network that produces soft weights over K filters.

    Takes fused features and outputs a probability distribution over K filters.
    Temperature controls sharpness (high = soft, low = hard selection).
    """

    def __init__(self, input_dim: int, K: int, hidden_dim: int = 64,
                 temperature: float = 1.0):
        super().__init__()
        self.K = K
        self.temperature = temperature
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )

    def set_temperature(self, temperature: float):
        """Update temperature for annealing."""
        self.temperature = max(temperature, 0.1)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: (B, input_dim) from fusion module.
        Returns:
            (B, K) soft gating weights (sum to 1 per sample).
        """
        logits = self.gate_net(fused_features)
        return F.softmax(logits / self.temperature, dim=-1)


# ── FilterBankANCModel (Proposed Architecture) ──────────────────────────────

class FilterBankANCModel(nn.Module):
    """Multi-modal filter-bank ANC model (proposed contribution).

    Architecture:
        audio_ref, accel_ref → encoders → attention fusion → gating network
        → soft weights over K pre-trained filters → weighted filter output
        + small residual correction

    The output is structurally constrained: always a convex combination of
    known-good filters, preventing the divergence seen in direct-output models.

    Supports:
        filterbank_attention — multi-modal with attention fusion
        filterbank_miconly — mic-only gating (ablation baseline)
        filterbank_learnable — multi-modal with learnable fusion
        filterbank_concat — multi-modal with concat fusion

    forward() interface matches FusedANCModel: (audio_ref, accel_ref) → (y, attn_weights)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']
        self.filter_order = model_cfg['filter_order']
        self.model_type = model_cfg['type']

        fb_cfg = model_cfg.get('filter_bank', {})
        K = fb_cfg.get('K', 8)
        trainable = fb_cfg.get('trainable_filters', False)
        gate_hidden = fb_cfg.get('gate_hidden_dim', 64)
        temperature = fb_cfg.get('temperature_start', 1.0)
        residual_scale_init = fb_cfg.get('residual_scale_init', 0.1)

        fusion_cfg = model_cfg.get('fusion', {})
        embed_dim = fusion_cfg.get('embed_dim', 64)
        num_heads = fusion_cfg.get('num_heads', 4)
        fusion_num_layers = fusion_cfg.get('num_layers', 1)
        fusion_dropout = fusion_cfg.get('dropout', 0.1)
        fusion_ffn_dim = fusion_cfg.get('ffn_dim', None)

        # 1. Filter bank (K pre-trained FIR filters)
        self.filter_bank = FilterBank(K, self.filter_order, trainable=trainable)

        # 2. Feature encoders
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.filter_order, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 3. Fusion (if multi-modal)
        self.is_multimodal = 'miconly' not in self.model_type
        if self.is_multimodal:
            self.accel_encoder = nn.Sequential(
                nn.Linear(self.filter_order, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # Pick fusion method from model_type suffix
            if 'attention' in self.model_type:
                self.fusion = AttentionFusion(
                    embed_dim, embed_dim, num_heads,
                    num_layers=fusion_num_layers,
                    dropout=fusion_dropout,
                    ffn_dim=fusion_ffn_dim,
                )
            elif 'learnable' in self.model_type:
                self.fusion = LearnableFusion(embed_dim, embed_dim, embed_dim)
            elif 'concat' in self.model_type:
                self.fusion = ConcatFusion(embed_dim, embed_dim)
            else:
                self.fusion = AttentionFusion(
                    embed_dim, embed_dim, num_heads,
                    num_layers=fusion_num_layers,
                    dropout=fusion_dropout,
                    ffn_dim=fusion_ffn_dim,
                )
            gate_input_dim = self.fusion.output_dim
        else:
            self.accel_encoder = None
            self.fusion = None
            gate_input_dim = embed_dim

        # 4. Gating network over K filters
        self.gate = FilterBankSelector(gate_input_dim, K, gate_hidden, temperature)

        # 5. Residual correction head (default [32] matches original 2-layer head)
        residual_hidden = fb_cfg.get('residual_hidden_dims', [32])
        if not residual_hidden:
            residual_hidden = [32]
        residual_dropout = float(fb_cfg.get('residual_dropout', 0.0))
        res_layers = []
        d_in = gate_input_dim
        for d_out in residual_hidden:
            res_layers.extend([nn.Linear(d_in, d_out), nn.ReLU()])
            if residual_dropout > 0:
                res_layers.append(nn.Dropout(residual_dropout))
            d_in = d_out
        res_layers.append(nn.Linear(d_in, 1))
        self.residual_head = nn.Sequential(*res_layers)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))

    def set_temperature(self, temperature: float):
        """Set gating temperature (for annealing during training)."""
        self.gate.set_temperature(temperature)

    def forward(self, audio_ref: torch.Tensor, accel_ref: torch.Tensor = None):
        """
        Args:
            audio_ref: (B, filter_order) mic reference window.
            accel_ref: (B, filter_order) accelerometer reference window.
        Returns:
            y: (B, 1) anti-noise prediction.
            attn_weights: attention/gate weights for analysis.
        """
        # Encode audio
        a_feat = self.audio_encoder(audio_ref)  # (B, embed_dim)

        # Fuse modalities
        if self.is_multimodal and accel_ref is not None:
            v_feat = self.accel_encoder(accel_ref)  # (B, embed_dim)
            fused, attn_weights = self.fusion(a_feat, v_feat)
        else:
            fused = a_feat
            attn_weights = None

        # Gate over filter bank
        gate_weights = self.gate(fused)  # (B, K)

        # Apply filter bank to raw audio reference
        filter_outputs = self.filter_bank.apply(audio_ref)  # (B, K)

        # Weighted combination of filter outputs
        y_bank = (gate_weights * filter_outputs).sum(dim=-1, keepdim=True)  # (B, 1)

        # Small residual correction
        y_residual = self.residual_head(fused) * self.residual_scale  # (B, 1)

        y = y_bank + y_residual

        # Pack gate weights into attn_weights for logging compatibility
        # If attention fusion also produced weights, combine them
        if attn_weights is None:
            attn_weights = gate_weights  # (B, K)

        return y, attn_weights

    def get_gate_weights(self, audio_ref, accel_ref=None):
        """Get gate weights without computing full output (for analysis)."""
        a_feat = self.audio_encoder(audio_ref)
        if self.is_multimodal and accel_ref is not None:
            v_feat = self.accel_encoder(accel_ref)
            fused, _ = self.fusion(a_feat, v_feat)
        else:
            fused = a_feat
        return self.gate(fused)


# ── SFANC Baseline (Published Method) ────────────────────────────────────────

class SFANCBaseline(nn.Module):
    """SFANC-style baseline: 1D CNN classifier + fixed filter bank.

    Replicates the core idea of SFANC-FxNLMS (Luo 2023):
    a 1D CNN classifies the noise type, selects from K pre-trained filters.
    Mic-only, no multi-modal fusion.

    This is the BASELINE for the paper — what we improve upon with
    multi-modal attention fusion.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']
        self.filter_order = model_cfg['filter_order']
        self.model_type = 'sfanc_baseline'

        fb_cfg = model_cfg.get('filter_bank', {})
        K = fb_cfg.get('K', 8)

        # Fixed filter bank
        self.filter_bank = FilterBank(K, self.filter_order, trainable=False)

        # 1D CNN classifier (SFANC-style)
        self.classifier = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, K),
        )

    def set_temperature(self, temperature: float):
        """No-op for API compatibility."""
        pass

    def forward(self, audio_ref: torch.Tensor, accel_ref: torch.Tensor = None):
        """
        Args:
            audio_ref: (B, filter_order) mic reference window.
            accel_ref: ignored (mic-only baseline).
        Returns:
            y: (B, 1) anti-noise prediction.
            gate_weights: (B, K) filter selection weights.
        """
        # 1D CNN classification
        x = audio_ref.unsqueeze(1)  # (B, 1, filter_order)
        features = self.classifier(x).squeeze(-1)  # (B, 32)
        gate_weights = F.softmax(self.fc(features), dim=-1)  # (B, K)

        # Apply filter bank
        filter_outputs = self.filter_bank.apply(audio_ref)  # (B, K)
        y = (gate_weights * filter_outputs).sum(dim=-1, keepdim=True)  # (B, 1)

        return y, gate_weights


# ── Filter Bank Pre-training ─────────────────────────────────────────────────

def pretrain_filter_bank(scenarios: list, config: dict, K: int = 8) -> torch.Tensor:
    """Pre-train filter bank by running FxLMS on scenarios and clustering.

    1. Run FxLMS on each scenario with physically correct error computation
    2. Extract final converged filter weights
    3. K-means clustering → K centroid filters

    Args:
        scenarios: List of scenario dicts with reference_mic, disturbance, etc.
        config: Config dict with model.filter_order and model.fxlms settings.
        K: Number of filter bank entries.

    Returns:
        (K, filter_order) tensor of centroid filters.
    """
    from .fxlms import FxLMSController
    from sklearn.cluster import KMeans

    filter_order = config['model']['filter_order']
    mu = config['model']['fxlms'].get('mu_real', 0.0001)

    all_weights = []
    all_nrs = []
    print(f"  Pre-training filter bank: running FxLMS on {len(scenarios)} scenarios...")

    for i, scenario in enumerate(scenarios):
        x = scenario['reference_mic']
        d = scenario['disturbance']
        s_hat = scenario['secondary_path_estimate']
        s = scenario['secondary_path']
        N = len(x)
        s_len = len(s)

        ctrl = FxLMSController(filter_order, mu, s_hat)
        y_cancel = np.zeros(N)
        error = np.zeros(N)

        for n in range(filter_order, N):
            x_buf = x[n:n - filter_order:-1]
            y_cancel[n] = ctrl.predict(x_buf)
            # Physically correct error: convolve through secondary path
            conv_sum = 0.0
            for k in range(min(s_len, n + 1)):
                conv_sum += s[k] * y_cancel[n - k]
            error[n] = d[n] - conv_sum
            ctrl.update(error[n], x_buf)

        # Only keep filters that actually converged (positive NR)
        d_active = d[filter_order:]
        e_active = error[filter_order:]
        d_power = np.mean(d_active ** 2)
        e_power = np.mean(e_active ** 2)
        if d_power > 1e-10 and e_power > 1e-10:
            nr = 10 * np.log10(d_power / e_power)
        else:
            nr = 0.0
        all_nrs.append(nr)

        w = ctrl.w.copy()
        # Skip degenerate filters (all zeros or NaN)
        if np.any(np.isnan(w)) or np.linalg.norm(w) < 1e-10:
            continue
        all_weights.append(w)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"    FxLMS done: {i+1}/{len(scenarios)}, NR={nr:.2f} dB, ||w||={np.linalg.norm(w):.4f}")

    if len(all_weights) < K:
        print(f"  Warning: only {len(all_weights)} valid filters (need K={K}). Padding with zeros.")
        while len(all_weights) < K:
            all_weights.append(np.zeros(filter_order))

    weight_matrix = np.array(all_weights)  # (N_valid, filter_order)

    # K-means clustering
    actual_K = min(K, len(all_weights))
    print(f"  Clustering {len(all_weights)} filter vectors into K={actual_K} centroids...")
    kmeans = KMeans(n_clusters=actual_K, random_state=42, n_init=10)
    kmeans.fit(weight_matrix)
    centroids = kmeans.cluster_centers_  # (K, filter_order)

    # Pad to K if needed
    if actual_K < K:
        pad = np.zeros((K - actual_K, filter_order))
        centroids = np.vstack([centroids, pad])

    # Report cluster sizes
    labels = kmeans.labels_
    for k in range(actual_K):
        count = np.sum(labels == k)
        norm = np.linalg.norm(centroids[k])
        print(f"    Cluster {k}: {count} members, ||w||={norm:.4f}")

    print(f"  FxLMS NR stats: mean={np.mean(all_nrs):.2f}, min={np.min(all_nrs):.2f}, max={np.max(all_nrs):.2f} dB")

    return torch.from_numpy(centroids).float()


def pretrain_filter_bank_topk(scenarios: list, config: dict, K: int = 8) -> torch.Tensor:
    """Initialize filter bank from the top-K FxLMS solutions ranked by NR.

    Unlike K-means which averages filters (often resulting in near-zero centroids
    when solutions are spread across environments), top-K selects the K *best*
    converged filters directly. These span diverse noise environments and are
    guaranteed to individually achieve positive NR.

    Args:
        scenarios: List of scenario dicts.
        config: Config dict.
        K: Number of filter bank entries.

    Returns:
        (K, filter_order) tensor of best-NR filter weights.
    """
    from .fxlms import FxLMSController

    filter_order = config['model']['filter_order']
    mu = config['model']['fxlms'].get('mu_real', 0.0001)

    all_weights = []
    all_nrs = []
    print(f"  Top-K filter bank init: running FxLMS on {len(scenarios)} scenarios...")

    for i, scenario in enumerate(scenarios):
        x = scenario['reference_mic']
        d = scenario['disturbance']
        s_hat = scenario['secondary_path_estimate']
        s = scenario['secondary_path']
        N = len(x)
        s_len = len(s)

        ctrl = FxLMSController(filter_order, mu, s_hat)
        y_cancel = np.zeros(N)
        error = np.zeros(N)

        for n in range(filter_order, N):
            x_buf = x[n:n - filter_order:-1]
            y_cancel[n] = ctrl.predict(x_buf)
            conv_sum = 0.0
            for k in range(min(s_len, n + 1)):
                conv_sum += s[k] * y_cancel[n - k]
            error[n] = d[n] - conv_sum
            ctrl.update(error[n], x_buf)

        d_active = d[filter_order:]
        e_active = error[filter_order:]
        d_power = np.mean(d_active ** 2)
        e_power = np.mean(e_active ** 2)
        if d_power > 1e-10 and e_power > 1e-10:
            nr = 10 * np.log10(d_power / e_power)
        else:
            nr = 0.0

        w = ctrl.w.copy()
        if not np.any(np.isnan(w)) and np.linalg.norm(w) > 1e-10:
            all_weights.append(w)
            all_nrs.append(nr)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    FxLMS done: {i+1}/{len(scenarios)}, NR={nr:.2f} dB")

    if len(all_weights) == 0:
        print("  Warning: no valid filters found. Using zeros.")
        centroids = np.zeros((K, filter_order))
        return torch.from_numpy(centroids).float()

    # Sort by NR descending, pick top-K (with replacement if needed)
    order = np.argsort(all_nrs)[::-1]
    print(f"  Top-K NR stats: best={all_nrs[order[0]]:.2f} dB, "
          f"K-th={all_nrs[order[min(K-1, len(order)-1)]]:.2f} dB, "
          f"mean={np.mean(all_nrs):.2f} dB")

    selected = []
    for idx in order[:K]:
        selected.append(all_weights[idx])
    # Pad by repeating if fewer than K valid filters
    while len(selected) < K:
        selected.append(selected[len(selected) % max(len(selected), 1)])

    centroids = np.stack(selected[:K])  # (K, filter_order)
    for k in range(K):
        print(f"    Filter {k}: NR={all_nrs[order[min(k, len(order)-1)]]:.2f} dB, "
              f"||w||={np.linalg.norm(centroids[k]):.4f}")
    return torch.from_numpy(centroids).float()


# ── Model Factory ────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Factory function to build ANC model from config.

    Supports all existing architectures plus new filter-bank models.
    Switch architectures by changing config['model']['type'].
    """
    from .fusion import FusedANCModel

    model_type = config['model']['type']

    if model_type.startswith('filterbank_'):
        return FilterBankANCModel(config)
    elif model_type == 'sfanc_baseline':
        return SFANCBaseline(config)
    else:
        return FusedANCModel(config)
