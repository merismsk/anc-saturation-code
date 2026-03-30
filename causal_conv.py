#!/usr/bin/env python3
"""
Causal Convolution Module for Deep ANC
=======================================

This module provides truly causal 1D convolution that ensures:
- Output at time n depends ONLY on inputs at times <= n
- No look-ahead (future samples are never used)
- Compatible with real-time streaming inference

The key difference from standard PyTorch Conv1d:
- Standard: padding='same' or padding=k//2 uses symmetric padding (look-ahead!)
- Causal: left-padding only, no right-padding

Usage:
    from models.causal_conv import CausalConv1d, DeepANC_Causal
    
    model = DeepANC_Causal()  # Truly causal 1D-CNN

Author: Manoj Singh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution Layer.
    
    Ensures output[n] depends only on input[n-k+1:n+1] where k is kernel_size.
    This is achieved by:
    1. Using padding=0 in the underlying Conv1d
    2. Manually applying left-padding of (kernel_size - 1) * dilation
    
    For kernel_size=64, dilation=1:
    - Left padding: 63 samples
    - Right padding: 0 samples
    - Output at time n uses input[n-63:n+1]
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Size of convolving kernel
        dilation: Spacing between kernel elements (default: 1)
        bias: If True, adds learnable bias (default: True)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation
        
        # Underlying convolution with NO padding
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # Critical: no built-in padding
            bias=bias
        )
    
    def forward(self, x):
        """
        Forward pass with causal (left-only) padding.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Output tensor of shape (batch, out_channels, time)
            Same temporal length as input (due to causal padding)
        """
        # Apply left-only padding: (left, right) along time dimension
        x_padded = F.pad(x, (self.left_pad, 0))
        return self.conv(x_padded)
    
    def extra_repr(self):
        return f'causal_padding={self.left_pad}'


class DeepANC_Causal(nn.Module):
    """
    Truly Causal 1D-CNN for Active Noise Control.
    
    Architecture:
        Input: (B, 1, T) - mono audio
        Conv1 (causal): 1 -> 16 channels, kernel=64, ReLU
        Conv2 (1x1): 16 -> 16 channels, ReLU  
        Conv3 (1x1): 16 -> 1 channel
        Output: (B, 1, T) - anti-noise signal
    
    Receptive field: 64 samples (4 ms at 16 kHz)
    Parameters: 1,329
    MACs per sample: 1,296
    
    CRITICAL: This version uses TRUE causal convolution.
    Output[n] depends only on Input[n-63:n+1].
    No look-ahead / future samples used.
    """
    
    def __init__(self):
        super().__init__()
        # Layer 1: Causal conv with 64-sample receptive field
        self.conv1 = CausalConv1d(1, 16, kernel_size=64)
        self.relu1 = nn.ReLU()
        
        # Layer 2: 1x1 conv (no temporal extent, inherently causal)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=1)
        self.relu2 = nn.ReLU()
        
        # Layer 3: 1x1 conv output projection
        self.conv3 = nn.Conv1d(16, 1, kernel_size=1)
        
        # Count parameters
        self._count_params()
    
    def _count_params(self):
        """Count and store number of parameters."""
        self.n_params = sum(p.numel() for p in self.parameters())
        # Layer breakdown:
        # conv1: 1*16*64 + 16 = 1,040
        # conv2: 16*16*1 + 16 = 272
        # conv3: 16*1*1 + 1 = 17
        # Total: 1,329
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, time)
            
        Returns:
            Anti-noise signal (batch, 1, time)
        """
        h = self.relu1(self.conv1(x))
        h = self.relu2(self.conv2(h))
        return self.conv3(h)
    
    @torch.no_grad()
    def inference_single_sample(self, buffer):
        """
        Single-sample streaming inference.
        
        Args:
            buffer: Ring buffer of last 64 samples, shape (64,)
            
        Returns:
            Single output sample (scalar)
        """
        # Reshape for batch processing
        x = buffer.view(1, 1, 64)
        y = self.forward(x)
        return y[0, 0, -1].item()


class DeepANC_NonCausal(nn.Module):
    """
    Non-Causal 1D-CNN (ORIGINAL BUGGY VERSION).
    
    This version uses symmetric padding (padding=32 for kernel=64),
    which means output[n] uses input[n-31:n+33] - looking 32 samples ahead!
    
    This is INCORRECT for real-time ANC and violates causality.
    Kept for comparison purposes only.
    
    WARNING: Do not use for real-time applications!
    """
    
    def __init__(self):
        super().__init__()
        # BUG: padding=32 causes look-ahead of 32 samples
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, padding=32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 16, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(16, 1, kernel_size=1)
        
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        h = self.relu1(self.conv1(x))
        h = self.relu2(self.conv2(h))
        return self.conv3(h)


def verify_causality():
    """
    Verify that CausalConv1d is truly causal.
    
    Test: Changing a future input should NOT affect current output.
    """
    print("=" * 60)
    print("CAUSALITY VERIFICATION TEST")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create test models
    causal_model = DeepANC_Causal()
    noncausal_model = DeepANC_NonCausal()
    
    # Copy weights for fair comparison
    noncausal_model.load_state_dict(causal_model.state_dict(), strict=False)
    
    # Test input
    x = torch.randn(1, 1, 128)
    
    # Get outputs
    causal_model.eval()
    noncausal_model.eval()
    
    with torch.no_grad():
        y_causal = causal_model(x)
        y_noncausal = noncausal_model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Causal output shape: {y_causal.shape}")
    print(f"Non-causal output shape: {y_noncausal.shape}")
    
    # Test: modify future samples, check if current output changes
    x_modified = x.clone()
    x_modified[0, 0, 64:] = 0  # Zero out samples 64-127
    
    with torch.no_grad():
        y_causal_mod = causal_model(x_modified)
        y_noncausal_mod = noncausal_model(x_modified)
    
    # For causal model: output[0:64] should be IDENTICAL
    # (since it only depends on input[0:64])
    causal_diff = (y_causal[0, 0, :64] - y_causal_mod[0, 0, :64]).abs().max().item()
    noncausal_diff = (y_noncausal[0, 0, :64] - y_noncausal_mod[0, 0, :64]).abs().max().item()
    
    print(f"\nTest: Zero out input[64:128], check output[0:64]")
    print(f"  Causal model max diff: {causal_diff:.2e}")
    print(f"  Non-causal model max diff: {noncausal_diff:.2e}")
    
    if causal_diff < 1e-6:
        print("\n✓ CAUSAL MODEL PASSES: Future inputs don't affect current output")
    else:
        print("\n✗ CAUSAL MODEL FAILS: Future inputs affect current output!")
    
    if noncausal_diff > 1e-6:
        print("✓ NON-CAUSAL MODEL CONFIRMED: Has look-ahead (as expected)")
    else:
        print("✗ NON-CAUSAL MODEL: No look-ahead detected (unexpected)")
    
    # Quantify look-ahead effect
    print(f"\n" + "=" * 60)
    print("LOOK-AHEAD QUANTIFICATION")
    print("=" * 60)
    
    # For non-causal with padding=32, output[n] uses input[n-31:n+33]
    # So modifying input[64] should affect output[32:64]
    
    x_single = x.clone()
    x_single[0, 0, 64] = 100  # Large spike at sample 64
    
    with torch.no_grad():
        y_spike_causal = causal_model(x_single)
        y_spike_noncausal = noncausal_model(x_single)
    
    causal_affected = (y_spike_causal - y_causal).abs()
    noncausal_affected = (y_spike_noncausal - y_noncausal).abs()
    
    # Find first and last affected samples
    causal_first = (causal_affected[0, 0] > 1e-6).nonzero()
    noncausal_first = (noncausal_affected[0, 0] > 1e-6).nonzero()
    
    if len(causal_first) > 0:
        print(f"Causal: Spike at input[64] affects output[{causal_first[0].item()}:{causal_first[-1].item()+1}]")
    else:
        print("Causal: No output affected (spike outside receptive field)")
    
    if len(noncausal_first) > 0:
        print(f"Non-causal: Spike at input[64] affects output[{noncausal_first[0].item()}:{noncausal_first[-1].item()+1}]")
        print(f"  → Look-ahead of {64 - noncausal_first[0].item()} samples detected!")
    
    return causal_diff < 1e-6


if __name__ == "__main__":
    verify_causality()
    
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    model = DeepANC_Causal()
    print(f"\nDeepANC_Causal:")
    print(f"  Parameters: {model.n_params:,}")
    print(f"  Receptive field: 64 samples (4 ms @ 16 kHz)")
    print(f"  Causality: TRUE (left-padding only)")
    
    # Verify parameter count matches paper
    expected_params = 1329
    assert model.n_params == expected_params, f"Expected {expected_params}, got {model.n_params}"
    print(f"\n✓ Parameter count verified: {model.n_params}")
