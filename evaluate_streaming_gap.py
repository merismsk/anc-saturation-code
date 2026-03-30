#!/usr/bin/env python3
"""
Evaluate offline-vs-streaming equivalence for the Paper 1 causal ANC model.

This compares:
1. Offline batch inference: y = model(x) on the full clip
2. Streaming ring-buffer inference: one sample at a time using Algorithm 1

The ANC plant and NR computation are identical for both modes.
"""

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from models.causal_conv import DeepANC_Causal
import train_proper_weighted as tpw
from train_proper_weighted import build_clip_database, ANCDataset, create_paths


ROOT = Path("/Users/manojsingh/Library/CloudStorage/OneDrive-Personal/PhD")
MODEL_PATH = ROOT / "causal_anc_proper_weighted.pth"
ALPHA = 2.0
R = 64


def configure_data_root() -> Path:
    candidates = [
        Path("/Users/manojsingh/Desktop/PhD/data_large"),
        Path("/Users/manojsingh/Library/CloudStorage/OneDrive-Personal/PhD/data_large"),
        Path("/Users/manojsingh/PhD_local/data_large"),
    ]
    for root in candidates:
        if (root / "ESC-50-master").exists():
            tpw.DATA_ROOT = root
            tpw.ESC50_PATH = root / "ESC-50-master"
            tpw.DEMAND_PATH = root
            tpw.RECORDED_PATH = root / "recorded audio"
            tpw.LIBRISPEECH_PATH = root / "LibriSpeech/train-other-500"
            return root
    raise FileNotFoundError("Could not locate Paper 1 data_large directory")


def ordered_ring_buffer(buffer: torch.Tensor, write_ptr: int) -> torch.Tensor:
    """Return the 64-sample window in chronological order, oldest->newest."""
    if write_ptr == R - 1:
        return buffer.clone()
    return torch.cat((buffer[write_ptr + 1 :], buffer[: write_ptr + 1]))


@torch.no_grad()
def streaming_forward(model: DeepANC_Causal, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sample-by-sample ring-buffer execution matching Algorithm 1."""
    x = x.flatten().to(device)
    buffer = torch.zeros(R, dtype=torch.float32, device=device)
    outputs = torch.zeros_like(x)
    write_ptr = -1

    for n in range(x.numel()):
        write_ptr = (write_ptr + 1) % R
        buffer[write_ptr] = x[n]
        window = ordered_ring_buffer(buffer, write_ptr)
        outputs[n] = model.inference_single_sample(window)

    return outputs.view(1, 1, -1)


@torch.no_grad()
def anc_signals(model: DeepANC_Causal, x: torch.Tensor, P_t: torch.Tensor, S_t: torch.Tensor,
                alpha: float, mode: str, device: torch.device):
    """Compute disturbance, controller output, and error for offline or streaming mode."""
    x = x.to(device)
    pad_p = P_t.shape[-1] - 1
    pad_s = S_t.shape[-1] - 1

    d = F.conv1d(F.pad(x, (pad_p, 0)), P_t)

    if mode == "offline":
        y = model(x)
    elif mode == "streaming":
        y = streaming_forward(model, x, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    min_len = min(d.shape[-1], y.shape[-1])
    d = d[:, :, :min_len]
    y = y[:, :, :min_len]

    v = F.conv1d(F.pad(y, (pad_s, 0)), S_t)[:, :, :min_len]
    z = torch.tanh(alpha * v)
    e = d + z

    return d, y, e


def nr_db(d: torch.Tensor, e: torch.Tensor) -> float:
    var_d = torch.var(d).item()
    var_e = torch.var(e).item()
    return 10.0 * np.log10(var_e / (var_d + 1e-10))


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_root = configure_data_root()
    print(f"Data root: {data_root}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    model = DeepANC_Causal().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_clips = build_clip_database(is_train=False)
    test_dataset = ANCDataset(test_clips)
    print(f"Test clips: {len(test_dataset)}")

    P_ir, S_ir = create_paths()
    P_t = torch.tensor(P_ir, dtype=torch.float32).view(1, 1, -1).to(device)
    S_t = torch.tensor(S_ir, dtype=torch.float32).view(1, 1, -1).to(device)

    offline_nrs = []
    streaming_nrs = []
    y_diffs = []
    e_diffs = []

    for idx in range(len(test_dataset)):
        x, _ = test_dataset[idx]
        x = x.unsqueeze(0)

        d_off, y_off, e_off = anc_signals(model, x, P_t, S_t, ALPHA, "offline", device)
        d_str, y_str, e_str = anc_signals(model, x, P_t, S_t, ALPHA, "streaming", device)

        offline_nrs.append(nr_db(d_off, e_off))
        streaming_nrs.append(nr_db(d_str, e_str))
        y_diffs.append(torch.max(torch.abs(y_off - y_str)).item())
        e_diffs.append(torch.max(torch.abs(e_off - e_str)).item())

    offline_mean = float(np.mean(offline_nrs))
    streaming_mean = float(np.mean(streaming_nrs))
    gap = streaming_mean - offline_mean

    print("\nOffline vs Streaming ANC")
    print("-" * 40)
    print(f"Offline mean NR:   {offline_mean:.4f} dB")
    print(f"Streaming mean NR: {streaming_mean:.4f} dB")
    print(f"Gap (stream-off):  {gap:+.4f} dB")
    print(f"Median clip gap:   {np.median(np.array(streaming_nrs) - np.array(offline_nrs)):+.6f} dB")
    print(f"Max |y_off-y_str|: {max(y_diffs):.8f}")
    print(f"Max |e_off-e_str|: {max(e_diffs):.8f}")

    out_path = ROOT / "streaming_gap_results.txt"
    with open(out_path, "w") as f:
        f.write("Offline vs Streaming ANC (Paper 1)\n")
        f.write(f"Test clips: {len(test_dataset)}\n")
        f.write(f"Alpha: {ALPHA}\n")
        f.write(f"Offline mean NR: {offline_mean:.6f} dB\n")
        f.write(f"Streaming mean NR: {streaming_mean:.6f} dB\n")
        f.write(f"Gap (stream-off): {gap:+.6f} dB\n")
        f.write(f"Median clip gap: {np.median(np.array(streaming_nrs) - np.array(offline_nrs)):+.6f} dB\n")
        f.write(f"Max |y_off-y_str|: {max(y_diffs):.10f}\n")
        f.write(f"Max |e_off-e_str|: {max(e_diffs):.10f}\n")

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
