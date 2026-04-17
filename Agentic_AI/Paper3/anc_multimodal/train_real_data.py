#!/usr/bin/env python3
"""Train and evaluate on real DEMAND noise + BUT ReverbDB RIRs."""
import os
import sys
import argparse
import time
import json
import csv
import glob
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import scipy.io.wavfile as wav
from scipy.signal import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.simulation.real_data import (
    RealDataGenerator, DEMAND_ENVS, load_rir, extract_early_rir, build_real_scenario,
)
from src.simulation.paths import apply_path
from src.models.filter_bank import build_model
from src.models.fxlms import FxLMSController, MultiModalFxLMS
from src.training.metrics import noise_reduction_db, frequency_band_nr


class RealDataChunkDataset(torch.utils.data.Dataset):
    """Windowed dataset from real-data scenarios."""
    def __init__(self, scenarios, filter_order, chunk_size=512, chunk_stride=4096):
        self.filter_order = filter_order
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.audio = [sc['reference_mic'] for sc in scenarios]
        self.accel = [sc['reference_accel'] for sc in scenarios]
        self.dist = [sc['disturbance'] for sc in scenarios]
        if scenarios:
            max_s_len = max(len(sc['secondary_path']) for sc in scenarios)
            self.secondary = np.stack([
                np.pad(sc['secondary_path'], (0, max_s_len - len(sc['secondary_path'])))
                for sc in scenarios
            ]).astype(np.float32)
        else:
            self.secondary = np.zeros((0, 0), dtype=np.float32)

        self.num_scenarios = len(scenarios)
        if self.num_scenarios > 0:
            self.seq_len = len(self.audio[0])
            self.max_start = self.seq_len - self.filter_order - self.chunk_size
            self.num_chunks = max(1, self.max_start // self.chunk_stride)
            self.total_len = self.num_scenarios * self.num_chunks
        else:
            self.total_len = 0

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        scenario_idx = idx // self.num_chunks
        chunk_idx = idx % self.num_chunks
        s_idx = chunk_idx * self.chunk_stride
        e_idx = s_idx + self.filter_order + self.chunk_size - 1

        x = self.audio[scenario_idx][s_idx:e_idx]
        a = self.accel[scenario_idx][s_idx:e_idx]
        d = self.dist[scenario_idx][s_idx + self.filter_order:e_idx + 1]

        x_wins = np.lib.stride_tricks.sliding_window_view(x, self.filter_order)
        a_wins = np.lib.stride_tricks.sliding_window_view(a, self.filter_order)

        return (torch.from_numpy(np.ascontiguousarray(x_wins)).float(),
                torch.from_numpy(np.ascontiguousarray(a_wins)).float(),
                torch.from_numpy(d.copy()).float(),
                torch.from_numpy(self.secondary[scenario_idx].copy()).float())


def _compute_secondary_path_error(y_cancel: torch.Tensor, disturbance: torch.Tensor,
                                   secondary_paths: torch.Tensor) -> torch.Tensor:
    """Compute e = d - S_i*y with per-sample secondary paths.

    Returns the (B, N) error tensor (not reduced to scalar).
    """
    if y_cancel.ndim != 2 or disturbance.ndim != 2 or secondary_paths.ndim != 2:
        raise ValueError("Expected y_cancel/disturbance/secondary_paths to be 2D tensors")
    if y_cancel.shape != disturbance.shape:
        raise ValueError("y_cancel and disturbance must have matching shapes")
    if y_cancel.shape[0] != secondary_paths.shape[0]:
        raise ValueError("Batch dimension mismatch between predictions and secondary paths")

    B, N = y_cancel.shape
    s_len = secondary_paths.shape[1]
    if s_len == 0:
        return disturbance - y_cancel

    kernels = torch.flip(secondary_paths, dims=[1]).unsqueeze(1)  # (B, 1, S)
    inp = y_cancel.unsqueeze(0)  # (1, B, N)
    y_filtered = F.conv1d(inp, kernels, padding=s_len - 1, groups=B)  # (1, B, N+S-1)
    y_filtered = y_filtered[:, :, :N].squeeze(0)  # (B, N)
    return disturbance - y_filtered


def variable_secondary_path_loss(y_cancel: torch.Tensor, disturbance: torch.Tensor,
                                 secondary_paths: torch.Tensor) -> torch.Tensor:
    """MSE of e = d - S_i*y with per-sample secondary paths."""
    error = _compute_secondary_path_error(y_cancel, disturbance, secondary_paths)
    return torch.mean(error ** 2)


def frequency_weighted_loss(y_cancel: torch.Tensor, disturbance: torch.Tensor,
                            secondary_paths: torch.Tensor,
                            fs: int = 16000, low_weight: float = 2.0) -> torch.Tensor:
    """MSE with extra weight on the 20-500 Hz band where DL complements FxLMS.

    Applies the standard secondary-path loss plus an additional weighted term
    computed on the low-frequency content of the error signal (via FFT masking).
    """
    error = _compute_secondary_path_error(y_cancel, disturbance, secondary_paths)
    base_loss = torch.mean(error ** 2)

    if low_weight <= 0:
        return base_loss

    B, N = error.shape
    E_fft = torch.fft.rfft(error, dim=1)
    freqs = torch.fft.rfftfreq(N, d=1.0 / fs).to(error.device)  # (N//2+1,)

    low_mask = ((freqs >= 20) & (freqs <= 500)).float()  # 1 inside band, 0 outside
    E_low = E_fft * low_mask.unsqueeze(0)
    error_low = torch.fft.irfft(E_low, n=N, dim=1)
    low_loss = torch.mean(error_low ** 2)

    return base_loss + low_weight * low_loss


def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def split_envs_by_category(seed: int = 42, holdout_per_category: int = 1) -> tuple:
    """Create train/test env split with every category represented in both sets."""
    rng = np.random.default_rng(seed)
    train_envs = []
    test_envs = []
    for _, envs in DEMAND_ENVS.items():
        env_list = list(envs)
        rng.shuffle(env_list)
        holdout = max(1, min(len(env_list) - 1, holdout_per_category))
        test_envs.extend(env_list[:holdout])
        train_envs.extend(env_list[holdout:])
    return train_envs, test_envs


def normalize_windows(windows: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-window standardization for robust cross-domain inference/training."""
    mean = windows.mean(dim=1, keepdim=True)
    std = windows.std(dim=1, keepdim=True, unbiased=False)
    return (windows - mean) / (std + eps)


def discover_external_audio_files(extra_root: str) -> list:
    """Discover external audio files (WAV + FLAC) from available large datasets.

    Scans ESC-50, UrbanSound8K, and LibriSpeech. LibriSpeech contributes the
    bulk of volume (~37k FLAC files across train-clean-100 and train-other-500).
    """
    if not extra_root or not os.path.isdir(extra_root):
        return []
    candidate_dirs = [
        os.path.join(extra_root, 'ESC-50-master', 'audio'),
        os.path.join(extra_root, 'UrbanSound8K', 'audio'),
        os.path.join(extra_root, 'recorded audio'),
        os.path.join(extra_root, 'LibriSpeech', 'train-clean-100'),
        os.path.join(extra_root, 'LibriSpeech', 'train-other-500'),
    ]
    audio_files = []
    exts = ('.wav', '.flac')
    for base in candidate_dirs:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for name in files:
                if name.lower().endswith(exts):
                    audio_files.append(os.path.join(root, name))
    return sorted(audio_files)


def _load_audio_any(path: str):
    """Decode WAV or FLAC. Returns (sample_rate, float64_mono_array).

    WAV uses scipy.io.wavfile (fast, in-process). FLAC uses pydub/ffmpeg.
    Callers handle resampling and mono-mixing downstream.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        sr, raw = wav.read(path)
        return sr, raw
    if ext == '.flac':
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path, format='flac')
        sample_width = seg.sample_width  # bytes per sample
        sr = seg.frame_rate
        channels = seg.channels
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        data = np.frombuffer(seg.raw_data, dtype=dtype)
        if channels > 1:
            data = data.reshape(-1, channels)
        return sr, data
    raise ValueError(f"Unsupported audio extension: {ext}")


def load_external_clip(wav_path: str, target_fs: int, duration: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Load a random clip from an arbitrary WAV or FLAC file."""
    try:
        sr, raw = _load_audio_any(wav_path)
    except Exception:
        return None

    if raw.ndim > 1:
        raw = raw.mean(axis=1)

    if np.issubdtype(raw.dtype, np.integer):
        max_val = float(np.iinfo(raw.dtype).max)
        if max_val <= 0:
            return None
        data = raw.astype(np.float64) / max_val
    else:
        data = raw.astype(np.float64)

    if data.size < 2:
        return None

    if sr != target_fs:
        new_len = int(round(len(data) * target_fs / sr))
        if new_len < 2:
            return None
        data = resample(data, new_len)

    needed = int(duration * target_fs)
    if needed <= 1:
        return None
    if len(data) < needed:
        reps = int(np.ceil(needed / max(len(data), 1)))
        data = np.tile(data, reps)

    max_start = len(data) - needed
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    clip = data[start:start + needed]
    peak = np.max(np.abs(clip))
    if peak < 1e-8:
        return None
    clip = (clip / peak) * 0.5
    return clip


def build_external_scenario(noise: np.ndarray, train_gen: RealDataGenerator,
                            sensor_config: dict, rng: np.random.Generator) -> dict:
    """Build a scenario from external noise using real RIRs."""
    room = rng.choice(list(train_gen.rir_paths.keys()))
    room_rirs = train_gen.rir_paths[room]
    indices = rng.choice(len(room_rirs), size=2, replace=False)
    primary_rir = extract_early_rir(load_rir(room_rirs[indices[0]], max_len=4096), max_taps=128)
    secondary_rir = extract_early_rir(load_rir(room_rirs[indices[1]], max_len=4096), max_taps=64)

    # Build a weakly decorrelated reference for single-channel external clips.
    delay = int(rng.integers(2, 16))
    reference_noise = np.roll(noise, delay)
    reference_noise[:delay] = 0.0
    reference_noise += 0.005 * rng.standard_normal(len(noise))

    return build_real_scenario(
        noise=noise,
        primary_rir=primary_rir,
        secondary_rir=secondary_rir,
        reference_noise=reference_noise,
        sensor_config=sensor_config,
        rng=rng,
    )


def add_external_scenarios(train_scenarios: list, train_gen: RealDataGenerator, config: dict,
                           extra_root: str, num_extra: int, seed: int) -> list:
    """Augment training scenarios with large-dataset external WAV/FLAC clips."""
    if num_extra <= 0:
        return train_scenarios

    audio_files = discover_external_audio_files(extra_root)
    if not audio_files:
        print(f"  No external WAV/FLAC files found under: {extra_root}")
        return train_scenarios

    rng = np.random.default_rng(seed + 2026)
    added = 0
    attempts = 0
    max_attempts = max(num_extra * 5, num_extra)
    while added < num_extra and attempts < max_attempts:
        attempts += 1
        path = audio_files[int(rng.integers(0, len(audio_files)))]
        clip = load_external_clip(
            wav_path=path,
            target_fs=config['simulation']['fs'],
            duration=config['simulation']['duration'],
            rng=rng,
        )
        if clip is None:
            continue
        scenario = build_external_scenario(
            noise=clip,
            train_gen=train_gen,
            sensor_config=config['simulation']['sensors'],
            rng=rng,
        )
        train_scenarios.append(scenario)
        added += 1

    print(f"  Added {added}/{num_extra} external scenarios from {len(audio_files)} WAV/FLAC files")
    return train_scenarios


def _fxlms_residual_worker(task):
    """Worker: runs FxLMS on one scenario, returns (idx, residual_scenario_dict)."""
    idx, sc, config = task
    fx = run_fxlms_online(sc, config)
    sc_res = dict(sc)
    sc_res['disturbance'] = fx['error'].copy()
    return idx, sc_res


def build_fxlms_residual_scenarios(scenarios: list, config: dict, tag: str = "set",
                                   num_workers: int = 0) -> list:
    """Transform scenarios so disturbance is residual after FxLMS.

    Serial-only path kept for backward compatibility. For parallel use of more than
    one labeled batch, prefer ``build_fxlms_residual_scenarios_batched`` — it
    instantiates exactly one multiprocessing.Pool, avoiding the double-spawn
    deadlock observed on macOS when Pool() is called twice from the same process.
    """
    total = len(scenarios)
    if num_workers <= 1 or total <= 1:
        residual_scenarios = []
        for idx, sc in enumerate(scenarios):
            fx = run_fxlms_online(sc, config)
            sc_res = dict(sc)
            sc_res['disturbance'] = fx['error'].copy()
            residual_scenarios.append(sc_res)
            if (idx + 1) % 20 == 0 or idx == 0 or (idx + 1) == total:
                print(f"  Residual targets ({tag}): {idx + 1}/{total}")
        return residual_scenarios

    # Single-batch parallel path — works fine when Pool is only opened once per run.
    return build_fxlms_residual_scenarios_batched(
        batches=[(tag, scenarios)],
        config=config,
        num_workers=num_workers,
    )[tag]


def build_fxlms_residual_scenarios_batched(batches: list, config: dict,
                                           num_workers: int) -> dict:
    """Parallel FxLMS residual building across multiple labeled scenario lists.

    Args:
        batches: list of (tag, scenarios_list) tuples — typically [('train', ...),
                 ('val', ...)]. All tags are processed with a single shared Pool.
        num_workers: pool size. If <=1, falls back to serial per batch.

    Returns:
        dict mapping tag -> residual_scenarios list (same order as input).
    """
    # Serial fallback (no pool)
    if num_workers <= 1:
        out = {}
        for tag, scenarios in batches:
            out[tag] = build_fxlms_residual_scenarios(
                scenarios, config, tag=tag, num_workers=0,
            )
        return out

    import multiprocessing as mp
    ctx = mp.get_context('spawn')  # macOS-safe; avoids fork-related MPS issues

    # Build one combined task list tagged with (batch_idx, local_idx)
    all_tasks = []
    batch_sizes = []
    for b_idx, (tag, scenarios) in enumerate(batches):
        batch_sizes.append(len(scenarios))
        for l_idx, sc in enumerate(scenarios):
            # Pack as (global_task_key, scenario, config) where global_task_key
            # encodes (batch_idx, local_idx). Worker echoes it back verbatim.
            all_tasks.append(((b_idx, l_idx), sc, config))

    total_all = len(all_tasks)
    results_by_batch = [[None] * n for n in batch_sizes]

    tags_str = ", ".join(f"{t}={n}" for (t, _), n in zip(batches, batch_sizes))
    print(f"  Residual targets (combined: {tags_str}): "
          f"{total_all} scenarios on {num_workers} workers (parallel)")

    done = 0
    done_per_batch = [0] * len(batches)
    with ctx.Pool(num_workers) as pool:
        for key, sc_res in pool.imap_unordered(
            _fxlms_residual_worker, all_tasks, chunksize=1,
        ):
            b_idx, l_idx = key
            results_by_batch[b_idx][l_idx] = sc_res
            done += 1
            done_per_batch[b_idx] += 1
            # Progress output — per-batch, like the old behavior
            b_total = batch_sizes[b_idx]
            dpb = done_per_batch[b_idx]
            tag_name = batches[b_idx][0]
            if dpb == 1 or dpb % 20 == 0 or dpb == b_total:
                print(f"  Residual targets ({tag_name}): {dpb}/{b_total}")

    return {batches[i][0]: results_by_batch[i] for i in range(len(batches))}


def apply_hybrid_energy_cap(y_dl: np.ndarray, residual_signal: np.ndarray,
                            energy_cap: float) -> tuple:
    """Cap DL residual energy relative to FxLMS residual RMS."""
    y = y_dl.copy()
    if energy_cap <= 0:
        return y, 1.0

    dl_rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    res_rms = float(np.sqrt(np.mean(residual_signal ** 2)) + 1e-12)
    max_dl_rms = energy_cap * res_rms
    if dl_rms <= max_dl_rms:
        return y, 1.0

    cap_scale = max_dl_rms / dl_rms
    return y * cap_scale, float(cap_scale)


def compose_safe_hybrid_output(y_fxlms: np.ndarray, y_dl: np.ndarray,
                               residual_signal: np.ndarray, dl_scale: float,
                               energy_cap: float) -> tuple:
    """Combine FxLMS and DL residual with safety cap + scale."""
    y_dl_capped, cap_scale = apply_hybrid_energy_cap(y_dl, residual_signal, energy_cap)
    y_total = y_fxlms + dl_scale * y_dl_capped
    return y_total, y_dl_capped, cap_scale


def run_adaptive_hybrid_mix(disturbance: np.ndarray, secondary_path: np.ndarray,
                            y_fxlms: np.ndarray, y_dl: np.ndarray,
                            init_scale: float, mu: float,
                            scale_min: float, scale_max: float) -> tuple:
    """Causal LMS adaptation of DL residual gain in hybrid mode."""
    N = len(disturbance)
    s_len = len(secondary_path)
    y_total = np.zeros(N, dtype=np.float64)
    error = np.zeros(N, dtype=np.float64)
    scale_trace = np.zeros(N, dtype=np.float64)
    scale = float(np.clip(init_scale, scale_min, scale_max))

    for n in range(N):
        y_total[n] = y_fxlms[n] + scale * y_dl[n]
        conv_sum = 0.0
        u_n = 0.0
        upper = min(s_len, n + 1)
        for k in range(upper):
            s_k = secondary_path[k]
            conv_sum += s_k * y_total[n - k]
            u_n += s_k * y_dl[n - k]
        error[n] = disturbance[n] - conv_sum
        if mu > 0:
            denom = (u_n * u_n) + 1e-10
            scale += mu * error[n] * u_n / denom
            scale = float(np.clip(scale, scale_min, scale_max))
        scale_trace[n] = scale

    return y_total, error, scale_trace


def calibrate_hybrid_dl_scale(model, val_scenarios: list, config: dict, device,
                              use_window_norm: bool, y_clip: float,
                              scale_min: float, scale_max: float, scale_steps: int,
                              energy_cap: float) -> tuple:
    """Find best DL scale on val scenarios by maximizing hybrid physical NR."""
    if not val_scenarios:
        return 1.0, {'scales': [1.0], 'mean_nr': [0.0]}

    lo = float(min(scale_min, scale_max))
    hi = float(max(scale_min, scale_max))
    steps = max(1, int(scale_steps))
    if steps == 1:
        scales = np.array([lo], dtype=np.float64)
    else:
        scales = np.linspace(lo, hi, steps, dtype=np.float64)

    score_sum = np.zeros_like(scales, dtype=np.float64)
    fo = config['model']['filter_order']

    print("  Calibrating hybrid DL scale on val set...")
    for idx, sc in enumerate(val_scenarios):
        fx = run_fxlms_online(sc, config)
        sc_res = dict(sc)
        sc_res['disturbance'] = fx['error'].copy()
        _, _, _, y_dl = evaluate_batched(
            model, sc_res, config, device,
            use_window_norm=use_window_norm,
            y_clip=y_clip,
            return_y=True,
        )
        y_dl_capped, _ = apply_hybrid_energy_cap(y_dl, fx['error'], energy_cap)
        for i, scale in enumerate(scales):
            y_total = fx['y_cancel'] + scale * y_dl_capped
            e_total = sc['disturbance'] - apply_path(y_total, sc['secondary_path'])
            score_sum[i] += noise_reduction_db(sc['disturbance'][fo:], e_total[fo:])
        if (idx + 1) % 10 == 0 or idx == 0 or (idx + 1) == len(val_scenarios):
            print(f"    Scale calibration progress: {idx + 1}/{len(val_scenarios)}")

    mean_scores = score_sum / len(val_scenarios)
    best_idx = int(np.argmax(mean_scores))
    return float(scales[best_idx]), {
        'scales': [float(s) for s in scales],
        'mean_nr': [float(v) for v in mean_scores],
    }


def quick_hybrid_nr_probe(model, probe_scenarios, probe_fxlms_cache, config, device,
                          use_window_norm: bool, y_clip: float,
                          hybrid_scale: float = 0.5) -> dict:
    """Fast per-epoch NR snapshot on a tiny held-out set (no logging).

    Runs batched DL eval + precomputed FxLMS cache to compute:
        hybrid_nr = NR of (y_fxlms + hybrid_scale * y_dl_capped)
        fxlms_nr  = NR of FxLMS alone
        delta_db  = hybrid_nr - fxlms_nr

    probe_fxlms_cache is precomputed once per seed to avoid CPU cost per epoch.
    Returns dict with mean values across probe_scenarios.
    """
    fo = config['model']['filter_order']
    hybrids, fxlmses = [], []
    for sc, fx in zip(probe_scenarios, probe_fxlms_cache):
        sc_res = dict(sc)
        sc_res['disturbance'] = fx['error'].copy()
        _, _, _, y_dl = evaluate_batched(
            model, sc_res, config, device,
            use_window_norm=use_window_norm,
            y_clip=y_clip,
            return_y=True,
        )
        y_dl_capped, _ = apply_hybrid_energy_cap(y_dl, fx['error'], 1.0)
        y_total = fx['y_cancel'] + hybrid_scale * y_dl_capped
        e_total = sc['disturbance'] - apply_path(y_total, sc['secondary_path'])
        hybrids.append(noise_reduction_db(sc['disturbance'][fo:], e_total[fo:]))
        fxlmses.append(fx['nr_db'])
    h_mean = float(np.mean(hybrids))
    f_mean = float(np.mean(fxlmses))
    return {'hybrid_nr': h_mean, 'fxlms_nr': f_mean, 'delta_db': h_mean - f_mean}


def evaluate_batched(model, scenario, config, device, use_window_norm: bool = True,
                     y_clip: float = 5.0, return_y: bool = False):
    """Batched physical evaluation."""
    model.eval()
    fo = config['model']['filter_order']
    d = scenario['disturbance']
    x = scenario['reference_mic']
    a = scenario['reference_accel']
    s = scenario['secondary_path']
    N = len(d)

    x_wins = np.lib.stride_tricks.sliding_window_view(x, fo)
    a_wins = np.lib.stride_tricks.sliding_window_view(a, fo)
    num_wins = x_wins.shape[0]
    y_cancel = np.zeros(N)

    with torch.no_grad():
        for start in range(0, num_wins, 4096):
            end = min(start + 4096, num_wins)
            x_t = torch.from_numpy(np.ascontiguousarray(x_wins[start:end])).float().to(device)
            a_t = torch.from_numpy(np.ascontiguousarray(a_wins[start:end])).float().to(device)
            if use_window_norm:
                x_t = normalize_windows(x_t)
                a_t = normalize_windows(a_t)
            y_pred, attn_w = model(x_t, a_t)
            if y_clip > 0:
                y_pred = torch.clamp(y_pred, -y_clip, y_clip)
            y_cancel[start + fo - 1:end + fo - 1] = y_pred.cpu().numpy().flatten()

    error = d - apply_path(y_cancel, s)
    nr = noise_reduction_db(d[fo:], error[fo:])
    band = frequency_band_nr(d[fo:], error[fo:], config['simulation']['fs'])
    attn_weights = None
    if attn_w is not None:
        attn_weights = attn_w.cpu().numpy()
    if return_y:
        return nr, band, attn_weights, y_cancel
    return nr, band, attn_weights


def run_fxlms_online(scenario, config):
    """Run FxLMS and return full signals + metrics."""
    fo = config['model']['filter_order']
    # Use much smaller step size for real data (long RIRs need careful adaptation)
    mu = config['model']['fxlms'].get('mu_real', 0.0001)
    d = scenario['disturbance']
    x = scenario['reference_mic']
    s = scenario['secondary_path']
    s_hat = scenario['secondary_path_estimate']
    N = len(d)

    ctrl = FxLMSController(fo, mu, s_hat)
    y_cancel = np.zeros(N)
    error = np.zeros(N)
    s_len = len(s)

    for n in range(fo, N):
        x_buf = x[n:n - fo:-1]
        y_cancel[n] = ctrl.predict(x_buf)
        conv_sum = sum(s[k] * y_cancel[n - k] for k in range(min(s_len, n + 1)))
        error[n] = d[n] - conv_sum
        ctrl.update(error[n], x_buf)

    nr = noise_reduction_db(d[fo:], error[fo:])
    band = frequency_band_nr(d[fo:], error[fo:], config['simulation']['fs'])
    return {
        'y_cancel': y_cancel,
        'error': error,
        'nr_db': nr,
        'band_nr': band,
    }


def evaluate_fxlms(scenario, config):
    """FxLMS baseline on a scenario."""
    result = run_fxlms_online(scenario, config)
    nr = result['nr_db']
    band = result['band_nr']
    return nr, band


def run_mm_fxlms_online(scenario, config, mu_accel_scale: float = 0.5):
    """Run MultiModalFxLMS (mic + accelerometer) and return NR metrics.

    Uses the same step size as FxLMS for the mic branch, and a scaled
    step size for the accel branch (accel signal has different SNR).

    This is the multi-modal adaptive baseline for fair comparison:
    DL multi-modal vs MM-FxLMS multi-modal (both use mic + accel).
    """
    fo = config['model']['filter_order']
    mu = config['model']['fxlms'].get('mu_real', 0.0001)
    mu_accel = mu * mu_accel_scale
    d = scenario['disturbance']
    x = scenario['reference_mic']
    a = scenario['reference_accel']
    s = scenario['secondary_path']
    s_hat = scenario['secondary_path_estimate']
    N = len(d)
    s_len = len(s)

    ctrl = MultiModalFxLMS(fo, mu, mu_accel, s_hat)
    y_cancel = np.zeros(N)
    error = np.zeros(N)

    for n in range(fo, N):
        x_buf = x[n:n - fo:-1]
        a_buf = a[n:n - fo:-1]
        y_cancel[n] = ctrl.predict(x_buf, a_buf)
        conv_sum = sum(s[k] * y_cancel[n - k] for k in range(min(s_len, n + 1)))
        error[n] = d[n] - conv_sum
        ctrl.update(error[n], x_buf, a_buf)

    nr = noise_reduction_db(d[fo:], error[fo:])
    band = frequency_band_nr(d[fo:], error[fo:], config['simulation']['fs'])
    return {'nr_db': nr, 'band_nr': band}


def update_results_registry(registry_root: str) -> tuple:
    """Build consolidated run registry CSV/JSON from all result files."""
    root = os.path.abspath(registry_root)
    pattern = os.path.join(root, '**', 'real_data_results.json')
    result_paths = sorted(glob.glob(pattern, recursive=True))
    rows = []
    for path in result_paths:
        try:
            with open(path, 'r') as f:
                payload = json.load(f)
        except Exception:
            continue
        metadata = payload.get('metadata', {})
        overall = payload.get('overall', {})
        rows.append({
            'run_path': os.path.relpath(path, os.getcwd()),
            'run_dir': os.path.relpath(os.path.dirname(path), os.getcwd()),
            'model_type': metadata.get('model_type'),
            'controller_type': metadata.get('controller_type'),
            'seed': metadata.get('seed'),
            'hybrid_residual': metadata.get('hybrid_residual'),
            'hybrid_adaptive_scale': metadata.get('hybrid_adaptive_scale'),
            'hybrid_energy_cap': metadata.get('hybrid_energy_cap'),
            'hybrid_scale_mu': metadata.get('hybrid_scale_mu'),
            'hybrid_scale_min': metadata.get('hybrid_scale_min'),
            'hybrid_scale_max': metadata.get('hybrid_scale_max'),
            'hybrid_dl_scale': metadata.get('hybrid_dl_scale'),
            'hybrid_dl_scale_mode': metadata.get('hybrid_dl_scale_mode'),
            'overall_dl_nr_mean': overall.get('dl_nr_mean'),
            'overall_fxlms_nr_mean': overall.get('fxlms_nr_mean'),
            'overall_mm_fxlms_nr_mean': overall.get('mm_fxlms_nr_mean'),
            'hybrid_minus_fxlms_db': overall.get('hybrid_minus_fxlms_db'),
            'dl_minus_mm_fxlms_db': overall.get('dl_minus_mm_fxlms_db'),
            'wins_vs_fxlms': overall.get('wins_vs_fxlms'),
            'losses_vs_fxlms': overall.get('losses_vs_fxlms'),
            'ties_vs_fxlms': overall.get('ties_vs_fxlms'),
            'wins_vs_mm_fxlms': overall.get('wins_vs_mm_fxlms'),
            'losses_vs_mm_fxlms': overall.get('losses_vs_mm_fxlms'),
            'ties_vs_mm_fxlms': overall.get('ties_vs_mm_fxlms'),
            'num_train_total': metadata.get('num_train_total'),
            'num_test_per_env': metadata.get('num_test_per_env'),
            'test_env_count': len(metadata.get('test_envs', []) or []),
            'registry_updated_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        })

    rows.sort(key=lambda r: (
        r['hybrid_minus_fxlms_db'] is None,
        -(r['hybrid_minus_fxlms_db'] or -1e9),
    ))

    fieldnames = [
        'run_path', 'run_dir', 'model_type', 'controller_type', 'seed',
        'hybrid_residual', 'hybrid_adaptive_scale', 'hybrid_energy_cap',
        'hybrid_scale_mu', 'hybrid_scale_min', 'hybrid_scale_max',
        'hybrid_dl_scale', 'hybrid_dl_scale_mode',
        'overall_dl_nr_mean', 'overall_fxlms_nr_mean', 'overall_mm_fxlms_nr_mean',
        'hybrid_minus_fxlms_db', 'dl_minus_mm_fxlms_db',
        'wins_vs_fxlms', 'losses_vs_fxlms', 'ties_vs_fxlms',
        'wins_vs_mm_fxlms', 'losses_vs_mm_fxlms', 'ties_vs_mm_fxlms',
        'num_train_total', 'num_test_per_env', 'test_env_count', 'registry_updated_utc',
    ]
    csv_path = os.path.join(root, 'paper_results_registry.csv')
    json_path = os.path.join(root, 'paper_results_registry.json')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    return csv_path, json_path, len(rows)


def log_to_mlflow_if_enabled(args, save_payload: dict, run_result_path: str,
                             registry_csv_path: str, registry_json_path: str):
    """Log run metadata/metrics/artifacts to MLflow when enabled."""
    if not args.mlflow_enable:
        return

    try:
        import mlflow
    except Exception as exc:
        print(f"MLflow not available, skipping MLflow logging: {exc}")
        return

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        metadata = save_payload.get('metadata', {})
        overall = save_payload.get('overall', {})
        per_environment = save_payload.get('per_environment', {})

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) and value is not None:
                mlflow.log_param(key, value)
        mlflow.log_param('run_result_path', run_result_path)

        for key, value in overall.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, float(value))

        if args.mlflow_log_per_environment:
            for env_name, env_metrics in per_environment.items():
                for key, value in env_metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        mlflow.log_metric(f"env/{env_name}/{key}", float(value))

        mlflow.log_artifact(run_result_path)
        if os.path.exists(registry_csv_path):
            mlflow.log_artifact(registry_csv_path)
        if os.path.exists(registry_json_path):
            mlflow.log_artifact(registry_json_path)
        print(f"MLflow logging complete: experiment={args.mlflow_experiment}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--model-type', default='fusion_attention')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-train', type=int, default=100,
                        help='Number of training scenarios')
    parser.add_argument('--num-test', type=int, default=5,
                        help='Number of test scenarios per DEMAND category')
    parser.add_argument('--filter-order', type=int, default=None,
                        help='Override filter order (default: from config, recommend 256 for real data)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load existing checkpoint (skip training)')
    parser.add_argument('--run-fxlms', action='store_true')
    parser.add_argument('--run-mm-fxlms', action='store_true',
                        help='Evaluate MultiModalFxLMS (mic+accel adaptive) baseline for multi-modal comparison')
    parser.add_argument('--holdout-per-category', type=int, default=1,
                        help='Number of DEMAND envs held out per category for testing')
    parser.add_argument('--extra-data-root', type=str, default=None,
                        help='Optional path to large audio datasets for augmentation')
    parser.add_argument('--extra-scenarios', type=int, default=0,
                        help='Number of additional scenarios sampled from extra datasets')
    parser.add_argument('--lambda-y', type=float, default=0.0,
                        help='Output-energy regularization coefficient')
    parser.add_argument('--y-clip', type=float, default=5.0,
                        help='Clamp DL anti-noise output to +/- this value (<=0 disables)')
    parser.add_argument('--window-norm', action='store_true',
                        help='Enable per-window input standardization')
    parser.add_argument('--hybrid-residual', action='store_true',
                        help='Train DL on FxLMS residual and evaluate combined y = y_fxlms + y_dl')
    parser.add_argument('--hybrid-dl-scale', type=float, default=-1.0,
                        help='Residual scale in hybrid mode (<0 auto-calibrates on val set)')
    parser.add_argument('--hybrid-dl-scale-min', type=float, default=0.0,
                        help='Minimum DL scale considered during auto-calibration')
    parser.add_argument('--hybrid-dl-scale-max', type=float, default=1.0,
                        help='Maximum DL scale considered during auto-calibration')
    parser.add_argument('--hybrid-dl-scale-steps', type=int, default=11,
                        help='Number of scale points between min/max for auto-calibration')
    parser.add_argument('--hybrid-energy-cap', type=float, default=1.0,
                        help='Max DL RMS relative to FxLMS residual RMS (<=0 disables cap)')
    parser.add_argument('--hybrid-adaptive-scale', action='store_true',
                        help='Enable online adaptive DL scaling during hybrid evaluation')
    parser.add_argument('--hybrid-scale-mu', type=float, default=0.005,
                        help='Step size for adaptive hybrid gain update')
    parser.add_argument('--hybrid-scale-init', type=float, default=-1.0,
                        help='Initial adaptive scale (<0 uses selected hybrid_dl_scale)')
    parser.add_argument('--hybrid-scale-min', type=float, default=0.0,
                        help='Lower bound for adaptive hybrid scale')
    parser.add_argument('--hybrid-scale-max', type=float, default=1.0,
                        help='Upper bound for adaptive hybrid scale')
    parser.add_argument('--low-freq-weight', type=float, default=0.0,
                        help='Extra weight on 20-500Hz band loss (0 disables, try 2.0)')
    parser.add_argument('--chunk-stride', type=int, default=None,
                        help='Override chunk stride (default 256 for dense overlap)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (default: from config)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override early stopping patience (default: from config)')
    parser.add_argument('--controller-type', type=str, default=None,
                        choices=['mlp', 'cnn', 'gru'],
                        help='Override controller backend (mlp/cnn/gru)')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    parser.add_argument('--fxlms-parallel-workers', type=int, default=0,
                        help='Number of worker processes for FxLMS residual pretraining (0=serial). '
                             'Recommended: 6-8 on M4, 1 on A100 (GPU already busy).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override simulation seed for multi-run sweeps')
    parser.add_argument('--registry-root', type=str, default=None,
                        help='Root dir to scan for run results and update paper registry (default: parent of save-dir)')
    parser.add_argument('--mlflow-enable', action='store_true',
                        help='Log run metadata, metrics, and artifacts to MLflow')
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None,
                        help='MLflow tracking URI (default uses MLflow env/default)')
    parser.add_argument('--mlflow-experiment', type=str, default='anc_multimodal_real_data',
                        help='MLflow experiment name')
    parser.add_argument('--mlflow-run-name', type=str, default=None,
                        help='Optional MLflow run name')
    parser.add_argument('--mlflow-log-per-environment', action='store_true',
                        help='Log per-environment metrics to MLflow (can be many metrics)')
    parser.add_argument('--save-dir', default='outputs/real_data')
    # Filter-bank architecture options
    parser.add_argument('--pretrain-filterbank', action='store_true',
                        help='Pre-train filter bank from FxLMS solutions (for filterbank_* models)')
    parser.add_argument('--filterbank-K', type=int, default=None,
                        help='Override number of filter bank entries (default: from config)')
    parser.add_argument('--filterbank-pretrain-scenarios', type=int, default=100,
                        help='Number of training scenarios used for filter-bank pretraining (<=0 uses full train set)')
    parser.add_argument('--mismatch-augment', action='store_true',
                        help='Enable secondary-path mismatch augmentation during training')
    parser.add_argument('--temperature-anneal', action='store_true',
                        help='Anneal filter-bank gating temperature from warm to cold')
    parser.add_argument('--trainable-filters', action='store_true',
                        help='Make filter bank FIR weights trainable (fine-tune from K-means init)')
    parser.add_argument('--fb-init-topk', action='store_true',
                        help='Initialize filter bank from top-K FxLMS solutions by NR (instead of K-means)')
    parser.add_argument('--mm-fxlms-mu-scale', type=float, default=0.5,
                        help='Accel step-size scale for MM-FxLMS (relative to mic step size, default 0.5)')
    args = parser.parse_args()
    if args.extra_scenarios > 0 and not args.extra_data_root:
        args.extra_data_root = '/Users/manojsingh/Desktop/PhD_Desktop/data_large'
        print(f"Using default extra-data root: {args.extra_data_root}")

    config = load_config(args.config)
    config['model']['type'] = args.model_type
    if args.filter_order:
        config['model']['filter_order'] = args.filter_order
    if args.controller_type:
        config['model']['controller']['type'] = args.controller_type
    if args.seed is not None:
        config['simulation']['seed'] = args.seed
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.patience is not None:
        config['training']['patience'] = args.patience
    set_seed(config['simulation']['seed'])
    device = get_device()
    use_window_norm = args.window_norm
    if args.hybrid_residual and not args.run_fxlms:
        print("Enabling FxLMS reporting because hybrid residual mode depends on FxLMS.")
        args.run_fxlms = True
    print(f"Device: {device}\n")

    # ── Generate real training data ──────────────────────────────────────
    print("=" * 70)
    print("Phase 1: Generating real-data scenarios")
    print("=" * 70)

    # Category-balanced split: held-out environments, not held-out categories.
    train_envs, test_envs = split_envs_by_category(
        seed=config['simulation']['seed'],
        holdout_per_category=args.holdout_per_category,
    )

    print(f"Training envs ({len(train_envs)}): {train_envs}")
    print(f"Test envs ({len(test_envs)}): {test_envs}")

    t0 = time.time()
    train_gen = RealDataGenerator(config, demand_envs=train_envs, seed=42)
    print(f"\nGenerating {args.num_train} training scenarios...")
    train_scenarios = train_gen.generate_scenarios(args.num_train)
    if args.extra_scenarios > 0:
        print(f"Augmenting with {args.extra_scenarios} external-data scenarios...")
        train_scenarios = add_external_scenarios(
            train_scenarios=train_scenarios,
            train_gen=train_gen,
            config=config,
            extra_root=args.extra_data_root,
            num_extra=args.extra_scenarios,
            seed=config['simulation']['seed'],
        )

    # Ensure all scenarios have same length (truncate to min)
    min_len = min(len(sc['reference_mic']) for sc in train_scenarios)
    for sc in train_scenarios:
        for key in ['noise_source', 'disturbance', 'reference_mic', 'reference_accel']:
            sc[key] = sc[key][:min_len]

    # Split train/val
    split_rng = np.random.default_rng(config['simulation']['seed'] + 123)
    split_rng.shuffle(train_scenarios)
    split = int(len(train_scenarios) * 0.85)
    train_set = train_scenarios[:split]
    val_set = train_scenarios[split:]
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}")
    print(f"  Data gen: {time.time() - t0:.1f}s")

    # Generate test scenarios from held-out environments
    print(f"\nGenerating test scenarios from held-out environments...")
    test_scenarios = []
    test_labels = []
    for env in test_envs:
        for i in range(args.num_test):
            env_gen = RealDataGenerator(config, demand_envs=[env], seed=9000 + i)
            sc = env_gen.generate_scenario(seed=9000 + i)
            test_scenarios.append(sc)
            test_labels.append(env)
    # Truncate test scenarios to consistent length
    if test_scenarios:
        test_min_len = min(len(sc['reference_mic']) for sc in test_scenarios)
        for sc in test_scenarios:
            for key in ['noise_source', 'disturbance', 'reference_mic', 'reference_accel']:
                sc[key] = sc[key][:test_min_len]
    print(f"  Test scenarios: {len(test_scenarios)} ({args.num_test} per env)")

    # ── Override filter-bank config from CLI ────────────────────────────
    if args.filterbank_K is not None:
        config.setdefault('model', {}).setdefault('filter_bank', {})['K'] = args.filterbank_K
    if args.trainable_filters:
        config.setdefault('model', {}).setdefault('filter_bank', {})['trainable_filters'] = True

    # ── Build model ──────────────────────────────────────────────────────
    model = build_model(config)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_info = " (trainable filters)" if args.trainable_filters else ""
    print(f"\nModel: {args.model_type}{trainable_info} | Params: {param_count:,}")

    # ── Filter-bank pre-training (if applicable) ─────────────────────────
    is_filterbank = args.model_type.startswith('filterbank_') or args.model_type == 'sfanc_baseline'
    if is_filterbank and (args.pretrain_filterbank or not args.checkpoint):
        from src.models.filter_bank import pretrain_filter_bank, pretrain_filter_bank_topk
        fb_K = config['model'].get('filter_bank', {}).get('K', 8)
        pretrain_count = len(train_set) if args.filterbank_pretrain_scenarios <= 0 else min(
            args.filterbank_pretrain_scenarios, len(train_set)
        )
        if args.fb_init_topk:
            print(f"\nPre-training filter bank (K={fb_K}, top-K by NR) "
                  f"using {pretrain_count}/{len(train_set)} scenarios...")
            centroids = pretrain_filter_bank_topk(train_set[:pretrain_count], config, K=fb_K)
        else:
            print(f"\nPre-training filter bank (K={fb_K}, K-means) from FxLMS solutions "
                  f"using {pretrain_count}/{len(train_set)} scenarios...")
            centroids = pretrain_filter_bank(train_set[:pretrain_count], config, K=fb_K)
        if hasattr(model, 'filter_bank'):
            model.filter_bank.load_centroids(centroids)
            print(f"  Loaded {fb_K} centroid filters into model.")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint} (skipping training)")
    else:
        # ── Train ────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("Phase 2: Training on real data")
        print("=" * 70)
        model.to(device)
        fo = config['model']['filter_order']
        chunk_size = config['training'].get('chunk_size', 512)
        chunk_stride = args.chunk_stride if args.chunk_stride is not None else config['training'].get('chunk_stride', 256)
        batch_size = config['training']['batch_size']
        lr = config['training']['lr']
        patience = config['training']['patience']
        print(f"  Window norm: {use_window_norm} | y_clip: {args.y_clip} | lambda_y: {args.lambda_y}")
        print(f"  LR: {lr} | Patience: {patience} | Chunk stride: {chunk_stride} | Grad accum: {args.grad_accum}")
        if args.low_freq_weight > 0:
            print(f"  Frequency-weighted loss: low_weight={args.low_freq_weight} (20-500Hz)")
        if args.hybrid_residual:
            print("  Hybrid residual mode: True (DL learns residual after FxLMS)")

        if args.hybrid_residual:
            print("  Building FxLMS residual targets for train/val...")
            batched = build_fxlms_residual_scenarios_batched(
                batches=[('train', train_set), ('val', val_set)],
                config=config,
                num_workers=args.fxlms_parallel_workers,
            )
            train_set_model = batched['train']
            val_set_model = batched['val']
        else:
            train_set_model = train_set
            val_set_model = val_set

        train_ds = RealDataChunkDataset(train_set_model, fo, chunk_size, chunk_stride)
        val_ds = RealDataChunkDataset(val_set_model, fo, chunk_size, chunk_stride)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Report path diversity (large mismatch means avg-path training is unreliable).
        max_s_len = max(len(sc['secondary_path']) for sc in train_set)
        padded = np.stack([
            np.pad(sc['secondary_path'], (0, max_s_len - len(sc['secondary_path'])))
            for sc in train_set
        ])
        mean_path = np.mean(padded, axis=0)
        print(f"  Secondary-path stats: mean ||S||={np.mean(np.linalg.norm(padded, axis=1)):.3f}, "
              f"||mean(S)||={np.linalg.norm(mean_path):.3f}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=config['training']['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_val_loss = float('inf')
        patience_counter = 0
        use_freq_loss = args.low_freq_weight > 0
        fs = config['simulation']['fs']

        # Build tiny probe set for per-epoch NR snapshot (reads same val_set)
        probe_scenarios = val_set[:min(3, len(val_set))] if args.hybrid_residual else []
        probe_fxlms_cache = []
        if probe_scenarios:
            print(f"  Building probe FxLMS cache on {len(probe_scenarios)} val scenarios...")
            for sc in probe_scenarios:
                probe_fxlms_cache.append(run_fxlms_online(sc, config))

        # ── Filter-bank training extras ─────────────────────────────────
        fb_cfg = config['model'].get('filter_bank', {})
        temp_start = fb_cfg.get('temperature_start', 2.0)
        temp_end = fb_cfg.get('temperature_end', 0.5)
        lambda_sparse = config['training'].get('lambda_sparse', 0.0)
        lambda_entropy = config['training'].get('lambda_entropy', 0.0)

        # Mismatch augmentation
        mismatch_aug = None
        if args.mismatch_augment:
            from src.training.augmentation import MismatchAugmentation
            mismatch_schedule = config['training'].get('mismatch_schedule', {})
            mismatch_aug = MismatchAugmentation(mismatch_schedule if mismatch_schedule else None)
            print(f"  Mismatch augmentation: enabled")
        if is_filterbank:
            print(f"  Temperature annealing: {temp_start} -> {temp_end}")
            if lambda_sparse > 0:
                print(f"  Gate sparsity loss: lambda={lambda_sparse}")
            if lambda_entropy > 0:
                print(f"  Gate entropy loss: lambda={lambda_entropy}")

        t0 = time.time()
        grad_accum = max(1, args.grad_accum)
        for epoch in range(args.epochs):
            # Temperature annealing for filter-bank gating
            if is_filterbank and hasattr(model, 'set_temperature') and args.temperature_anneal:
                epoch_frac = epoch / max(args.epochs - 1, 1)
                temp = temp_start + (temp_end - temp_start) * epoch_frac
                model.set_temperature(temp)

            # Train
            model.train()
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()
            for batch_idx, (audio, accel, dist, s_path) in enumerate(train_loader):
                B, C, F_dim = audio.shape
                audio = audio.view(B * C, F_dim).to(device)
                accel = accel.view(B * C, F_dim).to(device)
                dist = dist.to(device)
                s_path = s_path.to(device)
                if use_window_norm:
                    audio = normalize_windows(audio)
                    accel = normalize_windows(accel)

                y_pred, aux_weights = model(audio, accel)
                y_pred = y_pred.view(B, C)
                if args.y_clip > 0:
                    y_pred = torch.clamp(y_pred, -args.y_clip, args.y_clip)

                if use_freq_loss:
                    primary_loss = frequency_weighted_loss(
                        y_pred, dist, s_path, fs=fs, low_weight=args.low_freq_weight)
                else:
                    primary_loss = variable_secondary_path_loss(y_pred, dist, s_path)
                reg_loss = torch.mean(y_pred ** 2)
                loss = primary_loss + args.lambda_y * reg_loss

                # Filter-bank gate regularization
                if is_filterbank and aux_weights is not None and aux_weights.dim() == 2:
                    gate_w = aux_weights  # (B*C, K)
                    if lambda_sparse > 0:
                        loss = loss + lambda_sparse * gate_w.abs().mean()
                    if lambda_entropy > 0:
                        # Encourage confident selection (low entropy)
                        entropy = -(gate_w * (gate_w + 1e-8).log()).sum(dim=-1).mean()
                        loss = loss + lambda_entropy * entropy

                loss = loss / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * grad_accum
                n_batches += 1

            # Validate
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for audio, accel, dist, s_path in val_loader:
                    B, C, F = audio.shape
                    audio = audio.view(B * C, F).to(device)
                    accel = accel.view(B * C, F).to(device)
                    dist = dist.to(device)
                    s_path = s_path.to(device)
                    if use_window_norm:
                        audio = normalize_windows(audio)
                        accel = normalize_windows(accel)
                    y_pred, _ = model(audio, accel)
                    y_pred = y_pred.view(B, C)
                    if args.y_clip > 0:
                        y_pred = torch.clamp(y_pred, -args.y_clip, args.y_clip)
                    primary_loss = variable_secondary_path_loss(y_pred, dist, s_path)
                    val_loss += primary_loss.item()
                    val_batches += 1

            train_l = total_loss / max(n_batches, 1)
            val_l = val_loss / max(val_batches, 1)
            scheduler.step()

            # Flush MPS cache to prevent fragmentation over long training runs
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # Per-epoch NR snapshot (fast: 3 val scenarios with cached FxLMS)
            nr_snap_str = ""
            if probe_scenarios:
                snap = quick_hybrid_nr_probe(
                    model, probe_scenarios, probe_fxlms_cache, config, device,
                    use_window_norm=use_window_norm, y_clip=args.y_clip,
                    hybrid_scale=0.5,
                )
                nr_snap_str = (f" | Hybrid NR: {snap['hybrid_nr']:+.2f} dB "
                               f"| FxLMS: {snap['fxlms_nr']:+.2f} dB "
                               f"| Δ: {snap['delta_db']:+.3f} dB")

            print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_l:.4f} | "
                  f"Val: {val_l:.4f}{nr_snap_str}")

            if val_l < best_val_loss:
                best_val_loss = val_l
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_real.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"  Training: {time.time() - t0:.1f}s")
        # Load best
        best_path = os.path.join(args.save_dir, 'best_model_real.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True))

    model.to(device)

    selected_hybrid_scale = 1.0
    hybrid_scale_report = None
    if args.hybrid_residual:
        if args.hybrid_dl_scale < 0:
            selected_hybrid_scale, hybrid_scale_report = calibrate_hybrid_dl_scale(
                model=model,
                val_scenarios=val_set,
                config=config,
                device=device,
                use_window_norm=use_window_norm,
                y_clip=args.y_clip,
                scale_min=args.hybrid_dl_scale_min,
                scale_max=args.hybrid_dl_scale_max,
                scale_steps=args.hybrid_dl_scale_steps,
                energy_cap=args.hybrid_energy_cap,
            )
            print(f"  Auto-selected hybrid DL scale: {selected_hybrid_scale:.3f}")
        else:
            selected_hybrid_scale = float(args.hybrid_dl_scale)
            print(f"  Using manual hybrid DL scale: {selected_hybrid_scale:.3f}")
        print(f"  Hybrid energy cap (RMS ratio): {args.hybrid_energy_cap:.3f}")
        if args.hybrid_adaptive_scale:
            init_scale = selected_hybrid_scale if args.hybrid_scale_init < 0 else args.hybrid_scale_init
            print(f"  Adaptive hybrid scale: enabled | mu={args.hybrid_scale_mu:.5f} | "
                  f"init={init_scale:.3f} | range=[{args.hybrid_scale_min:.3f}, {args.hybrid_scale_max:.3f}]")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 3: Evaluation on held-out DEMAND environments")
    print("=" * 70)

    results = {}
    for sc, env_name in zip(test_scenarios, test_labels):
        bucket = results.setdefault(
            env_name,
            {
                'dl_nr': [],
                'dl_band': [],
                'fxlms_nr': [],
                'mm_fxlms_nr': [],
                'dl_cap_scale': [],
                'adaptive_scale_final': [],
                'adaptive_scale_mean': [],
            }
        )

        if args.hybrid_residual:
            fx = run_fxlms_online(sc, config)
            sc_res = dict(sc)
            sc_res['disturbance'] = fx['error'].copy()
            _, _, _, y_dl = evaluate_batched(
                model, sc_res, config, device,
                use_window_norm=use_window_norm,
                y_clip=args.y_clip,
                return_y=True,
            )
            y_dl_capped, cap_scale = apply_hybrid_energy_cap(y_dl, fx['error'], args.hybrid_energy_cap)
            if args.hybrid_adaptive_scale:
                init_scale = selected_hybrid_scale if args.hybrid_scale_init < 0 else args.hybrid_scale_init
                y_total, e_total, scale_trace = run_adaptive_hybrid_mix(
                    disturbance=sc['disturbance'],
                    secondary_path=sc['secondary_path'],
                    y_fxlms=fx['y_cancel'],
                    y_dl=y_dl_capped,
                    init_scale=init_scale,
                    mu=args.hybrid_scale_mu,
                    scale_min=min(args.hybrid_scale_min, args.hybrid_scale_max),
                    scale_max=max(args.hybrid_scale_min, args.hybrid_scale_max),
                )
                bucket['adaptive_scale_final'].append(float(scale_trace[-1]))
                bucket['adaptive_scale_mean'].append(float(np.mean(scale_trace)))
            else:
                y_total = fx['y_cancel'] + selected_hybrid_scale * y_dl_capped
                e_total = sc['disturbance'] - apply_path(y_total, sc['secondary_path'])
            fo = config['model']['filter_order']
            dl_nr = noise_reduction_db(sc['disturbance'][fo:], e_total[fo:])
            dl_band = frequency_band_nr(sc['disturbance'][fo:], e_total[fo:], config['simulation']['fs'])

            bucket['dl_nr'].append(dl_nr)
            bucket['dl_band'].append(dl_band)
            bucket['fxlms_nr'].append(fx['nr_db'])
            bucket['dl_cap_scale'].append(cap_scale)

            if args.run_mm_fxlms:
                mm_result = run_mm_fxlms_online(sc, config, args.mm_fxlms_mu_scale)
                bucket['mm_fxlms_nr'].append(mm_result['nr_db'])
        else:
            dl_nr, dl_band, _ = evaluate_batched(
                model, sc, config, device,
                use_window_norm=use_window_norm,
                y_clip=args.y_clip,
            )
            bucket['dl_nr'].append(dl_nr)
            bucket['dl_band'].append(dl_band)

            if args.run_fxlms:
                fx_nr, _ = evaluate_fxlms(sc, config)
                bucket['fxlms_nr'].append(fx_nr)

            if args.run_mm_fxlms:
                mm_result = run_mm_fxlms_online(sc, config, args.mm_fxlms_mu_scale)
                bucket['mm_fxlms_nr'].append(mm_result['nr_db'])

    # Print results
    model_label = "Hybrid NR" if args.hybrid_residual else "DL NR (mean)"
    has_fxlms = args.run_fxlms or args.hybrid_residual
    has_mm_fxlms = args.run_mm_fxlms
    col_width = 62 + (12 if has_fxlms else 0) + (12 if has_mm_fxlms else 0)
    print(f"\n{'Environment':<12} {'Category':<14} {model_label:>12} {'DL 20-200Hz':>12}", end="")
    if has_fxlms:
        print(f" {'FxLMS NR':>10}", end="")
    if has_mm_fxlms:
        print(f" {'MM-FxLMS':>10}", end="")
    print()
    print("-" * col_width)

    # Find category for each env
    env_to_cat = {}
    for cat, envs in DEMAND_ENVS.items():
        for e in envs:
            env_to_cat[e] = cat

    all_dl_nr = []
    all_fxlms_nr = []
    all_mm_fxlms_nr = []
    summary = {}
    win_count = 0
    loss_count = 0
    tie_count = 0
    mm_win_count = 0
    mm_loss_count = 0
    mm_tie_count = 0

    for env_name in sorted(results.keys()):
        r = results[env_name]
        dl_mean = np.mean(r['dl_nr'])
        dl_low = np.mean([b.get('20-200Hz', 0) for b in r['dl_band']])
        cat = env_to_cat.get(env_name, '?')
        all_dl_nr.extend(r['dl_nr'])

        row = f"{env_name:<12} {cat:<14} {dl_mean:>12.2f} {dl_low:>12.2f}"
        if has_fxlms and r['fxlms_nr']:
            fx_mean = np.mean(r['fxlms_nr'])
            all_fxlms_nr.extend(r['fxlms_nr'])
            row += f" {fx_mean:>10.2f}"
        if has_mm_fxlms and r['mm_fxlms_nr']:
            mm_mean = np.mean(r['mm_fxlms_nr'])
            all_mm_fxlms_nr.extend(r['mm_fxlms_nr'])
            row += f" {mm_mean:>10.2f}"
        print(row)

        summary[env_name] = {
            'category': cat,
            'dl_nr_mean': float(dl_mean),
            'dl_nr_std': float(np.std(r['dl_nr'])),
            'dl_20_200Hz_mean': float(dl_low),
        }
        if r['dl_cap_scale']:
            summary[env_name]['hybrid_cap_scale_mean'] = float(np.mean(r['dl_cap_scale']))
        if r['adaptive_scale_final']:
            summary[env_name]['adaptive_scale_final_mean'] = float(np.mean(r['adaptive_scale_final']))
        if r['adaptive_scale_mean']:
            summary[env_name]['adaptive_scale_mean'] = float(np.mean(r['adaptive_scale_mean']))
        if r['fxlms_nr']:
            fx_mean = float(np.mean(r['fxlms_nr']))
            summary[env_name]['fxlms_nr_mean'] = fx_mean
            delta = float(dl_mean - fx_mean)
            summary[env_name]['hybrid_minus_fxlms_db'] = delta
            if delta > 1e-9:
                win_count += 1
            elif delta < -1e-9:
                loss_count += 1
            else:
                tie_count += 1
        if r['mm_fxlms_nr']:
            mm_mean_env = float(np.mean(r['mm_fxlms_nr']))
            summary[env_name]['mm_fxlms_nr_mean'] = mm_mean_env
            mm_delta = float(dl_mean - mm_mean_env)
            summary[env_name]['dl_minus_mm_fxlms_db'] = mm_delta
            if mm_delta > 1e-9:
                mm_win_count += 1
            elif mm_delta < -1e-9:
                mm_loss_count += 1
            else:
                mm_tie_count += 1

    print("-" * col_width)
    overall_dl_mean = float(np.mean(all_dl_nr)) if all_dl_nr else 0.0
    overall_fxlms_mean = float(np.mean(all_fxlms_nr)) if all_fxlms_nr else None
    overall_mm_fxlms_mean = float(np.mean(all_mm_fxlms_nr)) if all_mm_fxlms_nr else None
    hybrid_minus_fxlms_db = (
        float(overall_dl_mean - overall_fxlms_mean)
        if overall_fxlms_mean is not None else None
    )
    dl_minus_mm_fxlms_db = (
        float(overall_dl_mean - overall_mm_fxlms_mean)
        if overall_mm_fxlms_mean is not None else None
    )
    overall = f"{'OVERALL':<12} {'':<14} {overall_dl_mean:>12.2f} {'':<12}"
    if overall_fxlms_mean is not None:
        overall += f" {overall_fxlms_mean:>10.2f}"
    if overall_mm_fxlms_mean is not None:
        overall += f" {overall_mm_fxlms_mean:>10.2f}"
    print(overall)
    if hybrid_minus_fxlms_db is not None:
        print(f"Delta DL vs FxLMS (single):  {hybrid_minus_fxlms_db:+.3f} dB | "
              f"W/L/T: {win_count}/{loss_count}/{tie_count}")
    if dl_minus_mm_fxlms_db is not None:
        print(f"Delta DL vs MM-FxLMS (multi): {dl_minus_mm_fxlms_db:+.3f} dB | "
              f"W/L/T: {mm_win_count}/{mm_loss_count}/{mm_tie_count}")

    # Save results
    save_payload = {
        'metadata': {
            'model_type': args.model_type,
            'controller_type': config['model']['controller']['type'],
            'hybrid_residual': args.hybrid_residual,
            'hybrid_dl_scale': selected_hybrid_scale if args.hybrid_residual else None,
            'hybrid_dl_scale_mode': (
                'auto' if args.hybrid_residual and args.hybrid_dl_scale < 0
                else 'manual' if args.hybrid_residual else None
            ),
            'hybrid_dl_scale_min': args.hybrid_dl_scale_min,
            'hybrid_dl_scale_max': args.hybrid_dl_scale_max,
            'hybrid_dl_scale_steps': args.hybrid_dl_scale_steps,
            'hybrid_energy_cap': args.hybrid_energy_cap if args.hybrid_residual else None,
            'hybrid_adaptive_scale': args.hybrid_adaptive_scale if args.hybrid_residual else None,
            'hybrid_scale_mu': args.hybrid_scale_mu if args.hybrid_residual else None,
            'hybrid_scale_init': args.hybrid_scale_init if args.hybrid_residual else None,
            'hybrid_scale_min': args.hybrid_scale_min if args.hybrid_residual else None,
            'hybrid_scale_max': args.hybrid_scale_max if args.hybrid_residual else None,
            'epochs': args.epochs,
            'lr': config['training']['lr'],
            'patience': config['training']['patience'],
            'seed': config['simulation']['seed'],
            'num_train_requested': args.num_train,
            'num_train_total': len(train_scenarios),
            'num_test_per_env': args.num_test,
            'filter_order': config['model']['filter_order'],
            'chunk_stride': args.chunk_stride,
            'grad_accum': args.grad_accum,
            'low_freq_weight': args.low_freq_weight,
            'window_norm': use_window_norm,
            'y_clip': args.y_clip,
            'lambda_y': args.lambda_y,
            'train_envs': train_envs,
            'test_envs': test_envs,
            'extra_data_root': args.extra_data_root,
            'extra_scenarios': args.extra_scenarios,
            'hybrid_dl_scale_report': hybrid_scale_report,
        },
        'overall': {
            'dl_nr_mean': overall_dl_mean,
            'fxlms_nr_mean': overall_fxlms_mean,
            'mm_fxlms_nr_mean': overall_mm_fxlms_mean,
            'hybrid_minus_fxlms_db': hybrid_minus_fxlms_db,
            'dl_minus_mm_fxlms_db': dl_minus_mm_fxlms_db,
            'wins_vs_fxlms': win_count if overall_fxlms_mean is not None else None,
            'losses_vs_fxlms': loss_count if overall_fxlms_mean is not None else None,
            'ties_vs_fxlms': tie_count if overall_fxlms_mean is not None else None,
            'wins_vs_mm_fxlms': mm_win_count if overall_mm_fxlms_mean is not None else None,
            'losses_vs_mm_fxlms': mm_loss_count if overall_mm_fxlms_mean is not None else None,
            'ties_vs_mm_fxlms': mm_tie_count if overall_mm_fxlms_mean is not None else None,
        },
        'per_environment': summary,
    }
    run_result_path = os.path.join(args.save_dir, 'real_data_results.json')
    with open(run_result_path, 'w') as f:
        json.dump(save_payload, f, indent=2)
    print(f"\nResults saved to {args.save_dir}/")

    registry_root = (
        os.path.abspath(args.registry_root)
        if args.registry_root else
        os.path.abspath(os.path.join(args.save_dir, os.pardir))
    )
    registry_csv_path, registry_json_path, registry_rows = update_results_registry(registry_root)
    print(f"Registry updated: {registry_rows} runs -> {registry_csv_path}")

    log_to_mlflow_if_enabled(
        args=args,
        save_payload=save_payload,
        run_result_path=run_result_path,
        registry_csv_path=registry_csv_path,
        registry_json_path=registry_json_path,
    )


if __name__ == '__main__':
    main()
