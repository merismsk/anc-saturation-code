#!/usr/bin/env python3
"""
Full-Pipeline Ablation Study for Paper 1
=========================================

Runs ablation experiments using the SAME training protocol as train_proper_weighted.py:
- 566 clips (DEMAND + ESC-50 + LibriSpeech + self-recorded)
- Frequency-weighted sampling (hard allocation + weighted fill)
- Adam lr=0.002, 15 epochs, early stopping patience=7
- NR metric: 10*log10(Var(e)/Var(d))
- α = 2.0

Ablation variables:
1. Depth: L2, L3 (proposed), L4, L6
2. Width/Kernel: F16/K32, F32/K64 (proposed), F64/K128
3. Data size: 100, 200, 400, 566 clips

Author: Manoj Singh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import os
import random
import time
import json
import sys

# Reuse data loading from the main training script
sys.path.insert(0, str(Path(__file__).parent))
from models.causal_conv import CausalConv1d

# =============================================================================
# Configuration (matches train_proper_weighted.py exactly)
# =============================================================================

DATA_ROOT = Path("/Users/manojsingh/Desktop/PhD/data_large")
ESC50_PATH = DATA_ROOT / "ESC-50-master"
DEMAND_PATH = DATA_ROOT
RECORDED_PATH = DATA_ROOT / "recorded audio"
LIBRISPEECH_PATH = DATA_ROOT / "LibriSpeech/train-other-500"

SAMPLE_RATE = 16000
CHUNK_SIZE = 2048
BATCH_SIZE = 32
EPOCHS = 15
BATCHES_PER_EPOCH = 35
ALPHA = 2.0

RESULTS_PATH = Path("/Users/manojsingh/Library/CloudStorage/OneDrive-Personal/PhD/ablation_full_results.json")


# =============================================================================
# Parameterized Model (replaces fixed DeepANC_Causal)
# =============================================================================

class DeepANC_Ablation(nn.Module):
    """Configurable causal 1D-CNN for ablation."""

    def __init__(self, n_layers=3, n_filters=16, kernel_size=64):
        super().__init__()
        layers = []

        # First layer: causal conv with receptive field
        layers.append(CausalConv1d(1, n_filters, kernel_size=kernel_size))
        layers.append(nn.ReLU())

        # Hidden layers: 1x1 convs
        for _ in range(n_layers - 2):
            layers.append(nn.Conv1d(n_filters, n_filters, kernel_size=1))
            layers.append(nn.ReLU())

        # Output layer: 1x1 conv
        layers.append(nn.Conv1d(n_filters, 1, kernel_size=1))

        self.net = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Dataset (copied from train_proper_weighted.py for self-containment)
# =============================================================================

# Category mapping for frequency-weighted sampling
CATEGORY_MAP = {
    1: {"name": "Low-freq machinery", "weight": 1.0},
    2: {"name": "HVAC/Ventilation", "weight": 1.0},
    3: {"name": "Vehicle/Transport", "weight": 0.8},
    4: {"name": "Power tools", "weight": 0.7},
    5: {"name": "Natural ambient", "weight": 0.7},
    6: {"name": "Periodic non-stationary", "weight": 0.6},
    7: {"name": "Traffic/Urban", "weight": 0.6},
    8: {"name": "Speech", "weight": 0.5},
    9: {"name": "Transients", "weight": 0.5},
}

# ESC-50 category to our category mapping
ESC50_TO_CAT = {
    # Cat 1: Low-freq machinery
    "washing_machine": 1, "engine": 1,
    # Cat 2: HVAC
    "vacuum_cleaner": 2, "air_conditioner": 2,
    # Cat 3: Vehicle
    "train": 3, "helicopter": 3, "airplane": 3,
    # Cat 4: Power tools
    "chainsaw": 4, "hand_saw": 4,
    # Cat 5: Natural
    "rain": 5, "sea_waves": 5, "wind": 5, "thunderstorm": 5,
    "water_drops": 5, "crackling_fire": 5,
    # Cat 6: Periodic
    "clock_tick": 6, "clock_alarm": 6,
    # Cat 7: Traffic/Urban
    "siren": 7, "car_horn": 7,
    # Cat 9: Transients
    "door_knock": 9, "mouse_click": 9, "keyboard_typing": 9,
    "can_opening": 9, "glass_breaking": 9,
}


def load_audio(path, target_sr=16000):
    """Load audio file and resample to target_sr."""
    try:
        sr, data = wavfile.read(str(path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float64:
            data = data.astype(np.float32)

        if len(data.shape) > 1:
            data = data.mean(axis=1)

        if sr != target_sr:
            n_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, n_samples)

        # Normalize
        mx = np.max(np.abs(data))
        if mx > 0:
            data = data / mx

        return data.astype(np.float32)
    except Exception as e:
        return None


def build_clip_database(is_train=True):
    """Build clip database with category labels."""
    clips = []
    split_name = "train" if is_train else "test"
    print(f"\nBuilding {split_name} clip database...")

    # ESC-50
    print(f"  Loading ESC-50 ({split_name})...")
    esc_meta = ESC50_PATH / "meta" / "esc50.csv"
    if esc_meta.exists():
        import csv
        with open(esc_meta) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fold = int(row['fold'])
                if is_train and fold > 3:
                    continue
                if not is_train and fold <= 3:
                    continue

                cat_name = row['category']
                if cat_name not in ESC50_TO_CAT:
                    continue

                cat = ESC50_TO_CAT[cat_name]
                audio_path = ESC50_PATH / "audio" / row['filename']
                if audio_path.exists():
                    clips.append({"path": str(audio_path), "category": cat})
    print(f"    Found {len([c for c in clips if 'ESC' in str(c['path'])])} ESC-50 clips")

    # DEMAND (train only)
    if is_train:
        print(f"  Loading DEMAND...")
        demand_envs = {"DWASHING": 1, "DKITCHEN": 1, "TBUS": 7, "OOFFICE": 7}
        for env, cat in demand_envs.items():
            env_path = DEMAND_PATH / env
            if env_path.exists():
                wavs = list(env_path.glob("*.wav"))
                print(f"    {env}: found {len(wavs)} wav files")
                for wav in sorted(wavs)[:20]:
                    clips.append({"path": str(wav), "category": cat})
        demand_count = len([c for c in clips if any(env in str(c['path']) for env in demand_envs.keys())])
        print(f"    Total DEMAND clips: {demand_count}")

    # Self-recorded
    print(f"  Loading self-recorded audio...")
    if RECORDED_PATH.exists():
        rec_cats = {
            "Fan": 2, "Refrigerator": 2, "RO": 2,
            "Tap Water": 5, "Walking": 9, "Puja Bell": 9, "Plastic Bottle": 9
        }
        for subdir in sorted(RECORDED_PATH.iterdir()):
            if subdir.is_dir() and subdir.name in rec_cats:
                cat = rec_cats[subdir.name]
                is_train_cat = subdir.name in ["Fan", "Refrigerator", "RO", "Puja Bell"]
                if is_train == is_train_cat:
                    for wav in sorted(subdir.glob("*.wav")):
                        clips.append({"path": str(wav), "category": cat})
        rec_count = len([c for c in clips if 'recorded' in str(c['path']).lower()])
        print(f"    Found {rec_count} recorded audio clips")

    # LibriSpeech (train only, 100 clips)
    if is_train:
        print(f"  Loading LibriSpeech...")
    if is_train and LIBRISPEECH_PATH.exists():
        count = 0
        # Optimize: iterate through speaker dirs instead of rglob
        for speaker_dir in sorted(LIBRISPEECH_PATH.iterdir())[:20]:
            if speaker_dir.is_dir():
                for chapter_dir in speaker_dir.iterdir():
                    if chapter_dir.is_dir():
                        for wav in sorted(chapter_dir.glob("*.flac")):
                            clips.append({"path": str(wav), "category": 8})
                            count += 1
                            if count >= 100:
                                break
                    if count >= 100:
                        break
            if count >= 100:
                break
        print(f"    Found {count} LibriSpeech clips")

    print(f"  Total: {len(clips)} clips for {split_name}")
    return clips


class ANCDataset(Dataset):
    """ANC dataset that loads and chunks audio."""

    def __init__(self, clip_list, chunk_size=CHUNK_SIZE):
        self.clips = clip_list
        self.chunk_size = chunk_size
        self.audio_cache = {}

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]

        if idx not in self.audio_cache:
            audio = load_audio(clip["path"])
            if audio is None or len(audio) < self.chunk_size:
                audio = np.zeros(self.chunk_size, dtype=np.float32)
            self.audio_cache[idx] = audio

        audio = self.audio_cache[idx]

        # Random chunk
        if len(audio) > self.chunk_size:
            start = random.randint(0, len(audio) - self.chunk_size)
            chunk = audio[start:start + self.chunk_size]
        else:
            chunk = audio[:self.chunk_size]

        x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
        return x, clip["category"]


class HardAllocationSampler:
    """Frequency-weighted batch sampler with hard allocation."""

    def __init__(self, dataset, batch_size=BATCH_SIZE, batches_per_epoch=BATCHES_PER_EPOCH):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        # Group indices by category
        self.cat_indices = {}
        for i, clip in enumerate(dataset.clips):
            cat = clip["category"]
            if cat not in self.cat_indices:
                self.cat_indices[cat] = []
            self.cat_indices[cat].append(i)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []

            # Hard allocation: stationary (cats 1-2): 13, transport (3-4): 6, transient (9): 3
            for cat, count in [(1, 7), (2, 6), (3, 3), (4, 3), (9, 3)]:
                if cat in self.cat_indices and len(self.cat_indices[cat]) > 0:
                    batch.extend(random.choices(self.cat_indices[cat], k=count))

            # Weighted fill for remaining slots
            remaining = self.batch_size - len(batch)
            if remaining > 0 and len(self.cat_indices) > 0:
                all_cats = list(self.cat_indices.keys())
                weights = [CATEGORY_MAP.get(c, {}).get("weight", 0.5) for c in all_cats]
                total_w = sum(weights)
                if total_w > 0:
                    weights = [w / total_w for w in weights]
                    for _ in range(remaining):
                        cat = random.choices(all_cats, weights=weights, k=1)[0]
                        if len(self.cat_indices[cat]) > 0:
                            batch.append(random.choice(self.cat_indices[cat]))

            yield batch[:self.batch_size]

    def __len__(self):
        return self.batches_per_epoch


# =============================================================================
# Acoustic Paths (matches train_proper_weighted.py)
# =============================================================================

def create_paths():
    P_ir = signal.firwin(32, 0.4).astype(np.float32)
    S_ir = signal.firwin(32, [0.025, 0.5], pass_zero=False).astype(np.float32)
    return P_ir, S_ir


# =============================================================================
# Training & Evaluation (matches train_proper_weighted.py)
# =============================================================================

def train_epoch(model, dataloader, optimizer, P_ir, S_ir, alpha, device):
    model.train()
    total_loss = 0
    n_batches = 0

    P_t = torch.tensor(P_ir).view(1, 1, -1).to(device)
    S_t = torch.tensor(S_ir).view(1, 1, -1).to(device)
    pad_p = len(P_ir) - 1
    pad_s = len(S_ir) - 1

    for batch_indices in dataloader.batch_sampler:
        batch_data = [dataloader.dataset[i] for i in batch_indices]
        x = torch.stack([item[0] for item in batch_data]).to(device)

        optimizer.zero_grad()

        d = F.conv1d(F.pad(x, (pad_p, 0)), P_t)
        y = model(x)

        min_len = min(d.shape[-1], y.shape[-1])
        d = d[:, :, :min_len]
        y = y[:, :, :min_len]

        v = F.conv1d(F.pad(y, (pad_s, 0)), S_t)[:, :, :min_len]
        z = torch.tanh(alpha * v)
        e = d + z

        loss = torch.mean(e ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_nr(model, dataset, P_ir, S_ir, alpha, device, n_samples=100):
    """Evaluate NR = 10*log10(Var(e)/Var(d))."""
    model.eval()

    P_t = torch.tensor(P_ir).view(1, 1, -1).to(device)
    S_t = torch.tensor(S_ir).view(1, 1, -1).to(device)
    pad_p = len(P_ir) - 1
    pad_s = len(S_ir) - 1

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:n_samples]

    total_var_e = 0
    total_var_d = 0

    with torch.no_grad():
        for idx in indices:
            x, _ = dataset[idx]
            x = x.unsqueeze(0).to(device)

            d = F.conv1d(F.pad(x, (pad_p, 0)), P_t)
            y = model(x)

            min_len = min(d.shape[-1], y.shape[-1])
            d = d[:, :, :min_len]
            y = y[:, :, :min_len]

            v = F.conv1d(F.pad(y, (pad_s, 0)), S_t)[:, :, :min_len]
            z = torch.tanh(alpha * v)
            e = d + z

            total_var_e += torch.var(e).item()
            total_var_d += torch.var(d).item()

    nr = 10 * np.log10(total_var_e / (total_var_d + 1e-10))
    return nr


def train_and_evaluate(model, train_dataset, test_dataset, P_ir, S_ir, config_name):
    """Full training pipeline for one ablation config. Returns test NR."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    sampler = HardAllocationSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=lambda batch: (torch.stack([b[0] for b in batch]), [b[1] for b in batch]),
        num_workers=0
    )

    best_nr = float('inf')
    patience_counter = 0
    best_state = None

    print(f"\n  Training {config_name} ({model.n_params} params)...")

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, P_ir, S_ir, ALPHA, device)
        test_nr = evaluate_nr(model, test_dataset, P_ir, S_ir, ALPHA, device, n_samples=100)

        scheduler.step(test_nr)

        if test_nr < best_nr:
            best_nr = test_nr
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test NR={test_nr:.2f} dB")

    # Final evaluation with best model
    model.load_state_dict(best_state)
    final_nr = evaluate_nr(model, test_dataset, P_ir, S_ir, ALPHA, device, n_samples=len(test_dataset))
    print(f"    Final NR: {final_nr:.2f} dB (params: {model.n_params})")

    return final_nr


# =============================================================================
# Main Ablation
# =============================================================================

def main():
    print("=" * 70)
    print("FULL-PIPELINE ABLATION STUDY (NR metric, α=2.0)")
    print("=" * 70)

    start_time = time.time()

    # Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Check data paths exist
    print("\nChecking data paths...")
    paths_to_check = [
        ("ESC-50", ESC50_PATH),
        ("DEMAND", DEMAND_PATH),
        ("Recorded Audio", RECORDED_PATH),
        ("LibriSpeech", LIBRISPEECH_PATH)
    ]
    for name, path in paths_to_check:
        exists = path.exists() if hasattr(path, 'exists') else False
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")

    # Build full dataset
    train_clips = build_clip_database(is_train=True)
    test_clips = build_clip_database(is_train=False)

    full_train_dataset = ANCDataset(train_clips)
    test_dataset = ANCDataset(test_clips)

    print(f"\nFull train: {len(full_train_dataset)} clips")
    print(f"Test: {len(test_dataset)} clips")

    if len(full_train_dataset) == 0:
        print("\n❌ ERROR: No training data found!")
        print("Please check that the data paths are correct and data files exist.")
        print(f"\nExpected data root: {DATA_ROOT}")
        return

    if len(test_dataset) == 0:
        print("\n⚠️  WARNING: No test data found!")
        print("Evaluation will not be possible.")

    P_ir, S_ir = create_paths()

    results = {}

    # -----------------------------------------------------------------
    # 1. DEPTH ABLATION (fix F=16, K=64 as in proposed model)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. DEPTH ABLATION (F=16, K=64)")
    print("=" * 70)

    for n_layers in [2, 3, 4, 6]:
        name = f"L{n_layers}"
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model = DeepANC_Ablation(n_layers=n_layers, n_filters=16, kernel_size=64)
        nr = train_and_evaluate(model, full_train_dataset, test_dataset, P_ir, S_ir, name)
        results[name] = {"nr_db": round(nr, 2), "params": model.n_params}

    # -----------------------------------------------------------------
    # 2. WIDTH/KERNEL ABLATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. WIDTH/KERNEL ABLATION")
    print("=" * 70)

    for n_filters, kernel_size in [(16, 32), (32, 64), (64, 128)]:
        name = f"F{n_filters}_K{kernel_size}"
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model = DeepANC_Ablation(n_layers=3, n_filters=n_filters, kernel_size=kernel_size)
        nr = train_and_evaluate(model, full_train_dataset, test_dataset, P_ir, S_ir, name)
        results[name] = {"nr_db": round(nr, 2), "params": model.n_params}

    # -----------------------------------------------------------------
    # 3. DATA SIZE ABLATION (using proposed L3/F16/K64 architecture)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DATA SIZE ABLATION (L3, F16, K64)")
    print("=" * 70)

    for n_clips in [100, 200, 400, len(train_clips)]:
        name = f"data_{n_clips}"
        random.seed(42); np.random.seed(42); torch.manual_seed(42)

        # Subsample training clips
        subset = train_clips[:n_clips] if n_clips < len(train_clips) else train_clips
        subset_dataset = ANCDataset(subset)

        model = DeepANC_Ablation(n_layers=3, n_filters=16, kernel_size=64)
        nr = train_and_evaluate(model, subset_dataset, test_dataset, P_ir, S_ir, name)
        results[name] = {"nr_db": round(nr, 2), "params": model.n_params, "n_clips": n_clips}

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    for name, res in results.items():
        print(f"  {name:20s}: NR = {res['nr_db']:+.2f} dB  (params: {res['params']})")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
