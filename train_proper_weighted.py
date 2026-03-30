#!/usr/bin/env python3
"""
Proper Sample-by-Sample ANC Training with Frequency-Weighted Sampling
======================================================================

Combines:
1. Sample-by-sample ANC simulation (proper temporal dynamics)
2. Frequency-weighted hard allocation + weighted fill sampling
3. Longer sequences (2048 samples = 128ms)

This should produce competitive NR values (~-12 dB) while using the
reviewer-approved sampling strategy.

Author: Manoj Singh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import os
import random
import time
import csv

# Import the causal model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.causal_conv import DeepANC_Causal

# =============================================================================
# Configuration (from sampling strategy)
# =============================================================================

DATA_ROOT = Path("/Users/manojsingh/Library/CloudStorage/OneDrive-Personal/PhD/data_large")
ESC50_PATH = DATA_ROOT / "ESC-50-master"
DEMAND_PATH = DATA_ROOT
RECORDED_PATH = DATA_ROOT / "recorded audio"
LIBRISPEECH_PATH = DATA_ROOT / "LibriSpeech/train-other-500"

SAMPLE_RATE = 16000
CHUNK_SIZE = 2048  # 128ms sequences (not 1-second chunks)
BATCH_SIZE = 32
EPOCHS = 15
BATCHES_PER_EPOCH = 35  # 35 × 32 = 1,120 samples

# Category definitions with weights
CATEGORIES = {
    1: {'name': 'Low-freq machinery', 'weight': 1.0},
    2: {'name': 'HVAC/Ventilation', 'weight': 1.0},
    3: {'name': 'Vehicle/Transport', 'weight': 0.8},
    4: {'name': 'Power tools', 'weight': 0.7},
    5: {'name': 'Natural ambient', 'weight': 0.7},
    6: {'name': 'Periodic non-stat.', 'weight': 0.6},
    7: {'name': 'Traffic/Urban', 'weight': 0.6},
    8: {'name': 'Speech', 'weight': 0.5},
    9: {'name': 'Transients', 'weight': 0.5},
}

# Hard allocation per batch
HARD_ALLOCATION = {
    'stationary': {'categories': [1, 2], 'samples': 13},
    'transport': {'categories': [3, 4], 'samples': 6},
    'transient': {'categories': [9], 'samples': 3},
}
WEIGHTED_FILL_SAMPLES = 10

# Category mappings (same as before)
ESC50_CATEGORY_MAP = {
    'engine': 1, 'washing_machine': 1, 'vacuum_cleaner': 2,
    'airplane': 3, 'train': 3, 'helicopter': 3, 'chainsaw': 4,
    'rain': 5, 'wind': 5, 'clock_alarm': 6, 'clock_tick': 6,
    'car_horn': 7, 'siren': 7, 'door_wood_knock': 9, 'glass_breaking': 9,
}

DEMAND_CATEGORY_MAP = {
    'DWASHING': 1, 'OOFFICE': 1, 'DKITCHEN': 1, 'TBUS': 3,
}

RECORDED_CATEGORY_MAP = {
    'Fan': 2, 'Refrigerator': 1, 'RO': 1, 'Puja Bell': 6,
    'Tap Water': 5, 'Walk': 9, 'Plastic Bottle': 9,
}

RECORDED_TRAIN_SESSIONS = ['Fan', 'Refrigerator', 'RO', 'Puja Bell']
RECORDED_TEST_SESSIONS = ['Tap Water', 'Walk', 'Plastic Bottle']

ESC50_TRAIN_FOLDS = [1, 2, 3]
ESC50_TEST_FOLDS = [4, 5]


# =============================================================================
# Data Loading (reuse from train_weighted_sampler.py)
# =============================================================================

def load_audio(path, target_sr=SAMPLE_RATE):
    """Load audio file and resample to target rate."""
    try:
        sr, audio = wavfile.read(path)
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.float64:
            audio = audio.astype(np.float32)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
        
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    except Exception as e:
        return None


def load_flac(path, target_sr=SAMPLE_RATE):
    """Load FLAC file."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        audio = audio.astype(np.float32)
        
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
        
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    except Exception as e:
        return None


def build_clip_database(is_train=True):
    """Build database of audio clips organized by category."""
    clips = {i: [] for i in range(1, 10)}
    
    # ESC-50
    esc50_meta = ESC50_PATH / "meta" / "esc50.csv"
    if esc50_meta.exists():
        with open(esc50_meta, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fold = int(row['fold'])
                category = row['category']
                
                if is_train and fold not in ESC50_TRAIN_FOLDS:
                    continue
                if not is_train and fold not in ESC50_TEST_FOLDS:
                    continue
                
                if category in ESC50_CATEGORY_MAP:
                    cat_id = ESC50_CATEGORY_MAP[category]
                    audio_path = ESC50_PATH / "audio" / row['filename']
                    if audio_path.exists():
                        clips[cat_id].append((str(audio_path), load_audio))
    
    # DEMAND (train only)
    if is_train:
        for demand_env, cat_id in DEMAND_CATEGORY_MAP.items():
            demand_dir = DEMAND_PATH / demand_env
            if demand_dir.exists():
                for wav_file in demand_dir.glob("*.wav"):
                    clips[cat_id].append((str(wav_file), load_audio))
    
    # Recorded Audio
    if RECORDED_PATH.exists():
        for wav_file in RECORDED_PATH.glob("*.wav"):
            filename = wav_file.stem
            session = None
            for s in RECORDED_CATEGORY_MAP.keys():
                if filename.startswith(s):
                    session = s
                    break
            
            if session is None:
                continue
            
            if is_train and session not in RECORDED_TRAIN_SESSIONS:
                continue
            if not is_train and session not in RECORDED_TEST_SESSIONS:
                continue
            
            cat_id = RECORDED_CATEGORY_MAP[session]
            clips[cat_id].append((str(wav_file), load_audio))
    
    # LibriSpeech (Category 8, train only)
    if is_train and LIBRISPEECH_PATH.exists():
        flac_files = list(LIBRISPEECH_PATH.glob("**/*.flac"))
        random.shuffle(flac_files)
        for flac_path in flac_files[:100]:
            clips[8].append((str(flac_path), load_flac))
    
    print(f"\n{'Train' if is_train else 'Test'} clip database:")
    total = 0
    for cat_id, cat_clips in clips.items():
        if cat_clips:
            print(f"  Category {cat_id} ({CATEGORIES[cat_id]['name']}): {len(cat_clips)} clips")
            total += len(cat_clips)
    print(f"  Total: {total} clips\n")
    
    return clips


class ANCDataset(Dataset):
    """Dataset that returns audio segments for ANC training."""
    
    def __init__(self, clips_db, chunk_size=CHUNK_SIZE):
        self.clips_db = clips_db
        self.chunk_size = chunk_size
        
        self.all_clips = []
        for cat_id, clips in clips_db.items():
            for path, load_func in clips:
                self.all_clips.append((path, load_func, cat_id))
        
        self.cache = {}
    
    def __len__(self):
        return len(self.all_clips)
    
    def get_by_category(self, cat_id):
        """Get indices for a specific category."""
        return [i for i, (_, _, c) in enumerate(self.all_clips) if c == cat_id]
    
    def __getitem__(self, idx):
        path, load_func, cat_id = self.all_clips[idx]
        
        if path not in self.cache:
            audio = load_func(path)
            if audio is None or len(audio) < self.chunk_size:
                return torch.zeros(1, self.chunk_size), cat_id
            self.cache[path] = audio
        
        audio = self.cache[path]
        
        # Random crop
        if len(audio) > self.chunk_size:
            start = random.randint(0, len(audio) - self.chunk_size)
            segment = audio[start:start + self.chunk_size]
        else:
            segment = np.pad(audio, (0, self.chunk_size - len(audio)))
        
        return torch.tensor(segment, dtype=torch.float32).unsqueeze(0), cat_id


class HardAllocationSampler(Sampler):
    """Hard allocation + weighted fill sampler."""
    
    def __init__(self, dataset, batches_per_epoch=BATCHES_PER_EPOCH):
        self.dataset = dataset
        self.batches_per_epoch = batches_per_epoch
        
        self.category_indices = {}
        for cat_id in range(1, 10):
            self.category_indices[cat_id] = dataset.get_by_category(cat_id)
        
        self.block_pools = {}
        for block_name, config in HARD_ALLOCATION.items():
            pool = []
            for cat_id in config['categories']:
                pool.extend(self.category_indices.get(cat_id, []))
            self.block_pools[block_name] = pool
        
        self._compute_weighted_probs()
    
    def _compute_weighted_probs(self):
        """Compute p_c ∝ w_c × n_c for weighted fill."""
        self.weighted_indices = []
        self.weighted_probs = []
        
        total_weight = 0
        for cat_id in range(1, 10):
            indices = self.category_indices.get(cat_id, [])
            n_c = len(indices)
            w_c = CATEGORIES[cat_id]['weight']
            weight = w_c * n_c
            
            if n_c > 0:
                self.weighted_indices.extend(indices)
                self.weighted_probs.extend([weight / n_c] * n_c)
                total_weight += weight
        
        if total_weight > 0:
            self.weighted_probs = np.array(self.weighted_probs) / sum(self.weighted_probs)
    
    def __iter__(self):
        """Generate batch indices for one epoch."""
        for _ in range(self.batches_per_epoch):
            batch_indices = []
            
            # Hard allocation
            for block_name, config in HARD_ALLOCATION.items():
                pool = self.block_pools.get(block_name, [])
                n_samples = config['samples']
                
                if pool:
                    sampled = random.choices(pool, k=n_samples)
                    batch_indices.extend(sampled)
                else:
                    if self.weighted_indices:
                        sampled = random.choices(
                            self.weighted_indices,
                            weights=self.weighted_probs,
                            k=n_samples
                        )
                        batch_indices.extend(sampled)
            
            # Weighted fill
            if self.weighted_indices:
                sampled = random.choices(
                    self.weighted_indices,
                    weights=self.weighted_probs,
                    k=WEIGHTED_FILL_SAMPLES
                )
                batch_indices.extend(sampled)
            
            yield batch_indices
    
    def __len__(self):
        return self.batches_per_epoch


def collate_batch(batch):
    """Collate function for DataLoader."""
    segments = [item[0] for item in batch]
    categories = [item[1] for item in batch]
    return torch.stack(segments), categories


# =============================================================================
# Acoustic Paths
# =============================================================================

def create_paths():
    """Create primary and secondary path FIR filters."""
    # Primary: lowpass 3.2 kHz
    P_ir = signal.firwin(32, 0.4)
    
    # Secondary: bandpass 200 Hz - 4 kHz
    S_ir = signal.firwin(32, [0.025, 0.5], pass_zero=False)
    
    return P_ir.astype(np.float32), S_ir.astype(np.float32)


# =============================================================================
# Training with Proper ANC Simulation
# =============================================================================

def train_epoch(model, dataloader, optimizer, P_ir, S_ir, alpha, device):
    """Train for one epoch with proper sample-by-sample ANC simulation."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    # Convert paths to torch tensors
    P_t = torch.tensor(P_ir, dtype=torch.float32).view(1, 1, -1).to(device)
    S_t = torch.tensor(S_ir, dtype=torch.float32).view(1, 1, -1).to(device)
    pad_p = len(P_ir) - 1
    pad_s = len(S_ir) - 1
    
    for batch_indices in dataloader.batch_sampler:
        # Get batch data
        batch_data = [dataloader.dataset[i] for i in batch_indices]
        x = torch.stack([item[0] for item in batch_data]).to(device)
        
        optimizer.zero_grad()
        
        # Primary path (disturbance)
        d = F.conv1d(F.pad(x, (pad_p, 0)), P_t)
        
        # Controller output
        y = model(x)
        
        # Align dimensions
        min_len = min(d.shape[-1], y.shape[-1])
        d = d[:, :, :min_len]
        y = y[:, :, :min_len]
        
        # Secondary path + saturation (Wiener model)
        v = F.conv1d(F.pad(y, (pad_s, 0)), S_t)[:, :, :min_len]
        z = torch.tanh(alpha * v)
        
        # Error signal
        e = d + z
        
        # Loss: minimize error power
        loss = torch.mean(e ** 2)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, dataset, P_ir, S_ir, alpha, device, n_samples=100):
    """Evaluate model on dataset."""
    model.eval()
    
    P_t = torch.tensor(P_ir, dtype=torch.float32).view(1, 1, -1).to(device)
    S_t = torch.tensor(S_ir, dtype=torch.float32).view(1, 1, -1).to(device)
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


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PROPER ANC TRAINING WITH FREQUENCY-WEIGHTED SAMPLING")
    print("=" * 70)
    
    start_time = time.time()
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Build clip databases
    train_clips = build_clip_database(is_train=True)
    test_clips = build_clip_database(is_train=False)
    
    # Create datasets
    train_dataset = ANCDataset(train_clips)
    test_dataset = ANCDataset(test_clips)
    
    print(f"Train dataset: {len(train_dataset)} clips")
    print(f"Test dataset: {len(test_dataset)} clips")
    
    # Create sampler and dataloader
    sampler = HardAllocationSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    # Create model
    model = DeepANC_Causal().to(device)
    print(f"\nModel: DeepANC_Causal ({model.n_params} parameters)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Acoustic paths
    P_ir, S_ir = create_paths()
    
    # Training
    alpha = 2.0
    best_nr = float('inf')
    patience = 7
    patience_counter = 0
    
    print(f"\nTraining with α={alpha} (proper sample-by-sample ANC)...")
    print("-" * 70)
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, P_ir, S_ir, alpha, device)
        
        # Evaluate
        train_nr = evaluate(model, train_dataset, P_ir, S_ir, alpha, device, n_samples=50)
        test_nr = evaluate(model, test_dataset, P_ir, S_ir, alpha, device, n_samples=50)
        
        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, "
              f"Train NR={train_nr:.2f} dB, Test NR={test_nr:.2f} dB")
        
        # Learning rate scheduling
        scheduler.step(train_nr)
        
        # Early stopping
        if train_nr < best_nr:
            best_nr = train_nr
            patience_counter = 0
            torch.save(model.state_dict(), 'causal_anc_proper_weighted.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("-" * 70)
    print(f"\nBest Training NR: {best_nr:.2f} dB")
    
    # Load best model and evaluate across alphas
    model.load_state_dict(torch.load('causal_anc_proper_weighted.pth'))
    
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ACROSS SATURATION LEVELS")
    print("=" * 70)
    
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = []
    
    for alpha in alphas:
        nr = evaluate(model, test_dataset, P_ir, S_ir, alpha, device, n_samples=100)
        results.append((alpha, nr))
        print(f"α={alpha:.1f}: NR = {nr:.2f} dB")
    
    # Save results
    with open('proper_weighted_training_results.txt', 'w') as f:
        f.write("Proper ANC Training with Frequency-Weighted Sampling\n")
        f.write("=" * 50 + "\n\n")
        for alpha, nr in results:
            f.write(f"α={alpha:.1f}: NR = {nr:.2f} dB\n")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training time: {elapsed/60:.1f} minutes")
    print(f"Best NR at α=2.0: {[r[1] for r in results if r[0]==2.0][0]:.2f} dB")
    print("\nResults saved to proper_weighted_training_results.txt")
    print("Model saved to causal_anc_proper_weighted.pth")


if __name__ == "__main__":
    main()
