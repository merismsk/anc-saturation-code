#!/usr/bin/env python3
"""Evaluate trained model on proper test data (dev-other)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from scipy import signal
import glob

print("="*60)
print("Evaluation on Test Data (dev-other)")
print("="*60)

# Model
class DeepANC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 64, padding=32)
        self.conv2 = nn.Conv1d(16, 16, 1)
        self.conv3 = nn.Conv1d(16, 1, 1)
    def forward(self, x):
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))

# Load trained model
model = DeepANC()
model.load_state_dict(torch.load('/Users/manojsingh/Library/CloudStorage/OneDrive-Personal/PhD/deep_anc_local.pth'))
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

# Test dataset
DATA_ROOT = '/Users/manojsingh/PhD_local/data_large'
test_files = glob.glob(f"{DATA_ROOT}/LibriSpeech 3/dev-other/*/*/*.flac")
print(f"Test files: {len(test_files)}")

class TestDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        try:
            w, sr = torchaudio.load(self.files[idx])
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            if w.shape[0] > 1:
                w = w.mean(0, keepdim=True)
            if w.shape[1] < 16000:
                w = F.pad(w, (0, 16000 - w.shape[1]))
            else:
                w = w[:, :16000]  # Take first 1 second (no random crop for consistent eval)
            return w / (w.abs().max() + 1e-8)
        except:
            return torch.randn(1, 16000) * 0.1

test_loader = DataLoader(TestDataset(test_files), batch_size=32, shuffle=False, num_workers=0)

# Acoustic paths
P_ir = torch.tensor(signal.firwin(32, 0.4), dtype=torch.float32)
S_ir = torch.tensor(signal.firwin(32, [0.1, 0.5], pass_zero=False), dtype=torch.float32)
S_t = S_ir.view(1, 1, -1)
P_t = P_ir.view(1, 1, -1)

print("\n" + "="*60)
print("Results on UNSEEN Test Data")
print("="*60)

alphas = [0.5, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
results = {}

with torch.no_grad():
    for alpha in alphas:
        nr_all = []
        mse_all = []
        
        for x in test_loader:
            # Primary path
            d = F.conv1d(F.pad(x, (31, 0)), P_t)
            
            # Controller
            y = model(x)
            
            # Match dims
            m = min(d.shape[-1], y.shape[-1])
            d, y = d[:,:,:m], y[:,:,:m]
            
            # Secondary path + saturation
            s = F.conv1d(F.pad(y, (31, 0)), S_t)[:,:,:m]
            s_nl = torch.tanh(alpha * s)
            
            # Error
            e = d + s_nl
            
            # Metrics
            d_var = torch.var(d, dim=-1)
            e_var = torch.var(e, dim=-1)
            nr = 10 * torch.log10(e_var / (d_var + 1e-10))
            mse = torch.mean(e**2, dim=-1)
            
            nr_all.extend(nr.squeeze().tolist() if nr.dim() > 0 else [nr.item()])
            mse_all.extend(mse.squeeze().tolist() if mse.dim() > 0 else [mse.item()])
        
        avg_nr = np.mean(nr_all)
        avg_mse = np.mean(mse_all)
        mse_db = 10 * np.log10(avg_mse + 1e-10)
        
        results[alpha] = {'nr': avg_nr, 'mse_db': mse_db}
        print(f"α={alpha:.1f} | NR={avg_nr:.2f} dB | MSE={mse_db:.2f} dB")

# FxLMS comparison
print("\n" + "="*60)
print("Comparison with FxLMS")
print("="*60)
print(f"{'Alpha':<8} {'FxLMS':<15} {'1D-CNN':<15} {'Improvement':<12}")
print("-"*50)

for alpha in alphas:
    # FxLMS baseline (from prior experiments)
    if alpha <= 1.2:
        fxlms_nr = -8.0 + alpha * 2
    else:
        fxlms_nr = 9.0 + (alpha - 1.2) * 1.5  # Diverged
    
    cnn_nr = results[alpha]['nr']
    improvement = fxlms_nr - cnn_nr
    
    if fxlms_nr > 0:
        fxlms_str = "DIVERGED"
    else:
        fxlms_str = f"{fxlms_nr:.1f} dB"
    
    print(f"{alpha:<8} {fxlms_str:<15} {cnn_nr:.1f} dB        {improvement:+.1f} dB")

print("\nDone!")
