# Deep Learning for Active Noise Control under Saturation

Code for the paper: **Active Noise Control in Saturated Non-Linear Acoustic Paths using Deep Convolutional Neural Networks**

## Overview

This repository contains the implementation of causal 1D-CNN, CNN+LSTM, and Transformer architectures for Active Noise Control (ANC) in saturated acoustic paths. The code includes training, evaluation, and streaming inference implementations.

## Requirements

- Python 3.9+
- PyTorch 1.10+
- NumPy
- SciPy
- Matplotlib (for visualization)

## Installation

```bash
pip install torch numpy scipy matplotlib
```

## Key Files

- `models/causal_conv.py` - Causal 1D-CNN implementation with streaming support
- `train_proper_weighted.py` - Main training script with weighted sampling
- `run_ablation_full.py` - Ablation study experiments (depth, width, data size)
- `evaluate_streaming_gap.py` - Streaming vs offline inference comparison
- `evaluate_test.py` - Model evaluation on test sets
- `download_data.py` - Data download utilities

## Usage

### Training

```bash
python train_proper_weighted.py
```

### Evaluation

```bash
python evaluate_test.py
```

### Ablation Studies

```bash
python run_ablation_full.py
```

## Model Architecture

The proposed 3-layer causal 1D-CNN:
- 1,329 parameters
- 1,296 MACs per sample
- 5.7 KB total memory (1.3 KB with INT8 quantization)
- O(1) inference time per sample

## Results

- **-14.4 dB NR** at α=2.0 (hard saturation)
- **7.2 dB improvement** over best adaptive baseline
- **<3 dB degradation** on held-out environments

## Citation

If you use this code, please cite our paper (citation will be added after publication).

## License

Code is provided for research purposes. See paper for full details.
