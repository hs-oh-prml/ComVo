# [ICLR 2026] Toward Complex-Valued Neural Networks for Waveform Generation
#### Hyung-Seok Oh, Deok-Hyeon Cho, Seung-Bin Kim and Seong-Whan Lee

This repository contains the official implementation of ComVo,
a complex-valued neural vocoder for waveform generation based on iSTFT.

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=U4GXPqm3Va)
[![Demo](https://img.shields.io/badge/Demo-Audio_Samples-green)](https://hs-oh-prml.github.io/ComVo/)
[![GitHub Stars](https://img.shields.io/github/stars/hs-oh-prml/ComVo?style=social)](https://github.com/hs-oh-prml/ComVo)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ComVo_Collection-yellow)](https://huggingface.co/collections/hsoh/comvo)

<p align="center">
  <img src="assets/architecture.png" alt="">
  <br>
  <em>Overall architecture of ComVo</em>
</p>


## Abstract

Neural vocoders have recently advanced waveform generation, yielding natural and expressive audio.
Among these approaches, iSTFT-based vocoders have gained attention.
They predict a complex-valued spectrogram and then synthesize the waveform via iSTFT, thereby avoiding redundant, computationally expensive upsampling.
However, current approaches use real-valued networks that process the real and imaginary parts independently.
This separation limits their ability to capture the inherent structure of complex spectrograms.
We present ComVo, a complex-valued neural vocoder whose generator and discriminator use native complex arithmetic.
This enables an adversarial training framework that provides structured feedback directly in the complex domain.
To guide phase transformations in a structured manner, we introduce phase quantization, which discretizes phase values and regularizes the training process.
Finally, we propose a block-matrix computation scheme to improve training efficiency by reducing redundant operations.
Experiments demonstrate that ComVo achieves higher synthesis quality than comparable real-valued baselines, and that its block-matrix scheme reduces training time by 25%.
Audio samples and code are available at [https://hs-oh-prml.github.io/ComVo/](https://hs-oh-prml.github.io/ComVo/).


### Installation

```bash
pip install -r requirements
```

#### Recommended environment

- Python >= 3.8
- PyTorch >= 2.0
- CUDA-enabled GPU

## Notes
- `from_waveform` performs end-to-end inference from raw audio.
- `build_feature_extractor` allows explicit feature extraction, useful for debugging or custom pipelines.

## Pretrained checkpoints

We provide pretrained ComVo checkpoints for quick inference:

| Model | Parameters | Hugging Face | Checkpoint | Sampling rate | n_fft | hop size | win length |
|------|-----------|--------------|------------|---------------|-------|----------|------------|
| Base | 13.28M | [![HF Model](https://img.shields.io/badge/HF-Model-yellow)](https://huggingface.co/hsoh/ComVo-base) | [![Download](https://img.shields.io/badge/Checkpoint-Download-orange)](https://works.do/xM2ttS4) | 24 kHz | 100 | 12000 | 256 |
| Large | 114.56M | [![HF Model](https://img.shields.io/badge/HF-Model-yellow)](https://huggingface.co/hsoh/ComVo-large) | [![Download](https://img.shields.io/badge/Checkpoint-Download-orange)](https://works.do/FYuHg2z) | 24 kHz | 100 | 12000 | 256 |

| Model | UTMOS ↑ | PESQ (wb) ↑ | PESQ (nb) ↑ | MRSTFT ↓ | Periodicity RMSE ↓ | V/UV F1 ↑ |
|------|--------|-------------|-------------|----------|--------------------|-----------|
| Base | 3.6744 | 3.8219 | 4.0727 | 0.8580 | 0.0925 | 0.9602 |
| Large | 3.7618 | 3.9993 | 4.1639 | 0.8227 | 0.0751 | 0.9706 |


## Hugging Face Inference
1. End-to-end inference from waveform
```python
import torch
import torchaudio
from hf_model import ComVoHF

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = ComVoHF.from_pretrained("hsoh/ComVo-base")
model = model.eval().to(device)

# Load audio
wav, sr = torchaudio.load("input.wav")

# Convert to mono if needed
if wav.size(0) > 1:
    wav = wav.mean(dim=0, keepdim=True)

wav = wav.to(device)  # [1, T]

# Inference
with torch.inference_mode():
    audio = model.from_waveform(wav)

# Save output
torchaudio.save("output.wav", audio.squeeze(0).cpu(), model.sample_rate)
```

2. Inference from extracted features
```python
import torch
import torchaudio
from hf_model import ComVoHF

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = ComVoHF.from_pretrained("hsoh/ComVo-base")
model = model.eval().to(device)

# Build feature extractor
feature_extractor = model.build_feature_extractor().to(device)

# Load audio
wav, sr = torchaudio.load("input.wav")

# Convert to mono if needed
if wav.size(0) > 1:
    wav = wav.mean(dim=0, keepdim=True)

wav = wav.to(device)  # [1, T]

# Feature extraction + inference
with torch.inference_mode():
    features = feature_extractor(wav)
    audio = model(features)

# Save output
torchaudio.save("output.wav", audio.squeeze(0).cpu(), model.sample_rate)
```

## Inference

```bash
python infer.py -c configs/configs.yaml --ckpt /path/to/comvo.ckpt --wavfile /path/to/input.wav --out_dir ./results
```

## Train

```bash
python train.py -c configs/configs.yaml
```

Hyperparameters are specified in `configs/configs.yaml`.

## Inference

```bash
python infer.py -c $CONFIG --ckpt=$CKPT --wavfile=$FILE_NAME --out_dir $OUTPUT_DIR
```

## Citation

```bibtex
@inproceedings{
  oh2026toward,
  title={Toward Complex-Valued Neural Networks for Waveform Generation},
  author={Hyung-Seok Oh and Deok-Hyeon Cho and Seung-Bin Kim and Seong-Whan Lee},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=U4GXPqm3Va}
}
```
