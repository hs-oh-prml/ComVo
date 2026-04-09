import torch
import torchaudio
from hf_model import ComVoHF

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ComVoHF.from_pretrained("hsoh/ComVo-base")
model = model.eval().to(device)

wav, sr = torchaudio.load("input.wav")
wav = wav.mean(dim=0, keepdim=True).to(device)
with torch.inference_mode():
    audio = model.from_waveform(wav)

torchaudio.save("output.wav", audio.squeeze(0).cpu(), model.sample_rate)


import torch
import torchaudio
from hf_model import ComVoHF

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ComVoHF.from_pretrained("hsoh/ComVo-base")
model = model.eval().to(device)

feature_extractor = model.build_feature_extractor().to(device)

wav, sr = torchaudio.load("input.wav")
if wav.size(0) > 1:
    wav = wav.mean(dim=0, keepdim=True)

wav = wav.to(device)

with torch.inference_mode():
    features = feature_extractor(wav)
    audio_output = model(features)

torchaudio.save("output.wav", audio_output.squeeze(0).cpu(), model.sample_rate)
