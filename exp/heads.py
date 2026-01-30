import torch
from exp.spectral_ops import ISTFT
from exp.cvnn import *


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft // 2 + 1
        self.out = cLinear(dim, out_dim)
        self.n_fft = n_fft
        self.win_length = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.win_length).cuda()
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        S = torch.exp(x)
        mag = torch.abs(S)
        S = S * (torch.clamp(mag, max=1e2) / (mag + 1e-9))
        audio = self.istft(S)
        return audio
