import json
import importlib
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


def build_from_class_path(class_path: str, init_args: Dict[str, Any]):
    pkg = ".".join(class_path.split(".")[:-1])
    name = class_path.split(".")[-1]
    cls = getattr(importlib.import_module(pkg), name)
    return cls(**init_args)


class ComVoHF(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        sample_rate: int,
        backbone_class_path: str,
        backbone_init_args: Dict[str, Any],
        head_class_path: str,
        head_init_args: Dict[str, Any],
        feature_extractor_class_path: str,
        feature_extractor_init_args: Dict[str, Any],
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.backbone_class_path = backbone_class_path
        self.backbone_init_args = backbone_init_args

        self.head_class_path = head_class_path
        self.head_init_args = head_init_args

        self.feature_extractor_class_path = feature_extractor_class_path
        self.feature_extractor_init_args = feature_extractor_init_args

        self.backbone = build_from_class_path(
            self.backbone_class_path,
            self.backbone_init_args,
        )
        self.head = build_from_class_path(
            self.head_class_path,
            self.head_init_args,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.backbone(features)
        audio_output = self.head(x)
        return audio_output

    def build_feature_extractor(self):
        return build_from_class_path(
            self.feature_extractor_class_path,
            self.feature_extractor_init_args,
        )

    @torch.inference_mode()
    def from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        feature_extractor = self.build_feature_extractor().to(waveform.device)
        features = feature_extractor(waveform)
        return self.forward(features)

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config = {
            "sample_rate": self.sample_rate,
            "backbone_class_path": self.backbone_class_path,
            "backbone_init_args": self.backbone_init_args,
            "head_class_path": self.head_class_path,
            "head_init_args": self.head_init_args,
            "feature_extractor_class_path": self.feature_extractor_class_path,
            "feature_extractor_init_args": self.feature_extractor_init_args,
        }

        with open(save_directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # complex tensor 포함 가능하도록 torch.save 사용
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str = None,
        cache_dir: str = None,
        force_download: bool = False,
        proxies: Dict[str, str] = None,
        resume_download: bool = None,
        local_files_only: bool = False,
        token: str = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs,
    ):
        if Path(model_id).is_dir():
            config_path = Path(model_id) / "config.json"
            weights_path = Path(model_id) / "pytorch_model.bin"
        else:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.update(model_kwargs)
        model = cls(**config)

        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        return model
