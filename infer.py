import os
import torch
import torchaudio
import argparse
import yaml
import importlib
from glob import glob
from tqdm import tqdm


def load_specific_module(model, state_dict, module_name):

    pretrained_dict = {
        k.replace(f"{module_name}.", ""): v
        for k, v in state_dict.items()
        if k.startswith(module_name)
    }

    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="config")
    parser.add_argument("--file_list", type=str, default=None)
    parser.add_argument("--dir", type=str, default="wavs")

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    print(f"Load config from {args.config}")

    feature_extractor_cls = config["model"]["init_args"]["feature_extractor"][
        "class_path"
    ]
    feature_extractor_pkg = ".".join(feature_extractor_cls.split(".")[:-1])
    feature_extractor_name = feature_extractor_cls.split(".")[-1]
    feature_extractor_cls_name = getattr(
        importlib.import_module(feature_extractor_pkg), feature_extractor_name
    )
    feature_extractor = feature_extractor_cls_name(
        **config["model"]["init_args"]["feature_extractor"]["init_args"]
    )

    backbone_cls = config["model"]["init_args"]["backbone"]["class_path"]
    backbone_pkg = ".".join(backbone_cls.split(".")[:-1])
    backbone_name = backbone_cls.split(".")[-1]
    backbone_cls_name = getattr(importlib.import_module(backbone_pkg), backbone_name)
    backbone = backbone_cls_name(
        **config["model"]["init_args"]["backbone"]["init_args"]
    )

    head_cls = config["model"]["init_args"]["head"]["class_path"]
    head_pkg = ".".join(head_cls.split(".")[:-1])
    head_name = head_cls.split(".")[-1]
    head_cls_name = getattr(importlib.import_module(head_pkg), head_name)
    head = head_cls_name(**config["model"]["init_args"]["head"]["init_args"])

    print(f"Load feature_extractor from {feature_extractor_cls_name}")
    print(f"Load backbone from {backbone_cls_name}")
    print(f"Load head from {head_cls_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    save_dir = config["trainer"]["logger"]["init_args"]["save_dir"]
    ckpt_path_pattern = f"{save_dir}/lightning_logs/version_*/checkpoints/last.ckpt"
    ckpt_list = glob(ckpt_path_pattern)
    ckpt = ckpt_list[-1]
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)

    load_specific_module(
        feature_extractor, state_dict["state_dict"], "feature_extractor"
    )
    load_specific_module(backbone, state_dict["state_dict"], "backbone")
    load_specific_module(head, state_dict["state_dict"], "head")
    feature_extractor = feature_extractor.to(device)
    backbone = backbone.to(device)
    head = head.to(device)
    print(f"Load checkpoint from {ckpt}")

    sr = config["model"]["init_args"]["sample_rate"]
    print(f"Sampling rate: {sr}")

    file_list = args.file_list
    lines = open(file_list, "r").readlines()
    os.makedirs(os.path.join(save_dir, args.dir), exist_ok=True)

    for i in tqdm(lines):
        audio_path = i.strip()
        basename = os.path.basename(audio_path)
        y, sr = torchaudio.load(audio_path)

        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)

        y = y.to(device)
        features = feature_extractor(y)
        x = backbone(features)
        audio_output = head(x)
        audio_output = audio_output.detach().cpu()
        save_name = os.path.join(save_dir, args.dir, basename)
        torchaudio.save(save_name, audio_output, sr)
