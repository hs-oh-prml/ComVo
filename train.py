from pytorch_lightning.cli import LightningCLI
from exp.complex_ddp_strategy import ComplexDDPStrategy
import os
import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = str


class CustomCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--ckpt_path", type=str, default=None, help="Path to the checkpoint file."
        )

    def instantiate_trainer(self, **kwargs):
        # 사용자가 YAML/CLI로 strategy를 주지 않았으면 우리가 박아넣기
        if not kwargs.get("strategy"):
            kwargs["strategy"] = ComplexDDPStrategy(
                comm_hook_state=None, find_unused_parameters=True
            )
        return super().instantiate_trainer(**kwargs)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    cli = CustomCLI(run=False, save_config_kwargs={"overwrite": True})
    os.makedirs(cli.trainer.logger.save_dir, exist_ok=True)

    ckpt_path = cli.config.get("ckpt_path", None)
    if ckpt_path:
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    else:
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
