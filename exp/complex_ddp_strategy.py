import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as default_hooks
from pytorch_lightning.strategies import DDPStrategy
import torch

from torch.nn.parallel import DistributedDataParallel as DDP


def complex_safe_allreduce_hook(state, bucket: dist.GradBucket):
    buf = bucket.buffer()
    if not torch.is_complex(buf):
        return default_hooks.allreduce_hook(state, bucket)

    real_view = torch.view_as_real(buf.contiguous())  # [..., 2] float
    flat = real_view.view(-1)
    work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
    fut = work.get_future()

    def _finish(_):
        world = dist.get_world_size() if dist.is_initialized() else 1
        flat.div_(world)
        return buf

    return fut.then(_finish)


class ComplexDDPStrategy(DDPStrategy):
    def __init__(self, comm_hook_state=None, **kwargs):
        super().__init__(**kwargs)
        self._comm_hook_state = comm_hook_state

    def configure_ddp(self) -> None:
        super().configure_ddp()

        ddp_model: DDP | None = None

        if isinstance(self.model, DDP):
            ddp_model = self.model

        elif hasattr(self, "_model") and isinstance(getattr(self, "_model"), DDP):
            ddp_model = getattr(self, "_model")

        elif hasattr(self.model, "register_comm_hook"):
            ddp_model = self.model  # type: ignore[assignment]

        if ddp_model is None:
            raise RuntimeError(
                "ComplexDDPStrategy: DDP wrapper not found; cannot register comm hook."
            )
        ddp_model.register_comm_hook(
            state=self._comm_hook_state,
            hook=complex_safe_allreduce_hook,
        )
