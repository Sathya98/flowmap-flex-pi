"""FlexPi's flow-map config on top of the model-agnostic objective in ``flowmap_core.flowmap``.

The objective math (``map_residuals``, time sampling, JVP helpers) and the shared
config fields live in the core and are re-exported here; FlexPi adds its streams
and its adaptation knobs (mode, initialization, LoRA rank/alpha).
"""
from dataclasses import dataclass
import math

from flowmap_core.flowmap import (  # noqa: F401  (re-exported for FlexPi callers)
    FlowMapObjectiveConfig, MapFn, affine_flow_map, delta_timestep, dX_dt_finite_difference,
    dX_dt_forward_ad, lmd_residual, map_residuals, sample_inference_grid_pairs, sample_level_pair_strip,
    training_time_pairs, tuple_jvp,
)

STREAMS = ("video", "dino", "pointmap", "action")


@dataclass
class FlowMapConfig(FlowMapObjectiveConfig):
    mode: str = "full"  # full | lora | adapter | heads
    streams: tuple = ("action",)
    initialization: str = "pretrained"  # random resets denoisers, not frozen encoders
    rank: int = 16
    lora_alpha: float = 16.0

    def __post_init__(self):
        super().__post_init__()
        self.streams = tuple(self.streams)
        if not self.streams or len(set(self.streams)) != len(self.streams) or not set(self.streams) <= set(STREAMS):
            raise ValueError("streams must contain unique action/video/dino/pointmap names")
        if "action" not in self.streams:
            raise ValueError("FlexPi experiments require the action stream")
        if self.mode not in ("adapter", "lora", "full", "heads"):
            raise ValueError(f"Unknown flow_map.mode: {self.mode}")
        if self.initialization not in ("pretrained", "random"):
            raise ValueError("initialization must be pretrained or random")
        if self.initialization == "random" and self.mode != "full":
            raise ValueError("Random initialization requires full training")
        if int(self.rank) != self.rank or self.rank < 1:
            raise ValueError(f"Invalid rank: {self.rank}")
        if not math.isfinite(float(self.lora_alpha)) or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be finite and positive")
