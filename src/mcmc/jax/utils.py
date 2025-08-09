from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EMA:
    ema_weight: float = 0.999
    step: int = 0
    val_: float | None = None

    def __call__(self, val: float) -> float:
        if self.val_ is None:
            self.val_ = val
            self.ema_correction = 1.0
        else:
            self.val_ = self.val_ * self.ema_weight + val * (1 - self.ema_weight)
            self.ema_correction = 1 - self.ema_weight ** (self.step + 1)
            self.step += 1
        return self.val_ / self.ema_correction

    @property
    def val(self):  # for parity with torch.utils.EMA
        return self.val_


