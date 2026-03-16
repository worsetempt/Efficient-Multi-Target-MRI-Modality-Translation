from __future__ import annotations

import math


def psnr_from_mse(mse: float, data_range: float = 2.0) -> float:
    mse = max(float(mse), 1e-12)
    return float(10.0 * math.log10((data_range ** 2) / mse))
