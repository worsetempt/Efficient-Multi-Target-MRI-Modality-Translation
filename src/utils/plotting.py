from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image


def save_stacked_rows(rows: Iterable[np.ndarray], out_path: str | Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return str(out_path)
    canvas = np.concatenate(rows, axis=0)
    Image.fromarray(canvas, mode="L").save(out_path)
    return str(out_path)
