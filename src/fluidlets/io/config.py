"""Parser for IB2d-style parameter files (``input2d``, or ``input3d`` in 3D).

Accepts the IB2d layout (``%`` comments, ``Section { ... }`` blocks) and the
flat layout in DESIGN.md (``#`` comments). Every ``key = value`` line becomes
one entry; section names are ignored. Values are converted to int, float,
a NumPy array (several numbers), or a string (quotes stripped).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _value(text: str):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    parts = text.split()
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return text
    if len(nums) > 1:
        return np.array(nums)
    num = nums[0]
    return int(num) if re.fullmatch(r"[+-]?\d+", parts[0]) else num


def read_input(path) -> dict:
    """Read an input2d/input3d file into a dict of parameters."""
    params = {}
    for raw in Path(path).read_text().splitlines():
        line = re.split(r"[%#]", raw, maxsplit=1)[0].strip()
        if not line or line.endswith("{") or line == "}" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[key.strip()] = _value(value)
    return params
