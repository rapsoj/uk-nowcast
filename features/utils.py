# features/utils.py
import math
import re
from typing import Tuple

import pandas as pd

HEX_SIZE = 0.9
SHRINK_FACTOR = 0.98

# colours
PARTY_COLOR_MAP = { "LAB": "#DC241f", "CON": "#0087DC", "SNP": "#d4cd02", "LDM": "#FDBB30", "GRN": "#6AB023", "RFM": "#3BB7D5", "PLC": "#6E86FF", "MIN": "#FFB6C1", "YOUR":"#FF69B4", "Oth": "#999999"}

def slug_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def hex_to_pixel(q: float, r: float, size: float = HEX_SIZE) -> Tuple[float, float]:
    w_factor = math.sqrt(3) * size
    x = w_factor * (q + 0.5 * (r & 1))
    y = 1.5 * size * r
    return x, y


def polygon_hex(center_x: float, center_y: float, size: float = HEX_SIZE * SHRINK_FACTOR):
    verts = []
    for i in range(6):
        angle_rad = math.radians(60 * i - 30)
        x = center_x + size * math.cos(angle_rad)
        y = center_y + size * math.sin(angle_rad)
        verts.append((x, y))
    verts.append(verts[0])
    return verts
