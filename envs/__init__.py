import math

import numpy as np
import transforms3d.euler as _t3d_euler

from .utils import *
from ._GLOBAL_CONFIGS import *


def _mat2euler_numpy2(mat, axes="sxyz"):
    try:
        firstaxis, parity, repetition, frame = _t3d_euler._AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _t3d_euler._TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _t3d_euler._NEXT_AXIS[i + parity]
    k = _t3d_euler._NEXT_AXIS[i - parity + 1]

    M = np.asarray(mat, dtype=np.float64)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _t3d_euler._EPS4:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _t3d_euler._EPS4:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


_t3d_euler.mat2euler = _mat2euler_numpy2
