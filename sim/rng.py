"""Shared deterministic draws keyed by seed, day, and creative."""

from __future__ import annotations

import hashlib
import math


def unit_draw(seed: int, day: int, creative_id: str, channel: str) -> float:
    """Return U ~ Uniform(0, 1) shared across policies for the same key."""
    payload = f"{int(seed)}|{int(day)}|{creative_id}|{channel}".encode()
    digest = hashlib.sha256(payload).digest()
    # Map 53 bits into [0, 1).
    value = int.from_bytes(digest[:7], "big") >> 3
    return value / float(1 << 53)


def noise_multiplier(u: float, cv: float) -> float:
    """Map a unit draw to a positive multiplier with approx coefficient of variation cv.

    Uses a log-normal centered at 1: log(X) ~ Normal( -0.5 s^2, s^2 ) with
    s chosen so Var(X)/E[X]^2 ≈ cv^2 for small cv.
    """
    if cv <= 0.0:
        return 1.0
    # Inverse CDF of standard normal via Beasley-Springer/Moro approximation light:
    # use erfinv relation: Phi^{-1}(u) = sqrt(2) erfinv(2u-1).
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    z = math.sqrt(2.0) * _erfinv(2.0 * u - 1.0)
    sigma = cv
    return math.exp(-0.5 * sigma * sigma + sigma * z)


def _erfinv(x: float) -> float:
    """Approximate inverse error function for x in (-1, 1)."""
    # Winitzki approximation.
    a = 0.147
    ln = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(first * first - ln / a) - first), x)
