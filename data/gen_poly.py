#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic dataset generators for the LeCo paper.

Datasets covered (as described in Section 4.1, footnotes 4 and 5):
  - cosmos  : 100M 32-bit, cosmic ray signal (sine mixture + Gaussian noise)
  - polylog : 10M  64-bit, alternating polynomial and logarithm segments every 500 records
  - exp     : 200M 64-bit, block-wise exponential curves with varying parameters
  - poly    : 200M 64-bit, block-wise cubic-polynomial curves with varying parameters

Output format:
  - If output path ends with ".bin": raw binary values (no header). Read with np.memmap / np.fromfile.
  - If output path ends with ".npy": NumPy .npy file created via memory map.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ArrayWriter:
    out_path: Path
    dtype: np.dtype
    n: int

    _mm: Optional[np.memmap] = None
    _fh: Optional[object] = None  # file handle

    def __post_init__(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.suffix.lower() == ".npy":
            self._mm = np.lib.format.open_memmap(
                self.out_path, mode="w+", dtype=self.dtype, shape=(self.n,)
            )
        else:
            self._fh = open(self.out_path, "wb")

    def write(self, start: int, data: np.ndarray) -> None:
        if data.dtype != self.dtype:
            data = data.astype(self.dtype, copy=False)

        if self._mm is not None:
            end = start + data.shape[0]
            self._mm[start:end] = data
        else:
            assert self._fh is not None
            data.tofile(self._fh)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._mm is not None:
            self._mm.flush()
            self._mm = None


def _progress(i: int, n: int, every: int = 1) -> None:
    if every <= 0:
        return
    if (i // every) * every != i:
        return
    pct = 100.0 * i / max(1, n)
    print(f"  progress: {i}/{n} ({pct:.1f}%)", flush=True)


def generate_cosmos(
    out_path: Path,
    n: int = 100_000_000,
    seed: int = 0,
    chunk: int = 5_000_000,
) -> None:
    """
    cosmos (paper footnote 4):
      ( sin((x+10)/(60*pi)) + (1/10) * sin( 3*(x+10)/(60*pi) ) ) * 1e6 + N(0, 100)
    Stored as int32 after rounding.
    """
    dtype = np.int32
    rng = np.random.default_rng(seed)
    writer = ArrayWriter(out_path, dtype=np.dtype(dtype), n=n)

    denom = 60.0 * math.pi
    scale = 1_000_000.0
    noise_std = 100.0

    try:
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            x = np.arange(start, end, dtype=np.float64)
            y = (np.sin((x + 10.0) / denom) + 0.1 * np.sin(3.0 * (x + 10.0) / denom)) * scale
            y += rng.normal(loc=0.0, scale=noise_std, size=(end - start))
            y_i32 = np.rint(y).astype(np.int32)
            writer.write(start, y_i32)
            _progress(end, n, every=chunk)
    finally:
        writer.close()


def generate_polylog(
    out_path: Path,
    n: int = 10_000_000,
    seed: int = 0,
    seg_len: int = 500,
    poly_deg: int = 2,
    poly_delta_range: Tuple[int, int] = (50_000, 250_000),
    log_delta_range: Tuple[int, int] = (10_000, 120_000),
    gap_range: Tuple[int, int] = (0, 50),
) -> None:
    """
    polylog (paper footnote 5):
      "Constructed by concatenating the polynomial and logarithm distribution,
       in turn, every 500 records."

    This implementation alternates segments of length `seg_len`:
      - even segments: polynomial growth curve (degree = poly_deg)
      - odd  segments: logarithmic growth curve (log1p)
    We keep the sequence non-decreasing by building it cumulatively.

    Output dtype: int64.
    """
    dtype = np.int64
    rng = np.random.default_rng(seed)
    writer = ArrayWriter(out_path, dtype=np.dtype(dtype), n=n)

    base = np.int64(0)
    seg_count = (n + seg_len - 1) // seg_len

    try:
        for s in range(seg_count):
            start = s * seg_len
            end = min(n, start + seg_len)
            L = end - start

            if L <= 0:
                break

            if L == 1:
                y = np.array([base], dtype=np.int64)
                writer.write(start, y)
                base = base + np.int64(rng.integers(gap_range[0], gap_range[1] + 1))
                continue

            t = np.arange(L, dtype=np.float64)
            u = t / float(L - 1)

            if (s % 2) == 0:
                delta = int(rng.integers(poly_delta_range[0], poly_delta_range[1] + 1))
                seg = delta * (u ** poly_deg)
            else:
                delta = int(rng.integers(log_delta_range[0], log_delta_range[1] + 1))
                seg = delta * (np.log1p(t) / np.log1p(L - 1))

            y = base + np.rint(seg).astype(np.int64)
            y = np.maximum.accumulate(y)

            writer.write(start, y)

            gap = int(rng.integers(gap_range[0], gap_range[1] + 1))
            base = np.int64(y[-1] + gap)

            if (s % 2000) == 0:
                _progress(start, n, every=max(1, seg_len * 2000))
    finally:
        writer.close()


def generate_exp(
    out_path: Path,
    n: int = 200_000_000,
    seed: int = 0,
    block_len: int = 10_000,
    delta_log10_range: Tuple[float, float] = (8.0, 13.0),
    k_range: Tuple[float, float] = (1.0, 12.0),
    gap_range: Tuple[int, int] = (0, 1_000),
) -> None:
    """
    exp: 200M 64-bit; each block follows an exponential distribution with different parameters.

    Paper doesn't specify exact params; we implement a normalized exponential per block:
        y = base + delta_end * (exp(k*u)-1) / (exp(k)-1),  u in [0,1]
    """
    dtype = np.int64
    rng = np.random.default_rng(seed)
    writer = ArrayWriter(out_path, dtype=np.dtype(dtype), n=n)

    base = np.int64(0)
    blocks = (n + block_len - 1) // block_len
    max_i64 = np.iinfo(np.int64).max

    try:
        for b in range(blocks):
            start = b * block_len
            end = min(n, start + block_len)
            L = end - start
            if L <= 0:
                break

            delta_end = int(round(10 ** rng.uniform(delta_log10_range[0], delta_log10_range[1])))
            k = float(rng.uniform(k_range[0], k_range[1]))
            gap = int(rng.integers(gap_range[0], gap_range[1] + 1))

            if base > max_i64 - delta_end - gap:
                delta_end = int(max(0, max_i64 - int(base) - gap))

            if L == 1 or delta_end == 0:
                y = np.array([base], dtype=np.int64)
            else:
                u = np.linspace(0.0, 1.0, L, dtype=np.float64)
                shape = (np.exp(k * u) - 1.0) / (math.exp(k) - 1.0)
                y = base + np.rint(shape * delta_end).astype(np.int64)
                y = np.maximum.accumulate(y)

            writer.write(start, y)

            base = np.int64(y[-1] + gap)
            if (b % 200) == 0:
                _progress(b, blocks, every=200)
    finally:
        writer.close()


def generate_poly(
    out_path: Path,
    n: int = 200_000_000,
    seed: int = 0,
    block_len: int = 10_000,
    start_val: int = 1_000_000_000,
    end_delta: int = 50_000_000,
    slope_scale: float = 0.25,
    clamp_range: Tuple[int, int] = (0, 2_000_000_000),
) -> None:
    """
    poly: 200M 64-bit; each block follows polynomial distribution with different parameters.

    Paper doesn't specify exact params; we implement block-wise cubic polynomials
    via cubic Hermite interpolation between random endpoints with random slopes.
    """
    dtype = np.int64
    rng = np.random.default_rng(seed)
    writer = ArrayWriter(out_path, dtype=np.dtype(dtype), n=n)

    y0 = int(start_val)
    blocks = (n + block_len - 1) // block_len
    lo, hi = clamp_range

    try:
        for b in range(blocks):
            start = b * block_len
            end = min(n, start + block_len)
            L = end - start
            if L <= 0:
                break

            if L == 1:
                y = np.array([y0], dtype=np.int64)
                writer.write(start, y)
                continue

            y1 = int(y0 + rng.integers(-end_delta, end_delta + 1))
            y1 = int(np.clip(y1, lo, hi))

            slope_mag = (end_delta / max(1, (L - 1))) * slope_scale
            m0 = float(rng.uniform(-slope_mag, slope_mag))
            m1 = float(rng.uniform(-slope_mag, slope_mag))

            u = np.linspace(0.0, 1.0, L, dtype=np.float64)

            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2

            y = (h00 * y0 + h10 * (m0 * (L - 1)) + h01 * y1 + h11 * (m1 * (L - 1)))
            y = np.rint(y).astype(np.int64)
            y = np.clip(y, lo, hi)

            writer.write(start, y)
            y0 = y1

            if (b % 200) == 0:
                _progress(b, blocks, every=200)
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic datasets described in the LeCo paper.")
    p.add_argument("dataset", choices=["cosmos", "polylog", "exp", "poly"], help="Dataset name.")
    p.add_argument("--out", required=True, type=Path, help="Output path (.bin or .npy).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    p.add_argument("--n", type=int, default=None, help="Override total length (records).")
    p.add_argument("--chunk", type=int, default=5_000_000, help="Chunk size for cosmos.")
    p.add_argument("--seg-len", type=int, default=500, help="Segment length for polylog (paper uses 500).")
    p.add_argument("--block-len", type=int, default=10_000, help="Block length for exp/poly.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = args.dataset
    out = args.out

    if ds == "cosmos":
        n = args.n if args.n is not None else 100_000_000
        generate_cosmos(out, n=n, seed=args.seed, chunk=args.chunk)

    elif ds == "polylog":
        n = args.n if args.n is not None else 10_000_000
        generate_polylog(out, n=n, seed=args.seed, seg_len=args.seg_len)

    elif ds == "exp":
        n = args.n if args.n is not None else 200_000_000
        generate_exp(out, n=n, seed=args.seed, block_len=args.block_len)

    elif ds == "poly":
        n = args.n if args.n is not None else 200_000_000
        generate_poly(out, n=n, seed=args.seed, block_len=args.block_len)

    else:
        raise ValueError(f"Unknown dataset: {ds}")

    print(f"Done. Wrote: {out}")


if __name__ == "__main__":
    main()
