#!/usr/bin/env python3
"""Phase 1 (Tier 1) 向量化重构验证脚本。

用途：
- 快速进行正确性/性能烟囱测试（本地/开发机）
- 不作为严格基准，阈值偏宽松，主要用于识别 O(n^2) 回退

运行：
    python verify_vectorization_phase1.py
"""

from __future__ import annotations

import time

import numpy as np

from factors.alpha_hunter_v2_factors import AdaptiveRSRSFactor, MarketState


def _gen_data(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    close = (rng.normal(0, 1, n).cumsum() + 100).astype(np.float64)
    volume = rng.integers(1_000_000, 10_000_000, size=n, dtype=np.int64).astype(np.float64)
    return close, volume


def benchmark_adaptive_window() -> None:
    close, volume = _gen_data(2000)
    factor = AdaptiveRSRSFactor()

    t0 = time.perf_counter()
    window = factor._calc_adaptive_window(close, volume)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"⏱️  _calc_adaptive_window: {elapsed * 1000:.2f} ms")

    assert window.shape == (len(close),)
    assert window.min() >= 12 and window.max() <= 24
    assert (window[:30] == factor.base_window).all()


def benchmark_robust_zscore() -> None:
    rng = np.random.default_rng(1)
    arr = rng.normal(0, 1, 4000).astype(np.float64)
    factor = AdaptiveRSRSFactor()

    t0 = time.perf_counter()
    z = factor._robust_zscore(arr, window=600)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"⏱️  _robust_zscore: {elapsed * 1000:.2f} ms")

    assert z.shape == arr.shape
    assert np.isfinite(z[~np.isnan(z)]).all()


def benchmark_detect_market_state() -> None:
    close, volume = _gen_data(5000)
    factor = AdaptiveRSRSFactor()

    t0 = time.perf_counter()
    states = factor._detect_market_state(close, volume)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"⏱️  _detect_market_state: {elapsed * 1000:.2f} ms")

    assert states.shape == (len(close),)
    assert (states[:60] == MarketState.SHOCK.value).all()


def main() -> None:
    print("=" * 70)
    print("LION_QUANT 2026 - Vectorization Phase 1 Verification")
    print("=" * 70)

    benchmark_adaptive_window()
    benchmark_robust_zscore()
    benchmark_detect_market_state()

    print("\n✓ Phase 1 Tier-1 vectorization checks passed")


if __name__ == "__main__":
    main()
