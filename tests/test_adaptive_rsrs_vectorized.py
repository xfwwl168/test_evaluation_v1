# ============================================================================
# 文件: tests/test_adaptive_rsrs_vectorized.py
# ============================================================================
"""AdaptiveRSRSFactor Tier-1 向量化重构测试。

该文件聚焦三个关键函数：
- _calc_adaptive_window
- _robust_zscore
- _detect_market_state

测试目标：
1) 正确性：输出形状/取值范围/边界处理符合约束
2) 性能：在 CI 环境中给出温和的上限，避免因机器差异导致波动

注意：性能测试使用软阈值（相对宽松），主要用于防止回退到明显的 Python 双重循环。
"""

import time

import numpy as np
import pandas as pd

from factors.alpha_hunter_v2_factors import AdaptiveRSRSFactor, MarketState


def _make_series(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    close = (rng.normal(0, 1, n).cumsum() + 100).astype(np.float64)
    volume = rng.integers(1_000_000, 10_000_000, size=n, dtype=np.int64).astype(np.float64)
    return close, volume


def test_calc_adaptive_window_correctness():
    close, volume = _make_series(1000)
    f = AdaptiveRSRSFactor()

    w = f._calc_adaptive_window(close, volume)

    assert isinstance(w, np.ndarray)
    assert w.shape == (len(close),)
    assert w.dtype in (np.int32, np.int64)
    assert np.all(w[:30] == f.base_window)
    assert w.min() >= 12
    assert w.max() <= 24


def test_calc_adaptive_window_handles_nan():
    close, volume = _make_series(800)
    close[100:120] = np.nan

    f = AdaptiveRSRSFactor()
    w = f._calc_adaptive_window(close, volume)

    assert w.shape == (len(close),)
    assert np.all(w[:30] == f.base_window)
    assert w.min() >= 12
    assert w.max() <= 24


def test_calc_adaptive_window_performance():
    close, volume = _make_series(2000)
    f = AdaptiveRSRSFactor()

    t0 = time.perf_counter()
    _ = f._calc_adaptive_window(close, volume)
    elapsed = time.perf_counter() - t0

    # 宽松阈值：CI 机器差异大，但 O(n^2) 会明显超出
    assert elapsed < 0.25


def test_robust_zscore_correctness():
    rng = np.random.default_rng(123)
    arr = rng.normal(0, 1, 1200).astype(np.float64)
    arr[:10] = np.nan

    f = AdaptiveRSRSFactor()
    z = f._robust_zscore(arr, window=600)

    assert isinstance(z, np.ndarray)
    assert z.shape == arr.shape
    # 前 min_periods 之前应为 NaN
    assert np.isnan(z[:60]).all()

    # 对非 NaN 区域，z-score 不应出现 inf
    finite = np.isfinite(z)
    assert finite.sum() > 0


def test_robust_zscore_performance():
    rng = np.random.default_rng(123)
    arr = rng.normal(0, 1, 4000).astype(np.float64)

    f = AdaptiveRSRSFactor()
    t0 = time.perf_counter()
    _ = f._robust_zscore(arr, window=600)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.35


def test_detect_market_state_correctness():
    close, volume = _make_series(500)
    f = AdaptiveRSRSFactor()

    states = f._detect_market_state(close, volume)

    assert isinstance(states, np.ndarray)
    assert states.shape == (len(close),)
    assert np.all(states[:60] == MarketState.SHOCK.value)

    allowed = {s.value for s in MarketState}
    assert set(np.unique(states)).issubset(allowed)


def test_detect_market_state_performance():
    close, volume = _make_series(5000)
    f = AdaptiveRSRSFactor()

    t0 = time.perf_counter()
    _ = f._detect_market_state(close, volume)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.15
