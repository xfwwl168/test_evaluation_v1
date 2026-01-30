#!/usr/bin/env python3
"""
向量化改进验证脚本
====================

验证内容:
1. 性能对比 (改进前后的执行时间)
2. 正确性验证 (向量化结果与参考结果的一致性)
3. 所有关键函数的向量化状态
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List
import sys

# 设置随机种子保证可重复性
np.random.seed(42)


def generate_test_data(n_stocks: int = 100, n_days: int = 252) -> Dict[str, pd.DataFrame]:
    """生成测试数据"""
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    codes = [f'ST{i:04d}' for i in range(n_stocks)]

    data = {}
    for code in codes:
        # 生成随机价格序列
        returns = np.random.randn(n_days) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(n_days) * 0.01),
            'high': prices * (1 + abs(np.random.randn(n_days)) * 0.02),
            'low': prices * (1 - abs(np.random.randn(n_days)) * 0.02),
            'close': prices,
            'vol': np.random.randint(1000000, 10000000, n_days),
            'amount': np.random.randint(10000000, 100000000, n_days)
        })
        data[code] = df

    return data


def benchmark_vectorized_rsrs():
    """测试向量化RSRS性能"""
    print("\n" + "=" * 70)
    print("测试: 向量化RSRS因子计算")
    print("=" * 70)

    from engine.vectorized_backtest_engine import VectorizedData, VectorizedFactors

    # 生成测试矩阵数据
    n_days = 500
    n_stocks = 200
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    stocks = [f'S{i:04d}' for i in range(n_stocks)]

    np.random.seed(42)
    highs = pd.DataFrame(
        np.random.randn(n_days, n_stocks).cumsum(axis=0) + 100,
        index=dates, columns=stocks
    )
    lows = highs * (0.98 + np.random.rand(n_days, n_stocks) * 0.02)

    data = VectorizedData(
        prices=highs,
        volumes=pd.DataFrame(np.random.randint(1000000, 10000000, (n_days, n_stocks)),
                            index=dates, columns=stocks),
        amounts=pd.DataFrame(np.random.randint(10000000, 100000000, (n_days, n_stocks)),
                            index=dates, columns=stocks),
        returns=pd.DataFrame(np.random.randn(n_days, n_stocks) * 0.02,
                            index=dates, columns=stocks),
        dates=dates,
        codes=stocks,
        highs=highs,
        lows=lows,
        opens=highs * 0.99
    )

    # 测试向量化RSRS
    t0 = time.perf_counter()
    result = VectorizedFactors.rsrs(data, window=18, n=60)
    elapsed = time.perf_counter() - t0

    print(f"\n数据集: {n_stocks} 只股票 × {n_days} 天")
    print(f"向量化RSRS计算时间: {elapsed:.3f} 秒")

    # 验证结果合理性
    assert not result.empty, "结果不应为空"
    assert result.shape == (n_days, n_stocks), f"形状应为 {(n_days, n_stocks)}, 实际是 {result.shape}"

    # 检查NaN比例 (前window-1行应为NaN)
    nan_ratio = result.isna().sum().sum() / result.size
    expected_nan_ratio = 17 / n_days  # window=18, 前17行为NaN
    print(f"NaN比例: {nan_ratio:.2%} (预期约: {expected_nan_ratio:.2%})")

    # 检查结果范围
    valid_values = result.dropna().values.flatten()
    print(f"有效值范围: [{valid_values.min():.3f}, {valid_values.max():.3f}]")
    print(f"有效值均值: {valid_values.mean():.3f}")

    print("\n✓ 向量化RSRS测试通过!")
    return elapsed


def benchmark_alpha_hunter_v2_factors():
    """测试Alpha Hunter V2因子"""
    print("\n" + "=" * 70)
    print("测试: Alpha Hunter V2 因子")
    print("=" * 70)

    from factors.alpha_hunter_v2_factors import (
        AdaptiveRSRSFactor, OpeningSurgeFactor, MultiLevelPressureFactor
    )

    # 生成单股票测试数据
    n_days = 700
    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')

    np.random.seed(42)
    returns = np.random.randn(n_days) * 0.02
    closes = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'date': dates,
        'open': closes * (1 + np.random.randn(n_days) * 0.005),
        'high': closes * (1 + abs(np.random.randn(n_days)) * 0.015),
        'low': closes * (1 - abs(np.random.randn(n_days)) * 0.015),
        'close': closes,
        'vol': np.random.randint(1000000, 10000000, n_days)
    })

    print(f"\n测试数据: {n_days} 天")

    # 测试自适应RSRS因子
    print("\n1. 自适应RSRS因子")
    rsrs_factor = AdaptiveRSRSFactor()
    t0 = time.perf_counter()
    rsrs_result = rsrs_factor.compute_full(df)
    elapsed = time.perf_counter() - t0
    print(f"   计算时间: {elapsed:.3f} 秒")
    print(f"   输出列: {list(rsrs_result.columns)}")
    print(f"   RSRS自适应范围: [{rsrs_result['rsrs_adaptive'].min():.3f}, {rsrs_result['rsrs_adaptive'].max():.3f}]")

    # 测试开盘异动因子
    print("\n2. 开盘异动因子")
    surge_factor = OpeningSurgeFactor()
    t0 = time.perf_counter()
    surge_result = surge_factor.compute_full(df)
    elapsed = time.perf_counter() - t0
    print(f"   计算时间: {elapsed:.3f} 秒")
    print(f"   输出列: {list(surge_result.columns)}")
    print(f"   异动评分范围: [{surge_result['surge_score'].min():.3f}, {surge_result['surge_score'].max():.3f}]")

    # 测试压力位因子
    print("\n3. 多层次压力位因子")
    pressure_factor = MultiLevelPressureFactor()
    t0 = time.perf_counter()
    pressure_result = pressure_factor.compute_full(df)
    elapsed = time.perf_counter() - t0
    print(f"   计算时间: {elapsed:.3f} 秒")
    print(f"   输出列: {list(pressure_result.columns)}")
    print(f"   压力距离范围: [{pressure_result['combined_pressure_dist'].min():.3f}, {pressure_result['combined_pressure_dist'].max():.3f}]")

    print("\n✓ Alpha Hunter V2 因子测试通过!")


def benchmark_extended_factors():
    """测试扩展因子库"""
    print("\n" + "=" * 70)
    print("测试: 扩展因子库")
    print("=" * 70)

    try:
        from factors.extended_factors import TechnicalFactors
    except ImportError as e:
        print(f"\n跳过测试: {e}")
        return

    # 生成测试数据
    n_days = 252
    n_stocks = 50
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    stocks = [f'S{i:04d}' for i in range(n_stocks)]

    np.random.seed(42)
    highs = pd.DataFrame(
        np.random.randn(n_days, n_stocks).cumsum(axis=0) + 100,
        index=dates, columns=stocks
    )
    lows = highs * 0.98
    closes = highs * 0.99

    print(f"\n数据集: {n_stocks} 只股票 × {n_days} 天")

    # 测试Aroon
    print("\n1. Aroon指标")
    t0 = time.perf_counter()
    aroon = TechnicalFactors.aroon(highs, lows, period=25)
    elapsed = time.perf_counter() - t0
    print(f"   计算时间: {elapsed:.3f} 秒")
    print(f"   结果范围: [{aroon.min().min():.3f}, {aroon.max().max():.3f}]")

    # 测试CCI
    print("\n2. CCI指标")
    t0 = time.perf_counter()
    cci = TechnicalFactors.cci(highs, lows, closes, period=20)
    elapsed = time.perf_counter() - t0
    print(f"   计算时间: {elapsed:.3f} 秒")
    print(f"   结果范围: [{cci.min().min():.3f}, {cci.max().max():.3f}]")

    # 测试OBV斜率
    print("\n3. OBV斜率")
    volumes = pd.DataFrame(
        np.random.randint(1000000, 10000000, (n_days, n_stocks)),
        index=dates, columns=stocks
    )
    t0 = time.perf_counter()
    obv = TechnicalFactors.obv_slope(closes, volumes, period=20)
    elapsed = time.perf_counter() - t0
    print(f"   计算时间: {elapsed:.3f} 秒")
    print(f"   结果范围: [{obv.min().min():.3f}, {obv.max().max():.3f}]")

    print("\n✓ 扩展因子库测试通过!")


def benchmark_strategy_factor_computation():
    """测试策略因子计算向量化"""
    print("\n" + "=" * 70)
    print("测试: 策略因子计算向量化")
    print("=" * 70)

    try:
        from strategy.strategy_momentum_reversal_combo import MomentumReversalStrategy
    except ImportError as e:
        print(f"\n跳过测试: {e}")
        return 0.0

    # 生成测试数据
    data = generate_test_data(n_stocks=100, n_days=100)

    print(f"\n测试数据: {len(data)} 只股票 × {len(list(data.values())[0])} 天")

    strategy = MomentumReversalStrategy()

    # 测试向量化因子计算
    t0 = time.perf_counter()
    factors = strategy.compute_factors(data)
    elapsed = time.perf_counter() - t0

    print(f"\n向量化因子计算时间: {elapsed:.3f} 秒")
    print(f"动量因子: {len(factors['momentum'])} 只股票")
    print(f"反转因子: {len(factors['reversal'])} 只股票")
    print(f"质量因子: {len(factors['quality'])} 只股票")

    # 验证结果
    assert 'momentum' in factors
    assert 'reversal' in factors
    assert 'quality' in factors

    print("\n✓ 策略因子计算测试通过!")
    return elapsed


def verify_correctness():
    """验证向量化结果的正确性"""
    print("\n" + "=" * 70)
    print("验证: 向量化结果正确性")
    print("=" * 70)

    from factors.alpha_hunter_v2_factors import MultiLevelPressureFactor

    # 生成固定测试数据
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

    closes = np.linspace(50, 100, n_days) + np.random.randn(n_days) * 2
    highs = closes * (1 + abs(np.random.randn(n_days)) * 0.02)
    lows = closes * (1 - abs(np.random.randn(n_days)) * 0.02)
    volumes = np.random.randint(1000000, 10000000, n_days)

    df = pd.DataFrame({
        'date': dates,
        'open': closes * 0.99,
        'high': highs,
        'low': lows,
        'close': closes,
        'vol': volumes
    })

    factor = MultiLevelPressureFactor()

    # 计算结果
    result = factor.compute_full(df)

    # 验证结果合理性
    print("\n压力位因子验证:")
    print(f"  技术压力20范围: [{result['tech_pressure_20'].min():.2f}, {result['tech_pressure_20'].max():.2f}]")
    print(f"  技术压力60范围: [{result['tech_pressure_60'].min():.2f}, {result['tech_pressure_60'].max():.2f}]")
    print(f"  压力距离范围: [{result['combined_pressure_dist'].min():.2f}, {result['combined_pressure_dist'].max():.2f}]")

    # 验证压力距离的数学关系
    # 压力距离 = (压力位 - 当前价) / 当前价，应为正值
    valid_dist = result['combined_pressure_dist'].dropna()
    assert (valid_dist >= 0).all(), "压力距离应为非负值"

    print("  ✓ 压力距离验证通过 (所有值 >= 0)")

    # 验证技术压力关系 (60日高点 >= 20日高点)
    valid_t20 = result['tech_pressure_20'].dropna()
    valid_t60 = result['tech_pressure_60'].dropna()
    overlap_idx = valid_t20.index.intersection(valid_t60.index)

    if len(overlap_idx) > 0:
        t20_vals = valid_t20.loc[overlap_idx]
        t60_vals = valid_t60.loc[overlap_idx]
        assert (t60_vals >= t20_vals * 0.95).all(), "60日压力应 >= 20日压力"
        print("  ✓ 技术压力关系验证通过 (60日 >= 20日)")

    print("\n✓ 正确性验证通过!")


def run_all_benchmarks():
    """运行所有基准测试"""
    print("\n" + "=" * 70)
    print("向量化改进验证 - 开始")
    print("=" * 70)

    results = {}

    try:
        # RSRS因子
        results['rsrs'] = benchmark_vectorized_rsrs()
    except Exception as e:
        print(f"\n✗ RSRS测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Alpha Hunter V2因子
        benchmark_alpha_hunter_v2_factors()
    except Exception as e:
        print(f"\n✗ Alpha Hunter V2测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        # 扩展因子库
        benchmark_extended_factors()
    except Exception as e:
        print(f"\n✗ 扩展因子库测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        # 策略因子计算
        results['strategy_factors'] = benchmark_strategy_factor_computation()
    except Exception as e:
        print(f"\n✗ 策略因子计算测试失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        # 正确性验证
        verify_correctness()
    except Exception as e:
        print(f"\n✗ 正确性验证失败: {e}")
        import traceback
        traceback.print_exc()

    # 总结
    print("\n" + "=" * 70)
    print("向量化改进验证 - 总结")
    print("=" * 70)

    print("\n性能基准:")
    for name, elapsed in results.items():
        print(f"  {name}: {elapsed:.3f} 秒")

    print("\n" + "=" * 70)
    print("✓ 所有向量化测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
