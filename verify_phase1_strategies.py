#!/usr/bin/env python3
"""
Phase 1 策略验证脚本

验证内容:
1. 4个新策略正确注册
2. 策略创建和初始化
3. 性能基准测试
4. StrategyFactory功能验证
"""
import time
import numpy as np
import pandas as pd
from typing import Dict

def generate_test_data(n_stocks: int = 50, n_days: int = 100) -> Dict[str, pd.DataFrame]:
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
            'open': prices * (1 + np.random.randn(n_days) * 0.005),
            'high': prices * (1 + abs(np.random.randn(n_days)) * 0.015),
            'low': prices * (1 - abs(np.random.randn(n_days)) * 0.015),
            'close': prices,
            'vol': np.random.randint(1000000, 10000000, n_days),
            'amount': np.random.randint(10000000, 100000000, n_days)
        })
        data[code] = df

    return data


def test_strategy_registry():
    """测试策略注册"""
    print("\n" + "=" * 70)
    print("测试: 策略注册")
    print("=" * 70)

    from strategy import StrategyRegistry

    strategies = StrategyRegistry.list_all()
    print(f"\n已注册策略数量: {len(strategies)}")

    required_strategies = [
        'rsrs', 'momentum', 'short_term_rsrs', 'alpha_hunter_v2',
        'bull_bear', 'ultra_short', 'dinger', 'hanbing'
    ]

    missing = [s for s in required_strategies if s not in strategies]
    if missing:
        print(f"✗ 缺失策略: {missing}")
        return False

    print("✓ 所有必需策略已注册")
    for name in strategies:
        info = StrategyRegistry.get_info(name)
        print(f"  - {name}: {info['class']} v{info['version']}")

    return True


def test_strategy_factory():
    """测试StrategyFactory"""
    print("\n" + "=" * 70)
    print("测试: StrategyFactory")
    print("=" * 70)

    from strategy import get_factory, StrategyRegistry

    factory = get_factory()

    print(f"\n可用策略: {len(factory.list_available())}")
    print(f"已配置策略: {len(factory.list_configured())}")

    # 测试创建策略
    print("\n创建策略测试:")
    test_names = ['momentum', 'bull_bear', 'dinger', 'hanbing']
    for name in test_names:
        try:
            strategy = factory.create(name)
            print(f"  ✓ 创建 {name} 成功: {strategy.name}")
        except Exception as e:
            print(f"  ✗ 创建 {name} 失败: {e}")
            return False

    # 测试组合策略
    print("\n组合策略测试:")
    try:
        combo = factory.create_combo(['momentum', 'bull_bear'], weights=[0.6, 0.4])
        print(f"  ✓ 创建组合策略成功: {len(combo.strategies)} 个子策略")
    except Exception as e:
        print(f"  ✗ 创建组合策略失败: {e}")
        return False

    print("\n✓ StrategyFactory 测试通过")
    return True


def test_strategy_initialization():
    """测试策略初始化"""
    print("\n" + "=" * 70)
    print("测试: 策略初始化")
    print("=" * 70)

    from strategy import get_factory

    factory = get_factory()

    # 测试新策略的初始化
    new_strategies = ['bull_bear', 'ultra_short', 'dinger', 'hanbing']

    for name in new_strategies:
        print(f"\n{name}:")
        try:
            strategy = factory.create(name)
            strategy.initialize()
            print(f"  ✓ 初始化成功")
            print(f"  - 参数数量: {len(strategy.params)}")
        except Exception as e:
            print(f"  ✗ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ 策略初始化测试通过")
    return True


def test_strategy_compute_factors():
    """测试因子计算"""
    print("\n" + "=" * 70)
    print("测试: 因子计算")
    print("=" * 70)

    from strategy import get_factory

    factory = get_factory()

    # 生成测试数据
    data = generate_test_data(n_stocks=30, n_days=100)
    print(f"\n测试数据: {len(data)} 只股票 × {len(list(data.values())[0])} 天")

    # 测试新策略的因子计算
    new_strategies = ['bull_bear', 'dinger', 'hanbing']

    for name in new_strategies:
        print(f"\n{name}:")
        try:
            strategy = factory.create(name)
            t0 = time.perf_counter()
            factors = strategy.compute_factors(data)
            elapsed = time.perf_counter() - t0

            print(f"  ✓ 计算成功: {len(factors)} 个因子")
            print(f"  - 耗时: {elapsed:.3f} 秒")
            print(f"  - 因子: {list(factors.keys())}")
        except Exception as e:
            print(f"  ✗ 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ 因子计算测试通过")
    return True


def test_strategy_performance():
    """测试策略性能"""
    print("\n" + "=" * 70)
    print("测试: 策略性能基准")
    print("=" * 70)

    from strategy import get_factory
    from strategy.base import StrategyContext

    factory = get_factory()

    # 生成测试数据
    data = generate_test_data(n_stocks=100, n_days=100)
    print(f"\n测试数据: {len(data)} 只股票 × {len(list(data.values())[0])} 天")

    # 创建模拟上下文
    dates = list(data.values())[0]['date'].tolist()
    current_date = dates[-1].strftime('%Y-%m-%d')

    current_data = pd.DataFrame({
        'code': list(data.keys()),
        'date': current_date,
        'open': [df['close'].iloc[-1] * 0.99 for df in data.values()],
        'high': [df['close'].iloc[-1] * 1.01 for df in data.values()],
        'low': [df['close'].iloc[-1] * 0.98 for df in data.values()],
        'close': [df['close'].iloc[-1] for df in data.values()],
        'vol': [df['vol'].iloc[-1] for df in data.values()],
        'amount': [df['amount'].iloc[-1] for df in data.values()],
    })

    # 测试新策略的信号生成性能
    new_strategies = ['momentum', 'bull_bear', 'dinger', 'hanbing']

    performance_results = {}

    for name in new_strategies:
        print(f"\n{name}:")
        try:
            strategy = factory.create(name)
            strategy.initialize()

            # 计算因子
            t0 = time.perf_counter()
            factors = strategy.compute_factors(data)
            factor_time = time.perf_counter() - t0

            # 创建上下文
            context = StrategyContext(
                current_date=current_date,
                current_data=current_data,
                history_data=data,
                factors=factors,
                positions={},
                cash=100000,
                total_equity=100000
            )

            # 生成信号
            t0 = time.perf_counter()
            signals = strategy.generate_signals(context)
            signal_time = time.perf_counter() - t0

            total_time = factor_time + signal_time
            performance_results[name] = total_time

            print(f"  ✓ 信号生成成功: {len(signals)} 个信号")
            print(f"  - 因子计算: {factor_time:.3f} 秒")
            print(f"  - 信号生成: {signal_time:.3f} 秒")
            print(f"  - 总耗时: {total_time:.3f} 秒")

            # 验证性能目标
            if total_time < 2.0:
                print(f"  ✓ 达到性能目标 (< 2秒)")
            else:
                print(f"  ⚠ 未达到性能目标 (>= 2秒)")

        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # 总结
    print("\n" + "=" * 70)
    print("性能基准总结")
    print("=" * 70)
    for name, elapsed in performance_results.items():
        status = "✓" if elapsed < 2.0 else "⚠"
        print(f"{status} {name:20s}: {elapsed:.3f} 秒")

    avg_time = sum(performance_results.values()) / len(performance_results)
    print(f"\n平均耗时: {avg_time:.3f} 秒")

    if avg_time < 2.0:
        print("✓ 所有策略达到性能目标")
        return True
    else:
        print("⚠ 部分策略未达到性能目标")
        return False


def test_combo_strategy():
    """测试组合策略"""
    print("\n" + "=" * 70)
    print("测试: 组合策略")
    print("=" * 70)

    from strategy import get_factory
    from strategy.base import StrategyContext

    factory = get_factory()

    # 生成测试数据
    data = generate_test_data(n_stocks=50, n_days=100)
    dates = list(data.values())[0]['date'].tolist()
    current_date = dates[-1].strftime('%Y-%m-%d')

    current_data = pd.DataFrame({
        'code': list(data.keys()),
        'date': current_date,
        'open': [df['close'].iloc[-1] * 0.99 for df in data.values()],
        'high': [df['close'].iloc[-1] * 1.01 for df in data.values()],
        'low': [df['close'].iloc[-1] * 0.98 for df in data.values()],
        'close': [df['close'].iloc[-1] for df in data.values()],
        'vol': [df['vol'].iloc[-1] for df in data.values()],
        'amount': [df['amount'].iloc[-1] for df in data.values()],
    })

    try:
        # 创建组合策略
        combo = factory.create_combo(['momentum', 'bull_bear'], weights=[0.6, 0.4])
        combo.initialize()

        # 计算因子
        factors = combo.compute_factors(data)
        print(f"✓ 计算因子成功: {len(factors)} 个因子")

        # 创建上下文
        context = StrategyContext(
            current_date=current_date,
            current_data=current_data,
            history_data=data,
            factors=factors,
            positions={},
            cash=100000,
            total_equity=100000
        )

        # 生成信号
        signals = combo.generate_signals(context)
        print(f"✓ 生成信号成功: {len(signals)} 个信号")

        # 按股票汇总信号
        from collections import defaultdict
        code_signals = defaultdict(list)
        for sig in signals:
            code_signals[sig.code].append(sig)

        print(f"\n信号分布:")
        for code, sigs in code_signals.items():
            total_weight = sum(s.weight for s in sigs if s.side.value == 'BUY')
            print(f"  {code}: 总仓位={total_weight:.2%}, 信号数={len(sigs)}")

        print("\n✓ 组合策略测试通过")
        return True

    except Exception as e:
        print(f"✗ 组合策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Phase 1 策略验证 - 开始")
    print("=" * 70)

    results = {}

    # 1. 策略注册测试
    results['registry'] = test_strategy_registry()

    # 2. StrategyFactory测试
    results['factory'] = test_strategy_factory()

    # 3. 策略初始化测试
    results['initialization'] = test_strategy_initialization()

    # 4. 因子计算测试
    results['factors'] = test_strategy_compute_factors()

    # 5. 性能基准测试
    results['performance'] = test_strategy_performance()

    # 6. 组合策略测试
    results['combo'] = test_combo_strategy()

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有Phase 1策略验证通过!")
        return True
    else:
        print("\n⚠ 部分测试未通过，请检查失败项目")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
