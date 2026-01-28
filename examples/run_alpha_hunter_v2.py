#!/usr/bin/env python
# ============================================================================
# 文件: examples/run_alpha_hunter_v2.py
# ============================================================================
"""
Alpha-Hunter-V2 策略回测示例
"""
import sys
from pathlib import Path

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine.backtest import BacktestEngine
from strategy.alpha_hunter_v2_strategy import AlphaHunterV2Strategy
from analysis import ReportGenerator
from utils.logger import setup_logging


def main():
    """运行 Alpha-Hunter-V2 回测"""

    setup_logging(level='INFO')

    print("=" * 70)
    print("🏆 Alpha-Hunter-V2 私募级超短线策略")
    print("=" * 70)
    print("""
策略特点:
  • 自适应 RSRS (市场状态感知)
  • 5重入场条件极致过滤
  • T+1 必杀卖出
  • 动态移动锁利 (每+3%→止损上移2%)
  • Kelly 准则动态仓位
  • 行业限额控制

目标:
  • 年化收益 > 30%
  • 最大回撤 < 10%
  • 持仓周期 T+1 到 T+2
""")
    print("=" * 70)

    # 创建引擎
    engine = BacktestEngine(
        initial_capital=1_000_000,
        commission_rate=0.0003,
        slippage_rate=0.001
    )

    # 创建策略
    strategy = AlphaHunterV2Strategy(params={
        'rsrs_threshold': 0.8,
        'rsrs_r2_threshold': 0.85,
        'min_signal_quality': 0.6,
        'hard_stop_loss': 0.03,
        't1_kill_threshold': 0.02,
        'max_holding_days': 2,
        'kelly_fraction': 0.5,
        'max_positions': 8,
    })

    engine.add_strategy(strategy)

    # 运行回测
    print("\n开始回测...")
    results = engine.run(
        start_date='2020-01-01',
        end_date='2023-12-31',
        rebalance_freq='D'  # 日度调仓
    )

    # 结果分析
    result = results['alpha_hunter_v2']

    # 绩效报告
    ReportGenerator.print_backtest_summary(result.metrics, "Alpha-Hunter-V2")

    # 策略统计
    perf = strategy.get_performance_summary()

    print("\n📊 交易统计:")
    print(f"   总交易: {perf.get('trades', 0)}")
    print(f"   胜率: {perf.get('win_rate', 0):.1%}")
    print(f"   平均盈利: {perf.get('avg_win', 0):.1%}")
    print(f"   平均亏损: {perf.get('avg_loss', 0):.1%}")
    print(f"   平均持仓: {perf.get('avg_holding_days', 0):.1f} 天")
    print(f"   最大单笔盈利: {perf.get('max_win', 0):.1%}")
    print(f"   最大单笔亏损: {perf.get('max_loss', 0):.1%}")

    # 导出
    equity = result.get_equity_curve()
    trades = result.get_trades()

    if not equity.empty:
        equity.to_csv('alpha_hunter_v2_equity.csv')
        print("\n✅ 权益曲线已保存到 alpha_hunter_v2_equity.csv")

    if not trades.empty:
        trades.to_csv('alpha_hunter_v2_trades.csv')
        print("✅ 交易记录已保存到 alpha_hunter_v2_trades.csv")

    return results


if __name__ == "__main__":
    main()