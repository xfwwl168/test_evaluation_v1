"""
性能测试套件
===========

完整的性能基准测试和压力测试
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==================== 基准测试 ====================
class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self):
        self.results = []
    
    def test_data_loading(self, start_date: str, end_date: str):
        """测试数据加载性能"""
        from engine.vectorized_backtest_engine import VectorizedBacktestEngine, BacktestConfig
        
        logger.info("\n" + "=" * 70)
        logger.info("测试1: 数据加载性能")
        logger.info("=" * 70)
        
        config = BacktestConfig()
        engine = VectorizedBacktestEngine(config=config)
        
        # 测试串行加载
        logger.info("\n[1.1] 串行加载...")
        t0 = time.time()
        engine.load_data(start_date, end_date, codes=None, use_parallel=False)
        serial_time = time.time() - t0
        n_stocks_serial = len(engine.data.codes)
        logger.info(f"  股票数: {n_stocks_serial}")
        logger.info(f"  耗时: {serial_time:.2f}s")
        
        # 测试并行加载
        logger.info("\n[1.2] 并行加载...")
        engine2 = VectorizedBacktestEngine(config=config)
        t0 = time.time()
        engine2.load_data(start_date, end_date, codes=None, use_parallel=True)
        parallel_time = time.time() - t0
        n_stocks_parallel = len(engine2.data.codes)
        logger.info(f"  股票数: {n_stocks_parallel}")
        logger.info(f"  耗时: {parallel_time:.2f}s")
        logger.info(f"  加速比: {serial_time/parallel_time:.2f}x")
        
        self.results.append({
            'test': 'Data Loading',
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': serial_time / parallel_time
        })
        
        return engine2  # 返回并行加载的引擎供后续使用
    
    def test_factor_computation(self, engine):
        """测试因子计算性能"""
        logger.info("\n" + "=" * 70)
        logger.info("测试2: 因子计算性能")
        logger.info("=" * 70)
        
        factors = ['momentum', 'rsrs', 'composite']
        
        for factor in factors:
            logger.info(f"\n[2.{factors.index(factor)+1}] 计算因子: {factor}")
            
            t0 = time.time()
            if factor == 'momentum':
                engine.compute_factors('momentum', period=20)
            elif factor == 'rsrs':
                engine.compute_factors('rsrs', window=18, n=600)
            elif factor == 'composite':
                engine.compute_factors('composite')
            
            elapsed = time.time() - t0
            logger.info(f"  耗时: {elapsed:.2f}s")
            
            self.results.append({
                'test': f'Factor: {factor}',
                'time': elapsed
            })
    
    def test_signal_generation(self, engine):
        """测试信号生成性能"""
        logger.info("\n" + "=" * 70)
        logger.info("测试3: 信号生成性能")
        logger.info("=" * 70)
        
        methods = ['topN', 'threshold', 'long_short']
        
        for method in methods:
            logger.info(f"\n[3.{methods.index(method)+1}] 方法: {method}")
            
            t0 = time.time()
            if method == 'topN':
                engine.generate_signals(method='topN', top_n=10)
            elif method == 'threshold':
                engine.generate_signals(method='threshold', threshold=0.7)
            elif method == 'long_short':
                engine.generate_signals(method='long_short', top_n=10)
            
            elapsed = time.time() - t0
            logger.info(f"  耗时: {elapsed:.2f}s")
            
            self.results.append({
                'test': f'Signal: {method}',
                'time': elapsed
            })
    
    def test_backtest_execution(self, engine):
        """测试回测执行性能"""
        logger.info("\n" + "=" * 70)
        logger.info("测试4: 回测执行性能")
        logger.info("=" * 70)
        
        t0 = time.time()
        results = engine.run_backtest()
        elapsed = time.time() - t0
        
        logger.info(f"\n  耗时: {elapsed:.2f}s")
        logger.info(f"  年化收益: {results['annual_return']:.2%}")
        logger.info(f"  夏普比率: {results['sharpe_ratio']:.2f}")
        logger.info(f"  最大回撤: {results['max_drawdown']:.2%}")
        
        self.results.append({
            'test': 'Backtest Execution',
            'time': elapsed,
            'sharpe': results['sharpe_ratio']
        })
    
    def run_full_benchmark(
        self,
        start_date: str = '2022-01-01',
        end_date: str = '2023-12-31'
    ):
        """运行完整基准测试"""
        logger.info("=" * 70)
        logger.info("性能基准测试套件")
        logger.info("=" * 70)
        logger.info(f"周期: {start_date} → {end_date}")
        logger.info("=" * 70)
        
        t_total = time.time()
        
        # 测试1: 数据加载
        engine = self.test_data_loading(start_date, end_date)
        
        # 测试2: 因子计算
        self.test_factor_computation(engine)
        
        # 测试3: 信号生成
        self.test_signal_generation(engine)
        
        # 测试4: 回测执行
        self.test_backtest_execution(engine)
        
        total_time = time.time() - t_total
        
        # 汇总
        logger.info("\n" + "=" * 70)
        logger.info("测试汇总")
        logger.info("=" * 70)
        logger.info(f"总耗时: {total_time:.2f}s")
        logger.info("=" * 70)
        
        return self.results


# ==================== 压力测试 ====================
class StressTest:
    """压力测试"""
    
    def test_different_stock_counts(self):
        """测试不同股票数量的性能"""
        from engine.vectorized_backtest_engine import VectorizedBacktestEngine, BacktestConfig
        
        logger.info("\n" + "=" * 70)
        logger.info("压力测试: 不同股票数量")
        logger.info("=" * 70)
        
        stock_counts = [100, 500, 1000, 2000]
        results = []
        
        for n_stocks in stock_counts:
            logger.info(f"\n测试 {n_stocks} 只股票...")
            
            config = BacktestConfig(top_n=10)
            engine = VectorizedBacktestEngine(config=config)
            
            try:
                t0 = time.time()
                
                # 加载数据（限制股票数）
                engine.load_data('2023-01-01', '2023-12-31', codes=None)
                
                # 只取前 N 只
                if len(engine.data.codes) > n_stocks:
                    codes_subset = engine.data.codes[:n_stocks]
                    engine.data.prices = engine.data.prices[codes_subset]
                    engine.data.returns = engine.data.returns[codes_subset]
                    engine.data.volumes = engine.data.volumes[codes_subset]
                    engine.data.codes = codes_subset
                
                # 计算因子
                engine.compute_factors('momentum', period=20)
                
                # 生成信号
                engine.generate_signals(method='topN', top_n=10)
                
                # 回测
                backtest_results = engine.run_backtest()
                
                elapsed = time.time() - t0
                
                results.append({
                    'stocks': n_stocks,
                    'time': elapsed,
                    'sharpe': backtest_results['sharpe_ratio']
                })
                
                logger.info(f"  ✓ 耗时: {elapsed:.2f}s")
                logger.info(f"  ✓ 夏普: {backtest_results['sharpe_ratio']:.2f}")
            
            except Exception as e:
                logger.error(f"  ✗ 失败: {e}")
                results.append({
                    'stocks': n_stocks,
                    'time': None,
                    'sharpe': None
                })
        
        # 绘制结果
        df = pd.DataFrame(results)
        logger.info("\n压力测试结果:")
        logger.info(df.to_string(index=False))
        
        return df
    
    def test_different_time_periods(self):
        """测试不同时间跨度的性能"""
        from engine.vectorized_backtest_engine import VectorizedBacktestEngine, BacktestConfig
        
        logger.info("\n" + "=" * 70)
        logger.info("压力测试: 不同时间跨度")
        logger.info("=" * 70)
        
        periods = [
            ('2023-01-01', '2023-03-31', '3个月'),
            ('2023-01-01', '2023-06-30', '6个月'),
            ('2023-01-01', '2023-12-31', '1年'),
            ('2022-01-01', '2023-12-31', '2年'),
        ]
        
        results = []
        
        for start, end, label in periods:
            logger.info(f"\n测试时间跨度: {label}")
            
            config = BacktestConfig(top_n=10)
            engine = VectorizedBacktestEngine(config=config)
            
            try:
                t0 = time.time()
                
                engine.load_data(start, end)
                engine.compute_factors('momentum', period=20)
                engine.generate_signals(method='topN', top_n=10)
                backtest_results = engine.run_backtest()
                
                elapsed = time.time() - t0
                
                results.append({
                    'period': label,
                    'days': len(engine.data.dates),
                    'time': elapsed,
                    'sharpe': backtest_results['sharpe_ratio']
                })
                
                logger.info(f"  ✓ 交易日: {len(engine.data.dates)}")
                logger.info(f"  ✓ 耗时: {elapsed:.2f}s")
            
            except Exception as e:
                logger.error(f"  ✗ 失败: {e}")
        
        df = pd.DataFrame(results)
        logger.info("\n时间跨度测试结果:")
        logger.info(df.to_string(index=False))
        
        return df


# ==================== 对比测试 ====================
class ComparisonTest:
    """引擎对比测试"""
    
    def compare_engines(
        self,
        start_date: str = '2022-01-01',
        end_date: str = '2023-12-31',
        strategy: str = 'momentum'
    ):
        """对比原引擎和向量化引擎"""
        logger.info("\n" + "=" * 70)
        logger.info("引擎对比测试")
        logger.info("=" * 70)
        logger.info(f"策略: {strategy}")
        logger.info(f"周期: {start_date} → {end_date}")
        logger.info("=" * 70)
        
        # 测试向量化引擎
        logger.info("\n[1/2] 向量化引擎...")
        from engine.vectorized_backtest_engine import VectorizedBacktestEngine, BacktestConfig
        
        config = BacktestConfig()
        engine_new = VectorizedBacktestEngine(config=config)
        
        t0 = time.time()
        engine_new.load_data(start_date, end_date)
        engine_new.compute_factors(strategy)
        engine_new.generate_signals(method='topN', top_n=10)
        results_new = engine_new.run_backtest()
        time_new = time.time() - t0
        
        logger.info(f"  ✓ 耗时: {time_new:.2f}s")
        logger.info(f"  ✓ 年化收益: {results_new['annual_return']:.2%}")
        logger.info(f"  ✓ 夏普比率: {results_new['sharpe_ratio']:.2f}")
        
        # 测试原引擎（如果可用）
        try:
            logger.info("\n[2/2] 原始引擎...")
            from engine.backtest import BacktestEngine
            from strategy.momentum_strategy import MomentumStrategy
            from strategy.rsrs_strategy import RSRSStrategy
            
            engine_old = BacktestEngine()
            
            if strategy == 'momentum':
                engine_old.add_strategy(MomentumStrategy())
            elif strategy == 'rsrs':
                engine_old.add_strategy(RSRSStrategy())
            else:
                raise ValueError(f"原引擎不支持: {strategy}")
            
            t0 = time.time()
            results_old = engine_old.run(start_date, end_date, rebalance_freq='W')
            time_old = time.time() - t0
            
            logger.info(f"  ✓ 耗时: {time_old:.2f}s")
            
            # 对比
            logger.info("\n" + "=" * 70)
            logger.info("性能对比")
            logger.info("=" * 70)
            logger.info(f"{'引擎':15} {'耗时':>12} {'加速比':>12}")
            logger.info("-" * 70)
            logger.info(f"{'原引擎':15} {time_old:>10.1f}s {1.0:>11.1f}x")
            logger.info(f"{'向量化引擎':15} {time_new:>10.1f}s {time_old/time_new:>11.1f}x")
            logger.info("=" * 70)
            
            return {
                'old_time': time_old,
                'new_time': time_new,
                'speedup': time_old / time_new,
                'old_sharpe': None,  # 原引擎结果格式不同
                'new_sharpe': results_new['sharpe_ratio']
            }
        
        except Exception as e:
            logger.warning(f"\n⚠️  原引擎测试失败: {e}")
            logger.info("（仅测试向量化引擎）")
            
            return {
                'old_time': None,
                'new_time': time_new,
                'speedup': None,
                'new_sharpe': results_new['sharpe_ratio']
            }


# ==================== 可视化 ====================
class PerformanceVisualizer:
    """性能可视化"""
    
    @staticmethod
    def plot_benchmark_results(results: List[Dict], save_path: str = None):
        """绘制基准测试结果"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # 提取不同类型的测试结果
        data_loading = [r for r in results if 'Data Loading' in str(r.get('test', ''))]
        factors = [r for r in results if 'Factor' in str(r.get('test', ''))]
        signals = [r for r in results if 'Signal' in str(r.get('test', ''))]
        backtest = [r for r in results if 'Backtest' in str(r.get('test', ''))]
        
        # 图1: 数据加载对比
        if data_loading:
            ax = axes[0, 0]
            test = data_loading[0]
            times = [test.get('serial_time', 0), test.get('parallel_time', 0)]
            ax.bar(['Serial', 'Parallel'], times, color=['#ff6b6b', '#4ecdc4'])
            ax.set_ylabel('Time (s)')
            ax.set_title('Data Loading: Serial vs Parallel')
            ax.text(1, test.get('parallel_time', 0) + 1, 
                   f"{test.get('speedup', 0):.1f}x faster", 
                   ha='center', fontweight='bold')
        
        # 图2: 因子计算时间
        if factors:
            ax = axes[0, 1]
            names = [r['test'].replace('Factor: ', '') for r in factors]
            times = [r['time'] for r in factors]
            ax.barh(names, times, color='#95e1d3')
            ax.set_xlabel('Time (s)')
            ax.set_title('Factor Computation Time')
        
        # 图3: 信号生成时间
        if signals:
            ax = axes[1, 0]
            names = [r['test'].replace('Signal: ', '') for r in signals]
            times = [r['time'] for r in signals]
            ax.barh(names, times, color='#f38181')
            ax.set_xlabel('Time (s)')
            ax.set_title('Signal Generation Time')
        
        # 图4: 回测执行
        if backtest:
            ax = axes[1, 1]
            test = backtest[0]
            ax.text(0.5, 0.6, f"Time: {test.get('time', 0):.2f}s", 
                   ha='center', fontsize=14)
            ax.text(0.5, 0.4, f"Sharpe: {test.get('sharpe', 0):.2f}", 
                   ha='center', fontsize=14)
            ax.set_title('Backtest Execution')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 图表已保存: {save_path}")
        else:
            plt.show()


# ==================== 主函数 ====================
def main():
    """运行完整测试套件"""
    print("\n" + "=" * 70)
    print("向量化回测引擎 - 完整性能测试套件")
    print("=" * 70)
    print("\n选择测试模式:")
    print("  1. 快速基准测试 (推荐)")
    print("  2. 完整基准测试")
    print("  3. 压力测试")
    print("  4. 引擎对比测试")
    print("  5. 全部测试")
    print()
    
    choice = input("请选择 [1-5]: ").strip()
    
    if choice == '1':
        # 快速基准测试
        benchmark = PerformanceBenchmark()
        results = benchmark.run_full_benchmark('2023-01-01', '2023-12-31')
        
        # 保存结果
        output_dir = Path('data/outputs')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        visualizer = PerformanceVisualizer()
        visualizer.plot_benchmark_results(results, 
                                         save_path=str(output_dir / 'benchmark_results.png'))
    
    elif choice == '2':
        # 完整基准测试
        benchmark = PerformanceBenchmark()
        results = benchmark.run_full_benchmark('2020-01-01', '2023-12-31')
    
    elif choice == '3':
        # 压力测试
        stress = StressTest()
        stress.test_different_stock_counts()
        stress.test_different_time_periods()
    
    elif choice == '4':
        # 引擎对比
        comparison = ComparisonTest()
        comparison.compare_engines('2022-01-01', '2023-12-31', 'momentum')
    
    elif choice == '5':
        # 全部测试
        logger.info("\n执行全部测试...")
        
        # 1. 基准测试
        benchmark = PerformanceBenchmark()
        results = benchmark.run_full_benchmark('2022-01-01', '2023-12-31')
        
        # 2. 压力测试
        stress = StressTest()
        stress.test_different_stock_counts()
        stress.test_different_time_periods()
        
        # 3. 对比测试
        comparison = ComparisonTest()
        comparison.compare_engines('2022-01-01', '2023-12-31', 'momentum')
        
        logger.info("\n" + "=" * 70)
        logger.info("全部测试完成！")
        logger.info("=" * 70)
    
    else:
        logger.error("无效选择")


if __name__ == "__main__":
    main()
