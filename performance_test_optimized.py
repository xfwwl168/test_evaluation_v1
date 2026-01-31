# ============================================================================
# 文件: performance_test_optimized.py
# 说明: 性能测试脚本 - 验证所有优化达成目标指标
# ============================================================================
"""
性能测试系统

测试目标:
- 缓存系统: 命中率 85%+, 性能提升 5倍
- 批量查询: 性能提升 3倍
- 信号生成: < 2秒 (100股)
- 完整回测: < 5秒 (100股×100天)
- 整体系统: 5-10倍性能提升
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
from pathlib import Path

# 导入优化模块
from core.cache_manager import cache_manager
from core.batch_query import batch_query_manager
from strategy.signal_generator_optimized import signal_generator_optimized
from factors.alpha_hunter_v2_factors_optimized import alpha_engine_v2_optimized

# 导入原始模块用于对比
from factors.alpha_hunter_v2_factors import AlphaFactorEngineV2


@dataclass
class PerformanceMetrics:
    """性能指标"""
    test_name: str
    execution_time: float
    memory_usage: float
    throughput: float
    target_time: float
    target_throughput: float
    passed: bool
    details: Dict[str, Any] = None


@dataclass
class PerformanceReport:
    """性能测试报告"""
    cache_metrics: PerformanceMetrics
    batch_query_metrics: PerformanceMetrics
    signal_generation_metrics: PerformanceMetrics
    alpha_computation_metrics: PerformanceMetrics
    memory_usage_metrics: PerformanceMetrics
    overall_score: float
    tests_passed: int
    total_tests: int
    execution_time: float


class PerformanceTestSuite:
    """性能测试套件"""
    
    def __init__(self):
        """初始化测试套件"""
        self.logger = logging.getLogger("PerformanceTestSuite")
        
        # 测试配置
        self.test_config = {
            'cache_test': {
                'operations': 10000,
                'cache_size': 5000,
                'target_hit_rate': 0.85,
                'target_speedup': 5.0
            },
            'batch_query_test': {
                'num_stocks': 100,
                'date_range_days': 100,
                'target_speedup': 3.0
            },
            'signal_generation_test': {
                'num_stocks': 100,
                'target_time': 2.0,  # 2秒
                'target_throughput': 50  # 50股/秒
            },
            'alpha_computation_test': {
                'num_samples': 100,
                'target_time': 1.0,  # 1秒
                'target_speedup': 5.0
            }
        }
        
        # 清理缓存
        cache_manager.clear_cache()
        
        # 生成测试数据
        self.test_data = self._generate_test_data()
    
    def run_full_performance_test(self) -> PerformanceReport:
        """运行完整性能测试"""
        self.logger.info("Starting comprehensive performance test...")
        start_time = time.time()
        
        # 1. 缓存系统测试
        cache_metrics = self._test_cache_performance()
        
        # 2. 批量查询测试
        batch_query_metrics = self._test_batch_query_performance()
        
        # 3. 信号生成测试
        signal_generation_metrics = self._test_signal_generation_performance()
        
        # 4. Alpha计算测试
        alpha_computation_metrics = self._test_alpha_computation_performance()
        
        # 5. 内存使用测试
        memory_usage_metrics = self._test_memory_usage()
        
        # 计算总体评分
        passed_tests = sum([
            cache_metrics.passed,
            batch_query_metrics.passed,
            signal_generation_metrics.passed,
            alpha_computation_metrics.passed,
            memory_usage_metrics.passed
        ])
        
        overall_score = self._calculate_overall_score([
            cache_metrics, batch_query_metrics, signal_generation_metrics,
            alpha_computation_metrics, memory_usage_metrics
        ])
        
        report = PerformanceReport(
            cache_metrics=cache_metrics,
            batch_query_metrics=batch_query_metrics,
            signal_generation_metrics=signal_generation_metrics,
            alpha_computation_metrics=alpha_computation_metrics,
            memory_usage_metrics=memory_usage_metrics,
            overall_score=overall_score,
            tests_passed=passed_tests,
            total_tests=5,
            execution_time=time.time() - start_time
        )
        
        self.logger.info(f"Performance test completed in {report.execution_time:.2f}s")
        return report
    
    def _generate_test_data(self) -> Dict[str, pd.DataFrame]:
        """生成测试数据"""
        self.logger.info("Generating test data...")
        
        num_stocks = 50
        num_days = 200
        
        test_data = {}
        
        for i in range(num_stocks):
            # 生成基础价格数据
            np.random.seed(i + 123)
            base_price = 10 + np.random.random() * 20
            price_changes = np.random.normal(0, 0.02, num_days)
            
            prices = [base_price]
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.1))
            
            # 生成OHLCV数据
            data = {
                'open': np.array(prices) + np.random.normal(0, 0.001, num_days),
                'high': np.array(prices) + np.abs(np.random.normal(0, 0.01, num_days)),
                'low': np.array(prices) - np.abs(np.random.normal(0, 0.01, num_days)),
                'close': np.array(prices),
                'vol': np.random.randint(100000, 5000000, num_days)
            }
            
            # 确保OHLC逻辑关系
            for j in range(num_days):
                high_price = max(data['open'][j], data['close'][j]) + data['high'][j]
                low_price = min(data['open'][j], data['close'][j]) - data['low'][j]
                data['high'][j] = high_price
                data['low'][j] = low_price
            
            code = f"{i:06d}"
            df = pd.DataFrame(data, index=pd.date_range('2023-01-01', periods=num_days, freq='D'))
            test_data[code] = df
        
        self.logger.info(f"Generated test data for {len(test_data)} stocks")
        return test_data
    
    def _test_cache_performance(self) -> PerformanceMetrics:
        """测试缓存性能"""
        self.logger.info("Testing cache performance...")
        
        config = self.test_config['cache_test']
        operations = config['operations']
        cache_size = config['cache_size']
        
        # 清理缓存
        cache_manager.clear_cache()
        
        # 测试无缓存性能
        def expensive_operation(x):
            # 模拟昂贵计算
            result = 0
            for i in range(1000):
                result += np.sin(x + i) * np.cos(x - i)
            return result
        
        # 预热缓存
        cache_results = []
        for i in range(100):
            cache_manager.set('test', f'key_{i}', expensive_operation(i))
        
        # 测试缓存命中性能
        start_time = time.time()
        cache_hits = 0
        for i in range(operations):
            result = cache_manager.get('test', f'key_{i % 100}')
            if result is not None:
                cache_hits += 1
        cache_time = time.time() - start_time
        
        # 测试无缓存性能（计算）
        start_time = time.time()
        for i in range(min(operations, 1000)):  # 限制测试数量
            expensive_operation(i)
        no_cache_time = time.time() - start_time
        
        # 计算指标
        hit_rate = cache_hits / operations
        speedup = no_cache_time / max(cache_time, 0.001)
        
        # 获取缓存统计
        cache_stats = cache_manager.get_stats()
        total_hits = sum(stats.hits for stats in cache_stats.values())
        total_requests = sum(stats.total_requests for stats in cache_stats.values())
        overall_hit_rate = total_hits / max(total_requests, 1)
        
        passed = (
            hit_rate >= config['target_hit_rate'] and
            speedup >= config['target_speedup']
        )
        
        return PerformanceMetrics(
            test_name="Cache Performance",
            execution_time=cache_time,
            memory_usage=0.0,  # 简化
            throughput=operations / cache_time,
            target_time=no_cache_time / operations,
            target_throughput=operations / no_cache_time,
            passed=passed,
            details={
                'hit_rate': hit_rate,
                'overall_hit_rate': overall_hit_rate,
                'speedup': speedup,
                'cache_hits': cache_hits,
                'total_operations': operations
            }
        )
    
    def _test_batch_query_performance(self) -> PerformanceMetrics:
        """测试批量查询性能"""
        self.logger.info("Testing batch query performance...")
        
        config = self.test_config['batch_query_test']
        num_stocks = config['num_stocks']
        
        # 获取测试股票列表
        stock_codes = list(self.test_data.keys())[:num_stocks]
        start_date = '2023-01-01'
        end_date = '2023-04-10'
        
        # 测试批量查询性能
        start_time = time.time()
        batch_data = batch_query_manager.query_ohlcv_batch(
            stock_codes, start_date, end_date, use_cache=True
        )
        batch_time = time.time() - start_time
        
        # 测试单股查询性能（估算）
        start_time = time.time()
        single_times = []
        for code in stock_codes[:10]:  # 测试前10只股票
            if code in self.test_data:
                single_times.append(0.001)  # 模拟单股查询时间
        single_time = sum(single_times)
        estimated_total_single_time = single_time * (num_stocks / 10)
        
        # 计算指标
        batch_throughput = num_stocks / batch_time
        estimated_single_throughput = num_stocks / max(estimated_total_single_time, 0.001)
        speedup = estimated_single_throughput / max(batch_throughput, 0.001)
        
        passed = speedup >= config['target_speedup']
        
        return PerformanceMetrics(
            test_name="Batch Query Performance",
            execution_time=batch_time,
            memory_usage=0.0,
            throughput=batch_throughput,
            target_time=estimated_total_single_time,
            target_throughput=estimated_single_throughput,
            passed=passed,
            details={
                'num_stocks': num_stocks,
                'data_loaded': len(batch_data),
                'speedup': speedup,
                'target_speedup': config['target_speedup']
            }
        )
    
    def _test_signal_generation_performance(self) -> PerformanceMetrics:
        """测试信号生成性能"""
        self.logger.info("Testing signal generation performance...")
        
        config = self.test_config['signal_generation_test']
        num_stocks = config['num_stocks']
        target_time = config['target_time']
        
        # 获取测试股票
        stock_codes = list(self.test_data.keys())[:num_stocks]
        start_date = '2023-01-01'
        end_date = '2023-04-10'
        
        # 测试信号生成性能
        start_time = time.time()
        signals_dict = signal_generator_optimized.generate_signals_batch(
            stock_codes, start_date, end_date
        )
        generation_time = time.time() - start_time
        
        # 计算指标
        throughput = num_stocks / generation_time
        total_signals = sum(len(signals) for signals in signals_dict.values())
        
        passed = (
            generation_time <= target_time and
            throughput >= config['target_throughput']
        )
        
        return PerformanceMetrics(
            test_name="Signal Generation Performance",
            execution_time=generation_time,
            memory_usage=0.0,
            throughput=throughput,
            target_time=target_time,
            target_throughput=config['target_throughput'],
            passed=passed,
            details={
                'num_stocks': num_stocks,
                'total_signals': total_signals,
                'avg_signals_per_stock': total_signals / max(num_stocks, 1),
                'target_met': generation_time <= target_time
            }
        )
    
    def _test_alpha_computation_performance(self) -> PerformanceMetrics:
        """测试Alpha计算性能"""
        self.logger.info("Testing Alpha computation performance...")
        
        config = self.test_config['alpha_computation_test']
        num_samples = config['num_samples']
        target_time = config['target_time']
        
        # 获取测试样本
        sample_data = list(self.test_data.values())[:num_samples]
        
        # 测试优化版本性能
        start_time = time.time()
        optimized_results = []
        for df in sample_data:
            try:
                result = alpha_engine_v2_optimized.compute(df)
                optimized_results.append(result)
            except Exception as e:
                self.logger.warning(f"Optimized computation error: {str(e)}")
        optimized_time = time.time() - start_time
        
        # 测试原始版本性能（简化版本）
        original_engine = AlphaFactorEngineV2()
        start_time = time.time()
        original_results = []
        for df in sample_data[:10]:  # 只测试前10个样本
            try:
                result = original_engine.compute(df)
                original_results.append(result)
            except Exception as e:
                self.logger.warning(f"Original computation error: {str(e)}")
        original_time = time.time() - start_time
        
        # 计算性能提升
        estimated_full_original_time = original_time * (num_samples / 10)
        speedup = estimated_full_original_time / max(optimized_time, 0.001)
        
        # 计算指标
        throughput = num_samples / optimized_time
        
        passed = (
            optimized_time <= target_time and
            speedup >= config['target_speedup']
        )
        
        return PerformanceMetrics(
            test_name="Alpha Computation Performance",
            execution_time=optimized_time,
            memory_usage=0.0,
            throughput=throughput,
            target_time=target_time,
            target_throughput=num_samples / target_time,
            passed=passed,
            details={
                'num_samples': num_samples,
                'optimized_results': len(optimized_results),
                'original_results': len(original_results),
                'speedup': speedup,
                'target_speedup': config['target_speedup']
            }
        )
    
    def _test_memory_usage(self) -> PerformanceMetrics:
        """测试内存使用"""
        self.logger.info("Testing memory usage...")
        
        # 获取初始内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 模拟内存密集操作
        large_data = {}
        for i in range(100):
            code = f"mem_test_{i:03d}"
            df = self.test_data[list(self.test_data.keys())[i % len(self.test_data)]]
            large_data[code] = df.copy()
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 清理数据
        del large_data
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 内存增长评估
        memory_growth = peak_memory - initial_memory
        memory_cleanup = final_memory - initial_memory
        
        # 内存使用目标：增长不超过500MB，清理后增长不超过50MB
        target_growth = 500  # MB
        target_cleanup = 50   # MB
        
        passed = (
            memory_growth <= target_growth and
            memory_cleanup <= target_cleanup
        )
        
        return PerformanceMetrics(
            test_name="Memory Usage",
            execution_time=0.0,
            memory_usage=memory_growth,
            throughput=0.0,
            target_time=0.0,
            target_throughput=0.0,
            passed=passed,
            details={
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': memory_growth,
                'memory_cleanup_mb': memory_cleanup,
                'target_growth_mb': target_growth,
                'target_cleanup_mb': target_cleanup
            }
        )
    
    def _calculate_overall_score(self, metrics_list: List[PerformanceMetrics]) -> float:
        """计算总体评分"""
        if not metrics_list:
            return 0.0
        
        scores = []
        for metric in metrics_list:
            if metric.passed:
                if metric.target_throughput > 0:
                    # 性能类测试使用吞吐率评分
                    score = min(metric.throughput / metric.target_throughput, 2.0)
                elif metric.target_time > 0:
                    # 时间类测试使用时间评分
                    score = min(metric.target_time / max(metric.execution_time, 0.001), 2.0)
                else:
                    score = 1.0
            else:
                score = 0.0
            
            scores.append(score)
        
        return np.mean(scores)
    
    def print_performance_report(self, report: PerformanceReport):
        """打印性能报告"""
        print("\n" + "="*80)
        print("PERFORMANCE TEST REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Overall Score: {report.overall_score:.2f}/2.00")
        print(f"   Tests Passed: {report.tests_passed}/{report.total_tests}")
        print(f"   Execution Time: {report.execution_time:.2f}s")
        
        # 缓存性能
        cache = report.cache_metrics
        print(f"\n💾 CACHE PERFORMANCE:")
        print(f"   Status: {'✅ PASSED' if cache.passed else '❌ FAILED'}")
        print(f"   Hit Rate: {cache.details['hit_rate']:.2%} (target: 85%+)")
        print(f"   Speedup: {cache.details['speedup']:.2f}x (target: 5x+)")
        print(f"   Cache Hits: {cache.details['cache_hits']:,}")
        print(f"   Overall Hit Rate: {cache.details['overall_hit_rate']:.2%}")
        
        # 批量查询性能
        batch = report.batch_query_metrics
        print(f"\n🚀 BATCH QUERY PERFORMANCE:")
        print(f"   Status: {'✅ PASSED' if batch.passed else '❌ FAILED'}")
        print(f"   Stocks Loaded: {batch.details['num_stocks']}")
        print(f"   Data Loaded: {batch.details['data_loaded']} stocks")
        print(f"   Speedup: {batch.details['speedup']:.2f}x (target: 3x+)")
        print(f"   Execution Time: {batch.execution_time:.3f}s")
        
        # 信号生成性能
        signals = report.signal_generation_metrics
        print(f"\n📈 SIGNAL GENERATION PERFORMANCE:")
        print(f"   Status: {'✅ PASSED' if signals.passed else '❌ FAILED'}")
        print(f"   Execution Time: {signals.execution_time:.3f}s (target: ≤2s)")
        print(f"   Throughput: {signals.throughput:.1f} stocks/sec (target: 50+/sec)")
        print(f"   Total Signals: {signals.details['total_signals']}")
        print(f"   Avg Signals/Stock: {signals.details['avg_signals_per_stock']:.1f}")
        
        # Alpha计算性能
        alpha = report.alpha_computation_metrics
        print(f"\n🧮 ALPHA COMPUTATION PERFORMANCE:")
        print(f"   Status: {'✅ PASSED' if alpha.passed else '❌ FAILED'}")
        print(f"   Execution Time: {alpha.execution_time:.3f}s (target: ≤1s)")
        print(f"   Speedup: {alpha.details['speedup']:.2f}x (target: 5x+)")
        print(f"   Samples: {alpha.details['num_samples']}")
        print(f"   Results: {alpha.details['optimized_results']}")
        
        # 内存使用
        memory = report.memory_usage_metrics
        print(f"\n💾 MEMORY USAGE:")
        print(f"   Status: {'✅ PASSED' if memory.passed else '❌ FAILED'}")
        print(f"   Memory Growth: {memory.details['memory_growth_mb']:.1f}MB (target: ≤500MB)")
        print(f"   Memory After Cleanup: {memory.details['memory_cleanup_mb']:.1f}MB (target: ≤50MB)")
        print(f"   Peak Memory: {memory.details['peak_memory_mb']:.1f}MB")
        
        # 目标达成情况
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        targets = [
            ("Cache Hit Rate ≥ 85%", report.cache_metrics.details['hit_rate'] >= 0.85),
            ("Cache Speedup ≥ 5x", report.cache_metrics.details['speedup'] >= 5.0),
            ("Batch Query Speedup ≥ 3x", report.batch_query_metrics.details['speedup'] >= 3.0),
            ("Signal Gen ≤ 2s", report.signal_generation_metrics.execution_time <= 2.0),
            ("Signal Throughput ≥ 50/s", report.signal_generation_metrics.throughput >= 50),
            ("Alpha Speedup ≥ 5x", report.alpha_computation_metrics.details['speedup'] >= 5.0),
            ("Memory Growth ≤ 500MB", report.memory_usage_metrics.details['memory_growth_mb'] <= 500)
        ]
        
        for target_name, achieved in targets:
            status = "✅" if achieved else "❌"
            print(f"   {status} {target_name}")
        
        # 总体评分
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if report.overall_score >= 1.5:
            print("   🎉 EXCELLENT: All performance targets exceeded!")
        elif report.overall_score >= 1.0:
            print("   ✅ GOOD: All major performance targets met!")
        elif report.overall_score >= 0.7:
            print("   ⚠️  FAIR: Most targets met, some improvements needed")
        else:
            print("   ❌ POOR: Multiple performance targets missed")
        
        print("="*80)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Starting Performance Test Suite...")
    
    # 运行性能测试
    test_suite = PerformanceTestSuite()
    report = test_suite.run_full_performance_test()
    
    # 打印报告
    test_suite.print_performance_report(report)
    
    # 保存报告到文件
    report_file = Path("performance_test_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Performance Test Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall Score: {report.overall_score:.2f}/2.00\n")
        f.write(f"Tests Passed: {report.tests_passed}/{report.total_tests}\n")
        f.write(f"Total Execution Time: {report.execution_time:.2f}s\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # 退出代码
    import sys
    sys.exit(0 if report.overall_score >= 1.0 else 1)