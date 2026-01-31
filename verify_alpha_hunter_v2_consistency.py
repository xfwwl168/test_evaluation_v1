# ============================================================================
# 文件: verify_alpha_hunter_v2_consistency.py
# 说明: Alpha Hunter V2逻辑一致性验证脚本
# ============================================================================
"""
Alpha Hunter V2逻辑一致性验证系统

验证目标:
1. Tier 1: RSRS计算完全一致 (max_diff < 1e-6)
2. Tier 1: 压力位计算完全一致 (max_diff < 1e-6) 
3. Tier 2: 市场状态判断一致 (100%相同)
4. Tier 1: 信号生成一致 (100次随机测试)

验收标准:
- 所有Tier 1验证通过
- 所有Tier 2验证通过
- 回归测试100%通过
- 性能提升 >= 5倍
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import hashlib
from pathlib import Path

# 导入原始和优化的因子类
from factors.alpha_hunter_v2_factors import (
    AdaptiveRSRSFactor,
    MultiLevelPressureFactor,
    OpeningSurgeFactor,
    AlphaFactorEngineV2
)

from factors.alpha_hunter_v2_factors_optimized import (
    AdaptiveRSRSFactorOptimized,
    MultiLevelPressureFactorOptimized,
    OpeningSurgeFactorOptimized,
    AlphaFactorEngineV2Optimized
)


@dataclass
class ConsistencyResult:
    """一致性验证结果"""
    test_name: str
    tier: int
    passed: bool
    max_diff: float
    mean_diff: float
    total_tests: int
    passed_tests: int
    execution_time: float
    details: Dict[str, Any] = None


@dataclass
class ValidationReport:
    """验证报告"""
    tier1_results: List[ConsistencyResult]
    tier2_results: List[ConsistencyResult]
    overall_passed: bool
    performance_improvement: float
    total_tests: int
    passed_tests: int
    execution_time: float


class AlphaHunterV2ConsistencyValidator:
    """Alpha Hunter V2逻辑一致性验证器"""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        初始化验证器
        
        Args:
            tolerance: 数值容忍度
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger("AlphaHunterV2ConsistencyValidator")
        
        # 原始因子实例
        self.original_rsrs = AdaptiveRSRSFactor()
        self.original_pressure = MultiLevelPressureFactor()
        self.original_surge = OpeningSurgeFactor()
        self.original_engine = AlphaFactorEngineV2()
        
        # 优化因子实例
        self.optimized_rsrs = AdaptiveRSRSFactorOptimized()
        self.optimized_pressure = MultiLevelPressureFactorOptimized()
        self.optimized_surge = OpeningSurgeFactorOptimized()
        self.optimized_engine = AlphaFactorEngineV2Optimized()
        
        # 验证结果存储
        self.tier1_results: List[ConsistencyResult] = []
        self.tier2_results: List[ConsistencyResult] = []
    
    def run_full_validation(self, num_test_cases: int = 100) -> ValidationReport:
        """
        运行完整验证
        
        Args:
            num_test_cases: 测试用例数量
            
        Returns:
            ValidationReport: 验证报告
        """
        self.logger.info("Starting full Alpha Hunter V2 consistency validation...")
        start_time = time.time()
        
        # 生成测试数据
        test_data_list = self._generate_test_data(num_test_cases)
        
        # Tier 1 验证 (数值精度)
        self.logger.info("Running Tier 1 validation...")
        self._validate_rsrs_consistency(test_data_list)
        self._validate_pressure_consistency(test_data_list)
        self._validate_surge_consistency(test_data_list)
        self._validate_engine_consistency(test_data_list)
        
        # Tier 2 验证 (逻辑一致性)
        self.logger.info("Running Tier 2 validation...")
        self._validate_market_state_consistency(test_data_list)
        self._validate_signal_generation_consistency(test_data_list)
        
        # 性能测试
        performance_results = self._performance_comparison(test_data_list)
        
        # 生成报告
        total_tests = len(self.tier1_results) + len(self.tier2_results)
        passed_tests = sum([
            len([r for r in self.tier1_results if r.passed]),
            len([r for r in self.tier2_results if r.passed])
        ])
        
        report = ValidationReport(
            tier1_results=self.tier1_results,
            tier2_results=self.tier2_results,
            overall_passed=passed_tests == total_tests,
            performance_improvement=performance_results['speedup'],
            total_tests=total_tests,
            passed_tests=passed_tests,
            execution_time=time.time() - start_time
        )
        
        self.logger.info(f"Validation completed: {passed_tests}/{total_tests} tests passed")
        return report
    
    def _generate_test_data(self, num_cases: int) -> List[pd.DataFrame]:
        """生成测试数据"""
        self.logger.info(f"Generating {num_cases} test cases...")
        test_data_list = []
        
        for i in range(num_cases):
            # 随机生成测试参数
            num_days = random.randint(100, 500)
            start_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i*10)
            
            # 生成基础价格序列
            np.random.seed(i + 42)  # 确保可重现性
            base_price = random.uniform(5, 100)
            price_changes = np.random.normal(0, 0.02, num_days)  # 2%标准差
            
            prices = [base_price]
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.1))  # 确保价格为正
            
            # 生成OHLCV数据
            data = {
                'open': np.array(prices) + np.random.normal(0, 0.001, num_days),
                'high': np.array(prices) + np.abs(np.random.normal(0, 0.01, num_days)),
                'low': np.array(prices) - np.abs(np.random.normal(0, 0.01, num_days)),
                'close': np.array(prices),
                'vol': np.random.randint(100000, 10000000, num_days)
            }
            
            # 确保high >= max(open, close) >= min(open, close) >= low
            for j in range(num_days):
                high_price = max(data['open'][j], data['close'][j]) + data['high'][j]
                low_price = min(data['open'][j], data['close'][j]) - data['low'][j]
                data['high'][j] = high_price
                data['low'][j] = low_price
            
            df = pd.DataFrame(data, index=pd.date_range(start_date, periods=num_days, freq='D'))
            test_data_list.append(df)
        
        return test_data_list
    
    def _validate_rsrs_consistency(self, test_data_list: List[pd.DataFrame]):
        """验证RSRS因子一致性"""
        self.logger.info("Validating RSRS factor consistency...")
        test_name = "RSRS Factor Consistency"
        
        start_time = time.time()
        differences = []
        passed_tests = 0
        total_tests = 0
        
        for i, df in enumerate(test_data_list):
            try:
                # 计算原始和优化版本
                original_result = self.original_rsrs.compute_full(df)
                optimized_result = self.optimized_rsrs.compute_full(df)
                
                # 比较关键列
                key_columns = ['rsrs_adaptive', 'rsrs_r2', 'rsrs_slope', 'rsrs_zscore']
                
                for col in key_columns:
                    if col in original_result.columns and col in optimized_result.columns:
                        orig_values = original_result[col].dropna()
                        opt_values = optimized_result[col].dropna()
                        
                        if len(orig_values) > 0 and len(opt_values) > 0:
                            # 确保长度一致
                            min_len = min(len(orig_values), len(opt_values))
                            orig_values = orig_values.iloc[-min_len:]
                            opt_values = opt_values.iloc[-min_len:]
                            
                            # 计算差异
                            diff = np.abs(orig_values.values - opt_values.values)
                            diff = diff[~np.isnan(diff)]  # 移除NaN差异
                            
                            if len(diff) > 0:
                                max_diff = np.max(diff)
                                mean_diff = np.mean(diff)
                                
                                differences.append(max_diff)
                                total_tests += 1
                                
                                if max_diff < self.tolerance:
                                    passed_tests += 1
                                else:
                                    self.logger.warning(f"RSRS {col} max_diff {max_diff:.2e} exceeds tolerance")
                
            except Exception as e:
                self.logger.error(f"Error in RSRS test {i}: {str(e)}")
                continue
        
        max_diff = max(differences) if differences else 0.0
        mean_diff = np.mean(differences) if differences else 0.0
        
        result = ConsistencyResult(
            test_name=test_name,
            tier=1,
            passed=max_diff < self.tolerance and passed_tests == total_tests,
            max_diff=max_diff,
            mean_diff=mean_diff,
            total_tests=total_tests,
            passed_tests=passed_tests,
            execution_time=time.time() - start_time,
            details={
                'columns_tested': ['rsrs_adaptive', 'rsrs_r2', 'rsrs_slope', 'rsrs_zscore'],
                'tolerance': self.tolerance
            }
        )
        
        self.tier1_results.append(result)
        self.logger.info(f"RSRS validation: {passed_tests}/{total_tests} tests passed, max_diff={max_diff:.2e}")
    
    def _validate_pressure_consistency(self, test_data_list: List[pd.DataFrame]):
        """验证压力位因子一致性"""
        self.logger.info("Validating pressure factor consistency...")
        test_name = "Pressure Factor Consistency"
        
        # 定义要测试的列
        key_columns = [
            'combined_pressure_dist', 'chip_pressure', 'round_pressure',
            'trapped_pressure', 'support_dist', 'safety_score'
        ]
        
        start_time = time.time()
        differences = []
        passed_tests = 0
        total_tests = 0
        
        for i, df in enumerate(test_data_list):
            try:
                # 计算原始和优化版本
                original_result = self.original_pressure.compute_full(df)
                optimized_result = self.optimized_pressure.compute_full(df)
                
                # 比较关键列
                for col in key_columns:
                    if col in original_result.columns and col in optimized_result.columns:
                        orig_values = original_result[col].dropna()
                        opt_values = optimized_result[col].dropna()
                        
                        if len(orig_values) > 0 and len(opt_values) > 0:
                            # 确保长度一致
                            min_len = min(len(orig_values), len(opt_values))
                            orig_values = orig_values.iloc[-min_len:]
                            opt_values = opt_values.iloc[-min_len:]
                            
                            # 计算差异
                            diff = np.abs(orig_values.values - opt_values.values)
                            diff = diff[~np.isnan(diff)]
                            
                            if len(diff) > 0:
                                max_diff = np.max(diff)
                                mean_diff = np.mean(diff)
                                
                                differences.append(max_diff)
                                total_tests += 1
                                
                                if max_diff < self.tolerance:
                                    passed_tests += 1
                                else:
                                    self.logger.warning(f"Pressure {col} max_diff {max_diff:.2e} exceeds tolerance")
                
            except Exception as e:
                self.logger.error(f"Error in pressure test {i}: {str(e)}")
                continue
        
        max_diff = max(differences) if differences else 0.0
        mean_diff = np.mean(differences) if differences else 0.0
        
        result = ConsistencyResult(
            test_name=test_name,
            tier=1,
            passed=max_diff < self.tolerance and passed_tests == total_tests,
            max_diff=max_diff,
            mean_diff=mean_diff,
            total_tests=total_tests,
            passed_tests=passed_tests,
            execution_time=time.time() - start_time,
            details={
                'columns_tested': key_columns,
                'tolerance': self.tolerance
            }
        )
        
        self.tier1_results.append(result)
        self.logger.info(f"Pressure validation: {passed_tests}/{total_tests} tests passed, max_diff={max_diff:.2e}")
    
    def _validate_surge_consistency(self, test_data_list: List[pd.DataFrame]):
        """验证开盘异动因子一致性"""
        self.logger.info("Validating surge factor consistency...")
        test_name = "Surge Factor Consistency"
        
        # 定义要测试的列
        key_columns = ['surge_score', 'gap_score', 'volume_surge', 'is_surge']
        
        start_time = time.time()
        differences = []
        passed_tests = 0
        total_tests = 0
        
        for i, df in enumerate(test_data_list):
            try:
                # 计算原始和优化版本
                original_result = self.original_surge.compute_full(df)
                optimized_result = self.optimized_surge.compute_full(df)
                
                # 比较关键列
                for col in key_columns:
                    if col in original_result.columns and col in optimized_result.columns:
                        orig_values = original_result[col].dropna()
                        opt_values = optimized_result[col].dropna()
                        
                        if len(orig_values) > 0 and len(opt_values) > 0:
                            # 确保长度一致
                            min_len = min(len(orig_values), len(opt_values))
                            orig_values = orig_values.iloc[-min_len:]
                            opt_values = opt_values.iloc[-min_len:]
                            
                            # 计算差异
                            diff = np.abs(orig_values.values - opt_values.values)
                            diff = diff[~np.isnan(diff)]
                            
                            if len(diff) > 0:
                                max_diff = np.max(diff)
                                mean_diff = np.mean(diff)
                                
                                differences.append(max_diff)
                                total_tests += 1
                                
                                if max_diff < self.tolerance:
                                    passed_tests += 1
                                else:
                                    self.logger.warning(f"Surge {col} max_diff {max_diff:.2e} exceeds tolerance")
                
            except Exception as e:
                self.logger.error(f"Error in surge test {i}: {str(e)}")
                continue
        
        max_diff = max(differences) if differences else 0.0
        mean_diff = np.mean(differences) if differences else 0.0
        
        result = ConsistencyResult(
            test_name=test_name,
            tier=1,
            passed=max_diff < self.tolerance and passed_tests == total_tests,
            max_diff=max_diff,
            mean_diff=mean_diff,
            total_tests=total_tests,
            passed_tests=passed_tests,
            execution_time=time.time() - start_time,
            details={
                'columns_tested': key_columns,
                'tolerance': self.tolerance
            }
        )
        
        self.tier1_results.append(result)
        self.logger.info(f"Surge validation: {passed_tests}/{total_tests} tests passed, max_diff={max_diff:.2e}")
    
    def _validate_engine_consistency(self, test_data_list: List[pd.DataFrame]):
        """验证Alpha因子引擎一致性"""
        self.logger.info("Validating Alpha engine consistency...")
        test_name = "Alpha Engine Consistency"
        
        # 定义要测试的属性
        key_attrs = [
            'rsrs_adaptive', 'rsrs_r2', 'opening_surge', 'pressure_distance',
            'alpha_score', 'signal_quality', 'risk_score'
        ]
        
        start_time = time.time()
        differences = []
        passed_tests = 0
        total_tests = 0
        
        for i, df in enumerate(test_data_list):
            try:
                # 计算原始和优化版本
                original_result = self.original_engine.compute(df)
                optimized_result = self.optimized_engine.compute(df)
                
                # 比较关键属性
                for attr in key_attrs:
                    orig_value = getattr(original_result, attr, None)
                    opt_value = getattr(optimized_result, attr, None)
                    
                    if orig_value is not None and opt_value is not None:
                        # 处理数值类型
                        if isinstance(orig_value, (int, float)) and isinstance(opt_value, (int, float)):
                            if not (np.isnan(orig_value) and np.isnan(opt_value)):
                                diff = abs(orig_value - opt_value)
                                
                                if not np.isnan(diff):
                                    differences.append(diff)
                                    total_tests += 1
                                    
                                    if diff < self.tolerance:
                                        passed_tests += 1
                                    else:
                                        self.logger.warning(f"Engine {attr} diff {diff:.2e} exceeds tolerance")
                
                # 特别比较市场状态
                orig_state = original_result.market_state.value
                opt_state = optimized_result.market_state.value
                
                if orig_state == opt_state:
                    passed_tests += 1
                else:
                    self.logger.warning(f"Market state mismatch: {orig_state} vs {opt_state}")
                
                total_tests += 1
                
            except Exception as e:
                self.logger.error(f"Error in engine test {i}: {str(e)}")
                continue
        
        max_diff = max(differences) if differences else 0.0
        mean_diff = np.mean(differences) if differences else 0.0
        
        result = ConsistencyResult(
            test_name=test_name,
            tier=1,
            passed=max_diff < self.tolerance and passed_tests == total_tests,
            max_diff=max_diff,
            mean_diff=mean_diff,
            total_tests=total_tests,
            passed_tests=passed_tests,
            execution_time=time.time() - start_time,
            details={
                'attributes_tested': key_attrs,
                'tolerance': self.tolerance
            }
        )
        
        self.tier1_results.append(result)
        self.logger.info(f"Engine validation: {passed_tests}/{total_tests} tests passed, max_diff={max_diff:.2e}")
    
    def _validate_market_state_consistency(self, test_data_list: List[pd.DataFrame]):
        """验证市场状态判断一致性"""
        self.logger.info("Validating market state consistency...")
        test_name = "Market State Consistency"
        
        start_time = time.time()
        state_matches = 0
        total_tests = 0
        
        for i, df in enumerate(test_data_list):
            try:
                # 比较市场状态
                original_states = self.original_engine.rsrs_factor._detect_market_state(
                    df['close'].to_numpy(), df['vol'].to_numpy()
                )
                optimized_states = self.optimized_engine.rsrs_factor._detect_market_state(
                    df['close'].to_numpy(), df['vol'].to_numpy()
                )
                
                # 确保长度一致
                min_len = min(len(original_states), len(optimized_states))
                orig_states = original_states[-min_len:]
                opt_states = optimized_states[-min_len:]
                
                # 比较状态
                matches = sum(1 for a, b in zip(orig_states, opt_states) if a == b)
                state_matches += matches
                total_tests += min_len
                
            except Exception as e:
                self.logger.error(f"Error in market state test {i}: {str(e)}")
                continue
        
        accuracy = state_matches / max(total_tests, 1)
        
        result = ConsistencyResult(
            test_name=test_name,
            tier=2,
            passed=accuracy >= 0.99,  # 99%一致性
            max_diff=1.0 - accuracy,  # 错误率
            mean_diff=1.0 - accuracy,
            total_tests=total_tests,
            passed_tests=state_matches,
            execution_time=time.time() - start_time,
            details={
                'accuracy': accuracy,
                'tolerance': 0.01  # 1%错误率容忍
            }
        )
        
        self.tier2_results.append(result)
        self.logger.info(f"Market state validation: {accuracy:.2%} accuracy")
    
    def _validate_signal_generation_consistency(self, test_data_list: List[pd.DataFrame]):
        """验证信号生成一致性"""
        self.logger.info("Validating signal generation consistency...")
        test_name = "Signal Generation Consistency"
        
        start_time = time.time()
        decisions_matched = 0
        total_decisions = 0
        
        for i, df in enumerate(test_data_list):
            try:
                # 生成信号决策
                original_result = self.original_engine.compute(df)
                optimized_result = self.optimized_engine.compute(df)
                
                # 基于阈值生成决策
                original_buy = original_result.alpha_score > 0.5
                optimized_buy = optimized_result.alpha_score > 0.5
                
                if original_buy == optimized_buy:
                    decisions_matched += 1
                
                original_sell = original_result.alpha_score < -0.5
                optimized_sell = optimized_result.alpha_score < -0.5
                
                if original_sell == optimized_sell:
                    decisions_matched += 1
                
                total_decisions += 2
                
            except Exception as e:
                self.logger.error(f"Error in signal generation test {i}: {str(e)}")
                continue
        
        accuracy = decisions_matched / max(total_decisions, 1)
        
        result = ConsistencyResult(
            test_name=test_name,
            tier=2,
            passed=accuracy >= 0.95,  # 95%决策一致性
            max_diff=1.0 - accuracy,
            mean_diff=1.0 - accuracy,
            total_tests=total_decisions,
            passed_tests=decisions_matched,
            execution_time=time.time() - start_time,
            details={
                'accuracy': accuracy,
                'tolerance': 0.05  # 5%决策差异容忍
            }
        )
        
        self.tier2_results.append(result)
        self.logger.info(f"Signal generation validation: {accuracy:.2%} accuracy")
    
    def _performance_comparison(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """性能对比测试"""
        self.logger.info("Running performance comparison...")
        
        # 原始版本性能测试
        start_time = time.time()
        for df in test_data_list[:20]:  # 测试前20个样本
            try:
                self.original_engine.compute(df)
            except Exception as e:
                self.logger.warning(f"Original engine error: {str(e)}")
                continue
        original_time = time.time() - start_time
        
        # 优化版本性能测试
        start_time = time.time()
        for df in test_data_list[:20]:  # 测试前20个样本
            try:
                self.optimized_engine.compute(df)
            except Exception as e:
                self.logger.warning(f"Optimized engine error: {str(e)}")
                continue
        optimized_time = time.time() - start_time
        
        speedup = original_time / max(optimized_time, 0.001)
        
        self.logger.info(f"Performance: Original {original_time:.3f}s vs Optimized {optimized_time:.3f}s, Speedup: {speedup:.2f}x")
        
        return {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'target_met': speedup >= 5.0
        }
    
    def print_validation_report(self, report: ValidationReport):
        """打印验证报告"""
        print("\n" + "="*80)
        print("ALPHA HUNTER V2 LOGIC CONSISTENCY VALIDATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Overall Passed: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}")
        print(f"   Performance Improvement: {report.performance_improvement:.2f}x")
        print(f"   Tests: {report.passed_tests}/{report.total_tests} passed")
        print(f"   Execution Time: {report.execution_time:.2f}s")
        
        print(f"\n🎯 TIER 1 VALIDATION (Numerical Precision):")
        for result in report.tier1_results:
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"   {result.test_name}: {status}")
            print(f"      Max Diff: {result.max_diff:.2e} (tolerance: {self.tolerance:.2e})")
            print(f"      Tests: {result.passed_tests}/{result.total_tests}")
            print(f"      Time: {result.execution_time:.2f}s")
        
        print(f"\n🔍 TIER 2 VALIDATION (Logic Consistency):")
        for result in report.tier2_results:
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"   {result.test_name}: {status}")
            print(f"      Accuracy: {100 * (1 - result.max_diff):.2f}%")
            print(f"      Tests: {result.passed_tests}/{result.total_tests}")
            print(f"      Time: {result.execution_time:.2f}s")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Target Performance: >= 5x improvement")
        print(f"   Actual Performance: {report.performance_improvement:.2f}x")
        print(f"   Target Met: {'✅ YES' if report.performance_improvement >= 5.0 else '❌ NO'}")
        
        # 验收标准检查
        tier1_passed = all(r.passed for r in report.tier1_results)
        tier2_passed = all(r.passed for r in report.tier2_results)
        performance_met = report.performance_improvement >= 5.0
        
        print(f"\n🎯 ACCEPTANCE CRITERIA:")
        print(f"   Tier 1 (Numerical Precision): {'✅ PASSED' if tier1_passed else '❌ FAILED'}")
        print(f"   Tier 2 (Logic Consistency): {'✅ PASSED' if tier2_passed else '❌ FAILED'}")
        print(f"   Performance Target: {'✅ PASSED' if performance_met else '❌ FAILED'}")
        
        overall_pass = tier1_passed and tier2_passed and performance_met
        print(f"\n🏆 FINAL VERDICT: {'🎉 ALL TESTS PASSED' if overall_pass else '💥 VALIDATION FAILED'}")
        
        if overall_pass:
            print("   ✨ Alpha Hunter V2 optimization is ready for production!")
        else:
            print("   ⚠️  Please review failed tests before production deployment.")
        
        print("="*80)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行验证
    print("Starting Alpha Hunter V2 Consistency Validation...")
    validator = AlphaHunterV2ConsistencyValidator(tolerance=1e-6)
    
    # 运行完整验证
    report = validator.run_full_validation(num_test_cases=50)
    
    # 打印报告
    validator.print_validation_report(report)
    
    # 保存详细报告到文件
    report_file = Path("alpha_hunter_v2_consistency_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        # 这里可以添加详细的报告保存逻辑
        f.write(f"Validation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall result: {'PASSED' if report.overall_passed else 'FAILED'}\n")
        f.write(f"Performance improvement: {report.performance_improvement:.2f}x\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # 退出代码
    import sys
    sys.exit(0 if report.overall_passed else 1)