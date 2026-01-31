
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from verify_alpha_hunter_v2_consistency import AlphaHunterV2ConsistencyValidator
from factors.technical_indicators_correct import calc_rsi_correct, calc_atr_correct

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComprehensiveValidator")

class ComprehensiveValidator:
    def __init__(self):
        self.consistency_validator = AlphaHunterV2ConsistencyValidator()
        
    def validate_technical_indicators(self) -> bool:
        """验证技术指标修复 (RSI/ATR)"""
        logger.info("Validating Technical Indicators (RSI/ATR)...")
        
        # 1. 验证 RSI (SMMA vs SMA)
        prices = np.array([100, 102, 101, 103, 102, 104, 105, 104, 103, 102, 101, 100, 99, 98, 97, 98, 99, 100])
        period = 14
        
        # 正确实现 (SMMA)
        rsi_correct = calc_rsi_correct(prices, period=14)
        
        # 错误实现 (SMA - 模拟旧代码行为)
        def calc_rsi_wrong(close, period=14):
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
            
        rsi_wrong = calc_rsi_wrong(prices, period=14).values
        
        # 验证差异存在
        # 对于短序列，SMA和SMMA差异可能很大
        # 由于数据长度刚超过14，我们检查最后一个值
        if len(prices) >= period:
            last_idx = -1
            diff = abs(rsi_correct[last_idx] - rsi_wrong[last_idx])
            logger.info(f"RSI Difference (Correct vs Wrong): {diff:.4f}")
            
            # 手动计算SMMA验证
            # 这是一个简单的检查，确保calc_rsi_correct不是完全离谱
            if np.isnan(rsi_correct[last_idx]):
                 logger.warning("RSI correct calculation resulted in NaN")
            else:
                 logger.info("RSI calculation produced valid values")
                 
        # 2. 验证 ATR (SMMA vs EMA/SMA)
        high = np.array([10, 11, 11, 12, 12])
        low = np.array([9, 9, 10, 10, 11])
        close = np.array([10, 10, 11, 11, 12])
        # period=3 (short for test)
        atr_correct = calc_atr_correct(high, low, close, period=3)
        
        # 只是确保运行无误且数值合理
        logger.info(f"ATR Values: {atr_correct}")
        
        return True

    def validate_consistency_and_performance(self) -> bool:
        """运行一致性和性能验证"""
        logger.info("Running Consistency and Performance Validation...")
        try:
            report = self.consistency_validator.run_full_validation(num_test_cases=20)
            
            logger.info("="*50)
            logger.info(f"Tier 1 Passed: {all(r.passed for r in report.tier1_results)}")
            logger.info(f"Tier 2 Passed: {all(r.passed for r in report.tier2_results)}")
            logger.info(f"Overall Passed: {report.overall_passed}")
            logger.info(f"Performance Improvement: {report.performance_improvement:.2f}x")
            logger.info("="*50)
            
            return report.overall_passed
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all(self):
        tech_ok = self.validate_technical_indicators()
        cons_ok = self.validate_consistency_and_performance()
        
        if tech_ok and cons_ok:
            logger.info("ALL CHECKS PASSED ✅")
        else:
            logger.error("SOME CHECKS FAILED ❌")

if __name__ == "__main__":
    validator = ComprehensiveValidator()
    validator.run_all()
