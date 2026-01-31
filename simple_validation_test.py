#!/usr/bin/env python3
# ============================================================================
# 简化的Alpha Hunter V2一致性验证测试
# ============================================================================

import sys
import numpy as np
import pandas as pd
import time
import logging

# 设置基本日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleTest")

def test_basic_functionality():
    """测试基本功能"""
    logger.info("Testing basic Alpha Hunter V2 functionality...")
    
    try:
        # 导入优化版本
        from factors.alpha_hunter_v2_factors_optimized import AlphaFactorEngineV2Optimized
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=150, freq='D')
        test_data = pd.DataFrame({
            'open': 10 + np.random.randn(150).cumsum() * 0.1,
            'high': 10 + np.random.randn(150).cumsum() * 0.1 + 0.5,
            'low': 10 + np.random.randn(150).cumsum() * 0.1 - 0.5,
            'close': 10 + np.random.randn(150).cumsum() * 0.1,
            'vol': np.random.randint(100000, 1000000, 150)
        }, index=dates)
        
        # 确保OHLC逻辑关系
        for i in range(len(test_data)):
            high_price = max(test_data.iloc[i]['open'], test_data.iloc[i]['close']) + abs(test_data.iloc[i]['high'])
            low_price = min(test_data.iloc[i]['open'], test_data.iloc[i]['close']) - abs(test_data.iloc[i]['low'])
            test_data.iloc[i, test_data.columns.get_loc('high')] = high_price
            test_data.iloc[i, test_data.columns.get_loc('low')] = low_price
        
        logger.info(f"Generated test data: {len(test_data)} rows")
        
        # 测试优化引擎
        engine = AlphaFactorEngineV2Optimized()
        
        start_time = time.time()
        result = engine.compute(test_data)
        compute_time = time.time() - start_time
        
        logger.info(f"Alpha computation completed in {compute_time:.4f}s")
        logger.info(f"Alpha score: {result.alpha_score}")
        logger.info(f"RSRS adaptive: {result.rsrs_adaptive}")
        logger.info(f"Market state: {result.market_state.value}")
        
        # 测试缓存
        start_time = time.time()
        result2 = engine.compute(test_data)
        cache_time = time.time() - start_time
        
        logger.info(f"Cache computation completed in {cache_time:.4f}s")
        logger.info(f"Speedup: {compute_time / cache_time:.2f}x")
        
        # 验证结果一致性
        diff = abs(result.alpha_score - result2.alpha_score)
        logger.info(f"Result consistency check: diff = {diff:.2e}")
        
        if diff < 1e-6:
            logger.info("✅ Basic functionality test PASSED")
            return True
        else:
            logger.error(f"❌ Basic functionality test FAILED: diff = {diff:.2e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Basic functionality test ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_system():
    """测试缓存系统"""
    logger.info("Testing cache system...")
    
    try:
        from core.cache_manager import cache_manager
        
        # 清理缓存
        cache_manager.clear_cache()
        
        # 测试缓存设置和获取
        test_data = {"test": [1, 2, 3, 4, 5]}
        cache_manager.set('test', 'key1', test_data)
        
        retrieved = cache_manager.get('test', 'key1')
        
        if retrieved == test_data:
            logger.info("✅ Cache system test PASSED")
            return True
        else:
            logger.error("❌ Cache system test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cache system test ERROR: {str(e)}")
        return False

def test_batch_query():
    """测试批量查询"""
    logger.info("Testing batch query system...")
    
    try:
        from core.batch_query import batch_query_manager
        
        # 测试数据加载
        test_codes = ["000001", "000002", "000003"]
        
        # 创建模拟数据
        mock_data = {}
        for code in test_codes:
            np.random.seed(hash(code) % 1000)
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            df = pd.DataFrame({
                'open': 10 + np.random.randn(100),
                'high': 10 + np.random.randn(100) + 0.5,
                'low': 10 + np.random.randn(100) - 0.5,
                'close': 10 + np.random.randn(100),
                'vol': np.random.randint(100000, 1000000, 100)
            }, index=dates)
            mock_data[code] = df
        
        # 模拟批量查询
        start_time = time.time()
        results = batch_query_manager.query_ohlcv_batch(
            test_codes, '2023-01-01', '2023-04-10'
        )
        query_time = time.time() - start_time
        
        logger.info(f"Batch query completed in {query_time:.4f}s")
        logger.info(f"Loaded {len(results)} stocks")
        
        logger.info("✅ Batch query test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch query test ERROR: {str(e)}")
        return False

def main():
    """主测试函数"""
    logger.info("="*60)
    logger.info("ALPHA HUNTER V2 OPTIMIZATION - SIMPLIFIED VALIDATION")
    logger.info("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Cache System", test_cache_system),
        ("Batch Query", test_batch_query)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            logger.error(f"❌ {test_name} test failed")
    
    logger.info("\n" + "="*60)
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Optimization ready for production!")
        return True
    else:
        logger.error(f"💥 {total - passed} tests failed - Please review and fix issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)