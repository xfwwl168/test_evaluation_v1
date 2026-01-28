# ============================================================================
# 文件: core/akshare_realtime_supplement.py
# ============================================================================
"""
AKShare 实时数据补充模块

功能:
- 使用 AKShare 获取当日全市场数据
- 列名自动映射
- 重试机制
- 返回执行统计
"""
import time
import random
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("Warning: akshare not installed. Install with: pip install akshare")
    ak = None


class AKShareRealtimeSupplement:
    """
    AKShare 实时数据补充器
    
    功能:
    - 使用 ak.stock_zh_a_hist() 下载当日全市场数据
    - 列名自动映射（代码→code, 日期→date等）
    - 重试机制：3次，延迟0.5-2秒
    - 返回执行统计：{stocks, rows, timestamp, errors}
    """
    
    # 列名映射关系
    COLUMN_MAPPING = {
        '代码': 'code',
        '日期': 'date',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'vol',
        '成交额': 'amount',
    }
    
    # 市场代码映射
    MARKET_MAPPING = {
        'sh': 0,  # 上海
        'sz': 1,  # 深圳
        'bj': 2,  # 北京
    }
    
    def __init__(self, retry_count: int = 3, retry_delay_range: tuple = (0.5, 2.0)):
        """
        初始化 AKShare 实时数据补充器
        
        Args:
            retry_count: 重试次数
            retry_delay_range: 重试延迟区间(秒)
        """
        if ak is None:
            raise ImportError("akshare not available. Please install with: pip install akshare")
        
        self.retry_count = retry_count
        self.retry_delay_range = retry_delay_range
        self.logger = logging.getLogger("AKShareSupplement")
        
        # 获取股票列表
        self.stock_list = self._get_stock_list()
        
    def _get_stock_list(self) -> List[Dict]:
        """
        获取 A 股股票列表
        
        Returns:
            股票列表 [{'code': '000001', 'name': '平安银行', 'market': 'sz'}, ...]
        """
        try:
            # 获取 A 股所有股票列表
            stock_info = ak.stock_info_a_code_name()
            
            # 过滤出有效的股票代码
            valid_stocks = []
            for _, row in stock_info.iterrows():
                code = str(row['code']).zfill(6)
                name = row['name']
                
                # 基本过滤：排除科创板、创业板等特殊板块（可选）
                if self._is_valid_stock(code):
                    market = self._get_market_from_code(code)
                    valid_stocks.append({
                        'code': code,
                        'name': name,
                        'market': market
                    })
            
            self.logger.info(f"Found {len(valid_stocks)} valid stocks")
            return valid_stocks
            
        except Exception as e:
            self.logger.error(f"Failed to get stock list: {e}")
            return []
    
    def _is_valid_stock(self, code: str) -> bool:
        """
        判断是否为有效股票
        
        Args:
            code: 股票代码
            
        Returns:
            是否有效
        """
        # 排除特殊代码
        if len(code) != 6:
            return False
            
        # 基本A股代码范围
        if code.startswith(('60', '68')):  # 主板、科创板
            return True
        elif code.startswith(('00', '30')):  # 主板、中小板
            return True
        elif code.startswith('43'):  # 北交所
            return True
        
        return False
    
    def _get_market_from_code(self, code: str) -> int:
        """
        根据股票代码获取市场代码
        
        Args:
            code: 股票代码
            
        Returns:
            市场代码 0=上海, 1=深圳, 2=北京
        """
        if code.startswith(('60', '68')):
            return 0  # 上海
        elif code.startswith(('00', '30')):
            return 1  # 深圳
        elif code.startswith('43'):
            return 2  # 北京
        else:
            return 1  # 默认深圳
    
    def _get_stock_data(self, code: str, market: str, period: str = "daily") -> Optional[pd.DataFrame]:
        """
        获取单只股票数据
        
        Args:
            code: 股票代码
            market: 市场代码
            period: 周期 ('daily', 'weekly', 'monthly')
            
        Returns:
            股票数据DataFrame，失败返回None
        """
        # AKShare 市场代码
        market_map = {'sh': 'sh', 'sz': 'sz', 'bj': 'bj'}
        ak_market = market_map.get(market, 'sz')
        
        try:
            # 尝试获取最近一天的数据
            today = date.today()
            
            # 使用 ak.stock_zh_a_hist 获取数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period=period,
                start_date=today.strftime('%Y%m%d'),
                end_date=today.strftime('%Y%m%d'),
                adjust=""
            )
            
            if df is None or df.empty:
                return None
            
            # 列名映射
            df = self._map_columns(df)
            
            # 添加市场代码
            df['market'] = self.MARKET_MAPPING.get(ak_market, 1)
            
            return df
            
        except Exception as e:
            self.logger.debug(f"Failed to get data for {code}: {e}")
            return None
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        列名映射
        
        Args:
            df: 原始DataFrame
            
        Returns:
            映射后的DataFrame
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 重命名列
        df.rename(columns=self.COLUMN_MAPPING, inplace=True)
        
        # 处理日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        # 确保数值类型
        numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def download_today_data(self, limit_stocks: Optional[int] = None) -> Dict:
        """
        下载当日全市场数据
        
        Args:
            limit_stocks: 限制股票数量（用于测试）
            
        Returns:
            执行统计: {stocks, rows, timestamp, errors}
        """
        start_time = time.time()
        successful_downloads = []
        failed_downloads = []
        
        stocks_to_download = self.stock_list
        if limit_stocks:
            stocks_to_download = stocks_to_download[:limit_stocks]
        
        self.logger.info(f"Starting download for {len(stocks_to_download)} stocks")
        
        for i, stock in enumerate(stocks_to_download):
            code = stock['code']
            market = stock['market']
            
            # 转换市场代码为字符串
            market_str = 'sh' if market == 0 else ('bj' if market == 2 else 'sz')
            
            # 重试下载
            df = None
            for attempt in range(self.retry_count):
                try:
                    df = self._get_stock_data(code, market_str)
                    if df is not None and not df.empty:
                        break
                    
                    # 失败后延迟
                    if attempt < self.retry_count - 1:
                        delay = random.uniform(*self.retry_delay_range)
                        time.sleep(delay)
                        
                except Exception as e:
                    self.logger.debug(f"Attempt {attempt + 1} failed for {code}: {e}")
                    if attempt < self.retry_count - 1:
                        delay = random.uniform(*self.retry_delay_range)
                        time.sleep(delay)
            
            if df is not None and not df.empty:
                successful_downloads.append(df)
                self.logger.debug(f"Successfully downloaded {code} ({i+1}/{len(stocks_to_download)})")
            else:
                failed_downloads.append(code)
                self.logger.debug(f"Failed to download {code} ({i+1}/{len(stocks_to_download)})")
            
            # 进度报告
            if (i + 1) % 100 == 0:
                self.logger.info(f"Progress: {i+1}/{len(stocks_to_download)} ({len(successful_downloads)} successful)")
            
            # 防封：每50只股票休息一下
            if (i + 1) % 50 == 0:
                time.sleep(2)
        
        # 合并数据
        all_data = pd.concat(successful_downloads, ignore_index=True) if successful_downloads else pd.DataFrame()
        
        # 标记今日数据
        if not all_data.empty:
            all_data = self._mark_today_data(all_data)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'stocks': len(stocks_to_download),
            'successful': len(successful_downloads),
            'failed': len(failed_downloads),
            'rows': len(all_data),
            'timestamp': datetime.now(),
            'elapsed_seconds': round(elapsed_time, 2),
            'data': all_data,
            'errors': failed_downloads
        }
        
        self.logger.info(f"Download completed: {result['successful']}/{result['stocks']} successful, "
                        f"{result['rows']} rows, {result['elapsed_seconds']:.1f}s")
        
        return result
    
    def _mark_today_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记今日数据
        
        Args:
            df: 股票数据
            
        Returns:
            标记后的数据
        """
        if df.empty:
            return df
            
        df = df.copy()
        today = date.today()
        df['is_today'] = df['date'] == today
        
        return df
    
    def get_download_stats(self, result: Dict) -> str:
        """
        获取下载统计信息
        
        Args:
            result: 下载结果
            
        Returns:
            格式化的统计信息
        """
        if not result.get('data') is not None or result['data'].empty:
            return "No data downloaded"
        
        data = result['data']
        today_data = data[data['is_today'] == True] if 'is_today' in data.columns else data
        
        stats = f"""
=== AKShare 下载统计 ===
股票总数: {result['stocks']}
成功下载: {result['successful']}
失败数量: {result['failed']}
数据行数: {result['rows']}
今日数据: {len(today_data)}
耗时: {result['elapsed_seconds']:.1f}秒
平均速度: {result['successful']/result['elapsed_seconds']:.1f} 股票/秒
        """
        
        return stats.strip()


# ==================== 快捷接口 ====================
def download_today_realtime_data(retry_count: int = 3, limit_stocks: int = None) -> Dict:
    """
    一键下载当日实时数据
    
    Args:
        retry_count: 重试次数
        limit_stocks: 限制股票数量
        
    Returns:
        下载结果
    """
    supplement = AKShareRealtimeSupplement(retry_count=retry_count)
    return supplement.download_today_data(limit_stocks=limit_stocks)


# ==================== 单元测试 ====================
if __name__ == "__main__":
    import unittest
    
    class TestAKShareSupplement(unittest.TestCase):
        """AKShare 补充模块测试"""
        
        def test_download_today_data(self):
            """测试下载当日数据"""
            if ak is None:
                self.skipTest("akshare not available")
                
            supplement = AKShareRealtimeSupplement(retry_count=1)
            result = supplement.download_today_data(limit_stocks=10)
            
            self.assertIn('stocks', result)
            self.assertIn('successful', result)
            self.assertIn('data', result)
            self.assertGreaterEqual(result['successful'], 0)
            
            if not result['data'].empty:
                self.assertIn('code', result['data'].columns)
                self.assertIn('date', result['data'].columns)
        
        def test_column_mapping(self):
            """测试列名映射"""
            if ak is None:
                self.skipTest("akshare not available")
                
            supplement = AKShareRealtimeSupplement()
            
            # 创建测试数据
            test_df = pd.DataFrame({
                '代码': ['000001', '000002'],
                '日期': ['2024-01-01', '2024-01-01'],
                '开盘': [10.0, 20.0],
                '最高': [11.0, 21.0],
                '最低': [9.0, 19.0],
                '收盘': [10.5, 20.5],
                '成交量': [1000000, 2000000],
                '成交额': [10000000, 20000000]
            })
            
            mapped_df = supplement._map_columns(test_df)
            
            self.assertIn('code', mapped_df.columns)
            self.assertIn('date', mapped_df.columns)
            self.assertEqual(list(mapped_df['code']), ['000001', '000002'])
    
    # 运行测试
    print("Running AKShare Supplement Tests...")
    unittest.main(verbosity=2, exit=False)