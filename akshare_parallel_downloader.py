"""
AKShare 并行下载器 - 完整生产版
=================================

特性：
- 多进程并行下载（4进程）
- 断点续传
- 自动重试（指数退避）
- 限流保护
- 进度监控
- 数据验证
- 错误日志

使用方法：
    python akshare_parallel_downloader.py

或在代码中：
    from akshare_parallel_downloader import AKShareDownloader
    
    downloader = AKShareDownloader()
    downloader.download_all()
"""

import akshare as ak
import pandas as pd
import time
import random
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ==================== 日志配置 ====================
def setup_logging(log_dir: str = "logs"):
    """配置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"akshare_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ==================== 核心下载函数 ====================
def download_single_stock(
    code: str,
    start_date: str = "20140101",
    end_date: str = "20241231",
    max_retries: int = 3,
    delay_range: Tuple[float, float] = (0.2, 0.5)
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    下载单只股票（子进程执行）
    
    Args:
        code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        max_retries: 最大重试次数
        delay_range: 延迟范围（秒）
    
    Returns:
        (code, dataframe, error_msg)
        - 成功: (code, df, None)
        - 失败: (code, None, error_msg)
    """
    for attempt in range(max_retries):
        try:
            # 随机延迟（避免限流）
            time.sleep(random.uniform(*delay_range))
            
            # 下载数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            # 验证数据
            if df is None or df.empty:
                raise ValueError("数据为空")
            
            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'vol',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            # 添加股票代码
            df['code'] = code
            
            # 数据验证
            if len(df) < 10:
                raise ValueError(f"数据量过少: {len(df)}条")
            
            # 检查必需列
            required_cols = ['date', 'open', 'high', 'low', 'close', 'vol']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少列: {missing_cols}")
            
            return (code, df, None)
        
        except Exception as e:
            error_msg = str(e)
            
            # 判断是否需要重试
            if attempt < max_retries - 1:
                # 指数退避
                wait_time = 2 ** attempt
                
                # 特殊处理限流错误
                if "429" in error_msg or "限流" in error_msg or "频繁" in error_msg:
                    wait_time = 5 * (attempt + 1)  # 限流错误等待更久
                
                time.sleep(wait_time)
            else:
                # 最后一次失败，返回错误
                return (code, None, f"下载失败（已重试{max_retries}次）: {error_msg}")
    
    return (code, None, "未知错误")


# ==================== 主下载器类 ====================
class AKShareDownloader:
    """AKShare 并行下载器"""
    
    def __init__(
        self,
        output_dir: str = "data/akshare",
        n_workers: int = 4,
        max_retries: int = 3,
        delay_range: Tuple[float, float] = (0.2, 0.5),
        start_date: str = "20140101",
        end_date: str = None
    ):
        """
        初始化下载器
        
        Args:
            output_dir: 输出目录
            n_workers: 并行进程数（推荐 2-4）
            max_retries: 最大重试次数
            delay_range: 延迟范围（秒）
            start_date: 开始日期
            end_date: 结束日期（默认今天）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_workers = min(n_workers, mp.cpu_count())
        self.max_retries = max_retries
        self.delay_range = delay_range
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y%m%d")
        
        # 日志
        self.logger = setup_logging(self.output_dir.parent / "logs")
        
        # 统计文件
        self.stats_file = self.output_dir / "download_stats.json"
        self.failed_file = self.output_dir / "failed_stocks.txt"
    
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        self.logger.info("正在获取股票列表...")
        
        try:
            # 获取A股列表
            stock_info = ak.stock_info_a_code_name()
            codes = stock_info['code'].tolist()
            
            self.logger.info(f"✓ 获取到 {len(codes)} 只股票")
            
            return codes
        
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {e}")
            raise
    
    def get_downloaded_codes(self) -> set:
        """获取已下载的股票代码"""
        downloaded = set()
        
        # 检查 parquet 文件
        for f in self.output_dir.glob("*.parquet"):
            downloaded.add(f.stem)
        
        # 检查 csv 文件（兼容）
        for f in self.output_dir.glob("*.csv"):
            downloaded.add(f.stem)
        
        return downloaded
    
    def save_stock_data(self, code: str, df: pd.DataFrame) -> bool:
        """
        保存股票数据
        
        Args:
            code: 股票代码
            df: 数据
        
        Returns:
            是否成功
        """
        try:
            # 保存为 parquet（推荐，体积小、速度快）
            output_path = self.output_dir / f"{code}.parquet"
            df.to_parquet(output_path, index=False)
            
            return True
        
        except Exception as e:
            self.logger.error(f"保存 {code} 失败: {e}")
            return False
    
    def save_statistics(self, stats: Dict):
        """保存统计信息"""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存统计信息失败: {e}")
    
    def save_failed_list(self, failed: List[Tuple[str, str]]):
        """保存失败列表"""
        try:
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                for code, error in failed:
                    f.write(f"{code}\t{error}\n")
        except Exception as e:
            self.logger.error(f"保存失败列表失败: {e}")
    
    def download_batch(
        self,
        stock_codes: List[str],
        resume: bool = True
    ) -> Dict:
        """
        批量下载
        
        Args:
            stock_codes: 股票代码列表
            resume: 是否断点续传
        
        Returns:
            统计信息
        """
        # 断点续传
        if resume:
            downloaded = self.get_downloaded_codes()
            pending = [c for c in stock_codes if c not in downloaded]
            
            self.logger.info(f"✓ 已下载: {len(downloaded)} 只")
            self.logger.info(f"⏳ 待下载: {len(pending)} 只")
            
            if not pending:
                self.logger.info("🎉 全部已下载！")
                return {
                    'total': len(stock_codes),
                    'downloaded': len(downloaded),
                    'pending': 0,
                    'success': 0,
                    'failed': 0
                }
        else:
            pending = stock_codes
        
        # 开始下载
        self.logger.info("=" * 70)
        self.logger.info("开始并行下载")
        self.logger.info("=" * 70)
        self.logger.info(f"总数: {len(pending)} 只")
        self.logger.info(f"进程数: {self.n_workers}")
        self.logger.info(f"日期范围: {self.start_date} - {self.end_date}")
        self.logger.info("=" * 70)
        
        success_count = 0
        failed_list = []
        
        t0 = time.time()
        
        # 多进程下载
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    download_single_stock,
                    code,
                    self.start_date,
                    self.end_date,
                    self.max_retries,
                    self.delay_range
                ): code for code in pending
            }
            
            # 收集结果
            completed = 0
            
            for future in as_completed(futures):
                code, df, error = future.result()
                
                if df is not None:
                    # 保存数据
                    if self.save_stock_data(code, df):
                        success_count += 1
                    else:
                        failed_list.append((code, "保存失败"))
                else:
                    failed_list.append((code, error))
                    self.logger.warning(f"✗ {code}: {error}")
                
                completed += 1
                
                # 显示进度
                if completed % 50 == 0 or completed == len(pending):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - completed) / rate if rate > 0 else 0
                    
                    self.logger.info(
                        f"进度: {completed}/{len(pending)} "
                        f"({completed/len(pending)*100:.1f}%) | "
                        f"成功: {success_count} | "
                        f"失败: {len(failed_list)} | "
                        f"速度: {rate:.2f}股/秒 | "
                        f"ETA: {eta/60:.1f}分钟"
                    )
        
        # 统计
        elapsed = time.time() - t0
        
        stats = {
            'total': len(stock_codes),
            'downloaded': len(self.get_downloaded_codes()) - len(pending) + success_count,
            'pending': len(pending),
            'success': success_count,
            'failed': len(failed_list),
            'elapsed_seconds': round(elapsed, 2),
            'elapsed_minutes': round(elapsed / 60, 2),
            'rate': round(len(pending) / elapsed if elapsed > 0 else 0, 2),
            'start_time': datetime.fromtimestamp(t0).isoformat(),
            'end_time': datetime.now().isoformat()
        }
        
        # 保存统计
        self.save_statistics(stats)
        
        # 保存失败列表
        if failed_list:
            self.save_failed_list(failed_list)
        
        # 打印汇总
        self.logger.info("\n" + "=" * 70)
        self.logger.info("下载完成！")
        self.logger.info("=" * 70)
        self.logger.info(f"✓ 成功: {success_count}/{len(pending)}")
        self.logger.info(f"✗ 失败: {len(failed_list)}")
        self.logger.info(f"⏱️  总耗时: {elapsed/60:.1f} 分钟")
        self.logger.info(f"🚀 平均速度: {len(pending)/elapsed:.2f} 股/秒")
        self.logger.info("=" * 70)
        
        if failed_list:
            self.logger.info(f"\n失败列表已保存到: {self.failed_file}")
            self.logger.info(f"前10个失败股票:")
            for code, error in failed_list[:10]:
                self.logger.info(f"  - {code}: {error}")
            if len(failed_list) > 10:
                self.logger.info(f"  ... 还有 {len(failed_list)-10} 只")
        
        return stats
    
    def download_all(self, resume: bool = True) -> Dict:
        """
        下载全部股票
        
        Args:
            resume: 是否断点续传
        
        Returns:
            统计信息
        """
        # 获取股票列表
        stock_codes = self.get_stock_list()
        
        # 批量下载
        stats = self.download_batch(stock_codes, resume=resume)
        
        return stats
    
    def retry_failed(self) -> Dict:
        """
        重试失败的股票
        
        Returns:
            统计信息
        """
        if not self.failed_file.exists():
            self.logger.info("没有失败列表")
            return {}
        
        # 读取失败列表
        failed_codes = []
        with open(self.failed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    code = line.split('\t')[0]
                    failed_codes.append(code)
        
        self.logger.info(f"重试 {len(failed_codes)} 只失败股票...")
        
        # 重试
        stats = self.download_batch(failed_codes, resume=False)
        
        return stats
    
    def verify_data(self, sample_size: int = 100) -> Dict:
        """
        验证数据完整性
        
        Args:
            sample_size: 抽样数量
        
        Returns:
            验证结果
        """
        self.logger.info(f"验证数据完整性（抽样 {sample_size} 只）...")
        
        # 获取已下载文件
        files = list(self.output_dir.glob("*.parquet"))
        
        if not files:
            self.logger.warning("没有找到数据文件")
            return {}
        
        # 随机抽样
        import random
        sample_files = random.sample(files, min(sample_size, len(files)))
        
        issues = []
        
        for f in sample_files:
            try:
                df = pd.read_parquet(f)
                
                # 检查1: 数据量
                if len(df) < 10:
                    issues.append((f.stem, f"数据量过少: {len(df)}条"))
                
                # 检查2: 必需列
                required_cols = ['date', 'open', 'high', 'low', 'close', 'vol']
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    issues.append((f.stem, f"缺少列: {missing}"))
                
                # 检查3: 缺失值
                null_cols = df[required_cols].isnull().sum()
                if null_cols.any():
                    issues.append((f.stem, f"有缺失值: {null_cols[null_cols > 0].to_dict()}"))
                
            except Exception as e:
                issues.append((f.stem, f"读取失败: {e}"))
        
        # 报告
        self.logger.info(f"✓ 验证完成: {len(sample_files)} 个文件")
        
        if issues:
            self.logger.warning(f"发现 {len(issues)} 个问题:")
            for code, issue in issues[:10]:
                self.logger.warning(f"  - {code}: {issue}")
        else:
            self.logger.info("✓ 所有抽样文件均正常")
        
        return {
            'total_checked': len(sample_files),
            'issues_found': len(issues),
            'issues': issues
        }


# ==================== 命令行接口 ====================
def main():
    """命令行主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AKShare 并行下载器")
    
    parser.add_argument(
        '--output-dir',
        default='data/akshare',
        help='输出目录（默认: data/akshare）'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='并行进程数（默认: 4）'
    )
    
    parser.add_argument(
        '--start-date',
        default='20140101',
        help='开始日期（默认: 20140101）'
    )
    
    parser.add_argument(
        '--end-date',
        default=None,
        help='结束日期（默认: 今天）'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='不使用断点续传（重新下载）'
    )
    
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='重试失败的股票'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='验证数据完整性'
    )
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = AKShareDownloader(
        output_dir=args.output_dir,
        n_workers=args.workers,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 执行操作
    if args.retry_failed:
        # 重试失败
        downloader.retry_failed()
    
    elif args.verify:
        # 验证数据
        downloader.verify_data()
    
    else:
        # 正常下载
        downloader.download_all(resume=not args.no_resume)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 方式1: 命令行使用
    # python akshare_parallel_downloader.py
    
    # 方式2: 代码调用
    """
    downloader = AKShareDownloader(
        output_dir="data/akshare",
        n_workers=4,
        start_date="20140101"
    )
    
    # 下载全部（支持断点续传）
    stats = downloader.download_all(resume=True)
    
    # 重试失败
    downloader.retry_failed()
    
    # 验证数据
    downloader.verify_data()
    """
    
    main()
