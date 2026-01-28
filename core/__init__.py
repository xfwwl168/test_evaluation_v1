# ============================================================================
# 文件: core/__init__.py
# ============================================================================
"""
核心模块 - 数据采集与存储

提供股票数据下载、存储、更新的完整解决方案
"""

from .node_scanner import NodeScanner
from .downloader import StockDownloader
from .database import StockDatabase
from .updater import DataUpdater

__all__ = [
    'NodeScanner',      # TDX 节点扫描器
    'StockDownloader',  # 股票数据下载器
    'StockDatabase',    # DuckDB 数据库接口
    'DataUpdater'       # 数据更新调度器
]

__version__ = '1.0.0'
