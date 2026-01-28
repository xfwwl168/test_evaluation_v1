# ============================================================================
# 文件: live/signal_monitor.py
# ============================================================================
"""
实时信号监控模块

功能:
- 多策略并行监控
- 入场/离场信号检测
- 持仓状态跟踪
- 信号推送 (可选)
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import sys
from pathlib import Path

# 添加项目根目录
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.database import StockDatabase
from strategy import StrategyRegistry, BaseStrategy, Signal, OrderSide, StrategyContext
from config import settings


class SignalType(Enum):
    """信号类型"""
    ENTRY = "入场"
    EXIT = "离场"
    HOLD = "持有"


@dataclass
class MonitorSignal:
    """监控信号"""
    timestamp: str
    strategy: str
    signal_type: SignalType
    code: str
    price: float
    reason: str
    score: float = 0.0
    
    def __str__(self):
        icon = "🟢" if self.signal_type == SignalType.ENTRY else "🔴" if self.signal_type == SignalType.EXIT else "🟡"
        return f"{icon} [{self.strategy}] {self.signal_type.value} {self.code} @ {self.price:.2f} | {self.reason}"


@dataclass
class VirtualPosition:
    """虚拟持仓 (用于跟踪离场信号)"""
    code: str
    entry_price: float
    entry_date: str
    strategy: str
    quantity: int = 100


class SignalMonitor:
    """
    信号监控器
    
    支持:
    - 单策略/多策略监控
    - 入场+离场信号
    - 虚拟持仓跟踪
    """
    
    def __init__(
        self,
        strategies: List[str] = None,
        db_path: str = None,
        scan_interval: int = 60,
        max_signals: int = 20
    ):
        """
        Args:
            strategies: 策略名称列表, None=全部策略
            db_path: 数据库路径
            scan_interval: 扫描间隔 (秒)
            max_signals: 每次最多显示信号数
        """
        self.db = StockDatabase(db_path or str(settings.path.DB_PATH))
        self.scan_interval = scan_interval
        self.max_signals = max_signals
        
        # 加载策略
        self.strategies: Dict[str, BaseStrategy] = {}
        self._load_strategies(strategies)
        
        # 虚拟持仓 (用于跟踪离场信号)
        self.virtual_positions: Dict[str, Dict[str, VirtualPosition]] = {
            name: {} for name in self.strategies
        }
        
        # 信号历史
        self.signal_history: List[MonitorSignal] = []
        
        # 数据缓存
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._factors_cache: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger("SignalMonitor")
        self._running = False
    
    def _load_strategies(self, strategy_names: List[str] = None):
        """加载策略"""
        available = StrategyRegistry.list_all()
        
        if strategy_names is None or 'all' in strategy_names:
            strategy_names = available
        
        for name in strategy_names:
            if name in available:
                try:
                    strategy_cls = StrategyRegistry.get(name)
                    strategy = strategy_cls()
                    strategy.initialize()
                    self.strategies[name] = strategy
                    self.logger.info(f"Loaded strategy: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {name}: {e}")
        
        if not self.strategies:
            raise ValueError(f"No valid strategies. Available: {available}")
    
    def _load_market_data(self) -> pd.DataFrame:
        """加载最新市场数据"""
        stats = self.db.get_stats()
        latest_date = str(stats.get('max_date', ''))[:10]
        
        if not latest_date:
            return pd.DataFrame()
        
        # 加载最近 N 天数据用于因子计算
        lookback_days = 300
        start_date = (datetime.strptime(latest_date, '%Y-%m-%d') - timedelta(days=lookback_days * 1.5)).strftime('%Y-%m-%d')
        
        with self.db.connect() as conn:
            df = conn.execute(f"""
                SELECT code, market, date, open, high, low, close, vol, amount
                FROM daily_bars
                WHERE date BETWEEN '{start_date}' AND '{latest_date}'
                ORDER BY code, date
            """).fetchdf()
        
        return df, latest_date
    
    def _prepare_data(self, market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """准备策略所需数据"""
        data_cache = {}
        
        for code in market_data['code'].unique():
            stock_df = market_data[market_data['code'] == code].copy()
            if len(stock_df) >= 60:  # 至少60天数据
                stock_df['date'] = pd.to_datetime(stock_df['date'])
                stock_df.set_index('date', inplace=True)
                data_cache[code] = stock_df
        
        return data_cache
    
    def _compute_factors(self, data_cache: Dict[str, pd.DataFrame]):
        """预计算因子"""
        for name, strategy in self.strategies.items():
            try:
                self._factors_cache[name] = strategy.compute_factors(data_cache)
            except Exception as e:
                self.logger.warning(f"Factor computation failed for {name}: {e}")
                self._factors_cache[name] = {}
    
    def _build_context(
        self,
        strategy_name: str,
        current_date: str,
        current_data: pd.DataFrame,
        data_cache: Dict[str, pd.DataFrame]
    ) -> StrategyContext:
        """构建策略上下文"""
        # 获取虚拟持仓
        positions = {
            code: pos.quantity
            for code, pos in self.virtual_positions[strategy_name].items()
        }
        
        # 计算总权益 (简化)
        total_equity = 1_000_000  # 假设 100万
        cash = total_equity - sum(
            pos.entry_price * pos.quantity
            for pos in self.virtual_positions[strategy_name].values()
        )
        
        return StrategyContext(
            current_date=current_date,
            current_data=current_data,
            history_data=data_cache,
            factors=self._factors_cache.get(strategy_name, {}),
            positions=positions,
            cash=cash,
            total_equity=total_equity
        )
    
    def scan_once(self) -> List[MonitorSignal]:
        """执行一次扫描"""
        signals = []
        
        try:
            # 1. 加载数据
            market_data, latest_date = self._load_market_data()
            
            if market_data.empty:
                self.logger.warning("No market data available")
                return signals
            
            # 2. 准备数据
            if not self._data_cache:
                self._data_cache = self._prepare_data(market_data)
                self._compute_factors(self._data_cache)
            
            # 3. 获取当日数据
            current_data = market_data[
                market_data['date'].astype(str).str[:10] == latest_date
            ].copy()
            
            if current_data.empty:
                return signals
            
            # 4. 每个策略生成信号
            for name, strategy in self.strategies.items():
                try:
                    context = self._build_context(name, latest_date, current_data, self._data_cache)
                    strategy_signals = strategy.generate_signals(context)
                    
                    for sig in strategy_signals:
                        # 获取价格
                        price_row = current_data[current_data['code'] == sig.code]
                        price = price_row['close'].iloc[0] if not price_row.empty else 0
                        
                        # 判断信号类型
                        if sig.side == OrderSide.BUY:
                            signal_type = SignalType.ENTRY
                            # 记录虚拟持仓
                            self.virtual_positions[name][sig.code] = VirtualPosition(
                                code=sig.code,
                                entry_price=price,
                                entry_date=latest_date,
                                strategy=name
                            )
                        else:
                            signal_type = SignalType.EXIT
                            # 移除虚拟持仓
                            self.virtual_positions[name].pop(sig.code, None)
                        
                        monitor_signal = MonitorSignal(
                            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            strategy=name,
                            signal_type=signal_type,
                            code=sig.code,
                            price=price,
                            reason=sig.reason,
                            score=sig.weight
                        )
                        signals.append(monitor_signal)
                        
                except Exception as e:
                    self.logger.warning(f"Signal generation failed for {name}: {e}")
            
            # 记录历史
            self.signal_history.extend(signals)
            
        except Exception as e:
            self.logger.error(f"Scan failed: {e}")
            import traceback
            traceback.print_exc()
        
        return signals
    
    def run(self, duration: int = None):
        """
        运行监控
        
        Args:
            duration: 运行时长 (秒), None=永久运行
        """
        self._running = True
        start_time = time.time()
        scan_count = 0
        
        print("=" * 70)
        print("📡 信号监控已启动")
        print(f"   策略: {list(self.strategies.keys())}")
        print(f"   间隔: {self.scan_interval} 秒")
        print("   按 Ctrl+C 停止")
        print("=" * 70)
        
        try:
            while self._running:
                scan_count += 1
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\n[{now}] 第 {scan_count} 次扫描...")
                
                signals = self.scan_once()
                
                if signals:
                    print(f"\n🌟 发现 {len(signals)} 个信号:")
                    print("-" * 70)
                    
                    # 按策略分组显示
                    for name in self.strategies:
                        strat_signals = [s for s in signals if s.strategy == name]
                        if strat_signals:
                            print(f"\n【{name}】")
                            for sig in strat_signals[:self.max_signals]:
                                print(f"  {sig}")
                else:
                    print("   暂无新信号")
                
                # 显示当前虚拟持仓
                total_positions = sum(len(p) for p in self.virtual_positions.values())
                if total_positions > 0:
                    print(f"\n📊 当前监控持仓: {total_positions} 只")
                    for name, positions in self.virtual_positions.items():
                        if positions:
                            print(f"   [{name}] {list(positions.keys())}")
                
                # 检查时长
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n⏰ 已运行 {duration} 秒，自动停止")
                    break
                
                # 等待
                print(f"\n⏳ 等待下次扫描 ({self.scan_interval}秒)...")
                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")
        
        self._running = False
    
    def stop(self):
        """停止监控"""
        self._running = False
    
    def get_summary(self) -> Dict:
        """获取监控摘要"""
        entry_count = sum(1 for s in self.signal_history if s.signal_type == SignalType.ENTRY)
        exit_count = sum(1 for s in self.signal_history if s.signal_type == SignalType.EXIT)
        
        return {
            'total_signals': len(self.signal_history),
            'entry_signals': entry_count,
            'exit_signals': exit_count,
            'strategies': list(self.strategies.keys()),
            'current_positions': {
                name: list(pos.keys())
                for name, pos in self.virtual_positions.items()
            }
        }


def run_signal_monitor(
    strategies: List[str] = None,
    interval: int = 60,
    duration: int = None
):
    """
    快捷启动信号监控
    
    Args:
        strategies: 策略列表, None=全部
        interval: 扫描间隔 (秒)
        duration: 运行时长 (秒), None=永久
    
    Usage:
        # 监控所有策略
        run_signal_monitor()
        
        # 只监控短线策略
        run_signal_monitor(['short_term_rsrs'])
        
        # 监控5分钟
        run_signal_monitor(duration=300)
    """
    monitor = SignalMonitor(
        strategies=strategies,
        scan_interval=interval
    )
    monitor.run(duration=duration)


# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='信号监控')
    parser.add_argument('--strategy', '-s', nargs='+', default=None, help='策略名称')
    parser.add_argument('--interval', '-i', type=int, default=60, help='扫描间隔(秒)')
    parser.add_argument('--duration', '-d', type=int, default=None, help='运行时长(秒)')
    
    args = parser.parse_args()
    
    run_signal_monitor(
        strategies=args.strategy,
        interval=args.interval,
        duration=args.duration
    )