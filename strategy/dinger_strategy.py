# ============================================================================
# 文件: strategy/dinger_strategy.py
# ============================================================================
"""
涨停板策略 - 涨停板预测和追踪

策略逻辑:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DingerStrategy                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【核心思想】 捕捉强势股的涨停机会，通过板块联动和成长性筛选                  │
│                                                                             │
│  【入场条件】 ALL 必须满足:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 涨停预测:                                                            │   │
│  │    - 当前涨幅 > 8% 且 < 9.8%                                           │   │
│  │    - 委买量 > 委卖量 × 2                                               │   │
│  │    - 封单金额 > 成交额 × 0.5                                           │   │
│  │                                                                         │   │
│  │ 2. 成长性筛选:                                                          │   │
│  │    - 营收增长率 > 20% (季报)                                          │   │
│  │    - 净利润增长率 > 15%                                                │   │
│  │    - ROE > 10%                                                         │   │
│  │                                                                         │   │
│  │ 3. 技术面确认:                                                          │   │
│  │    - 股价突破60日高点                                                   │   │
│  │    - 成交量 > 5日均值 × 1.5                                            │   │
│  │    - MACD金叉或RSRS > 0.8                                              │   │
│  │                                                                         │   │
│  │ 4. 板块联动:                                                            │   │
│  │    - 同板块已有1只以上涨停                                              │   │
│  │    - 板块指数当日涨幅 > 2%                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【离场条件】 ANY 触发:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 开板止损: 涨停板打开且下跌 > 2%                                    │   │
│  │ 2. 连板失败: 次日未能再次涨停                                          │   │
│  │ 3. 高位止盈: 连续3板后打开                                             │   │
│  │ 4. 趋势反转: 跌破5日均线                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【仓位管理】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 首板仓位: 25%                                                         │   │
│  │ 2-3板仓位: 35% (加仓)                                                 │   │
│  │ 4板及以上: 45% (再加速)                                               │   │
│  │ 单只最大: 50%                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【风险控制】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 避开ST、*ST股票                                                       │   │
│  │ 避开近期涨幅 > 50%的股票                                              │   │
│  │ 涨停时间过滤 (10:00前的不追)                                          │   │
│  │ 单日最大涨停板持仓: 2只                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, time

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from config import settings


@dataclass
class LimitUpRecord:
    """涨停记录"""
    code: str
    limit_up_date: str
    limit_up_price: float
    board_count: int = 1  # 连板数
    
    def is_next_day(self, current_date: str) -> bool:
        """判断是否是次日"""
        try:
            from datetime import datetime, timedelta
            dt = datetime.strptime(self.limit_up_date, '%Y-%m-%d')
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
            return (current_dt - dt).days == 1
        except:
            return False


@dataclass
class LimitUpPosition:
    """涨停板持仓"""
    code: str
    entry_date: str
    entry_price: float
    quantity: int
    consecutive_boards: int = 1  # 当前连板数
    highest_board: int = 1       # 最高连板数
    highest_price: float = field(default=0.0)
    
    def update_price(self, price: float):
        """更新最高价"""
        self.highest_price = max(self.highest_price, price)
    
    def get_pnl_pct(self, current_price: float) -> float:
        """获取盈亏百分比"""
        return (current_price - self.entry_price) / self.entry_price


@StrategyRegistry.register
class DingerStrategy(BaseStrategy):
    """
    涨停板策略
    
    特点:
    - 追逐强势涨停板
    - 板块联动分析
    - 成长性筛选
    - 连板操作
    """
    
    name = "dinger"
    version = "1.0.0"
    
    def __init__(self, params: Dict = None):
        default_params = {
            # 涨停预测参数
            'min_gain': 0.08,               # 最低涨幅 8%
            'max_gain': 0.098,             # 最高涨幅 9.8% (未涨停)
            'buy_sell_ratio': 2.0,         # 买卖委托比
            'seal_amount_ratio': 0.5,      # 封单金额比
            'min_seal_amount': 10000000,   # 最小封单金额 1000万
            
            # 成长性筛选
            'min_revenue_growth': 0.20,    # 最低营收增长率 20%
            'min_profit_growth': 0.15,     # 最低净利润增长率 15%
            'min_roe': 0.10,               # 最低ROE 10%
            
            # 技术面参数
            'breakout_period': 60,          # 突破周期
            'volume_multiplier': 1.5,       # 放量倍数
            'rsrs_threshold': 0.8,          # RSRS阈值
            
            # 板块联动
            'min_board_limit_ups': 1,      # 最小板块涨停数
            'min_sector_gain': 0.02,       # 最小板块涨幅 2%
            
            # 风控参数
            'avoid_st': True,               # 避开ST
            'max_recent_gain': 0.50,        # 最大近期涨幅 50%
            'min_limit_up_time': time(10, 0),  # 最早涨停时间
            'max_position_count': 2,        # 最大涨停板持仓数
            
            # 仓位参数
            'position_first_board': 0.25,  # 首板仓位 25%
            'position_2_3_board': 0.35,    # 2-3板仓位 35%
            'position_4_plus_board': 0.45, # 4板+仓位 45%
            'max_single_position': 0.50,   # 单只最大仓位 50%
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
        
        # 持仓管理
        self._positions: Dict[str, LimitUpPosition] = {}
        
        # 涨停记录 (用于板块联动分析)
        self._limit_up_records: List[LimitUpRecord] = []
        self._today_limit_ups: Dict[str, LimitUpRecord] = {}
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算因子"""
        breakout_period = self.get_param('breakout_period')
        
        # 获取因子
        rsrs_factor = FactorRegistry.get("rsrs_slope")
        momentum_factor = FactorRegistry.get("momentum")
        atr_factor = FactorRegistry.get("atr_pct")
        
        factors = {
            'rsrs': rsrs_factor.compute_batch(history),
            'momentum': momentum_factor.compute_batch(history),
            'atr_pct': atr_factor.compute_batch(history),
        }
        
        # 手动计算技术指标和涨停相关因子
        technical_factors = self._compute_technical_indicators(history, breakout_period)
        factors.update(technical_factors)
        
        self.logger.info(f"Computed dinger factors for {len(history)} stocks")
        
        return factors
    
    def _compute_technical_indicators(self, history: Dict[str, pd.DataFrame],
                                     breakout_period: int) -> Dict[str, pd.DataFrame]:
        """计算技术指标"""
        recent_gain_period = 20
        ma5 = 5
        
        ma5_dict = {}
        highest_60_dict = {}
        recent_gain_dict = {}
        wash_signal_dict = {}
        
        for code, df in history.items():
            if len(df) < breakout_period:
                continue
            
            df = df.copy()
            
            # 计算MA5
            df['ma5'] = df['close'].rolling(ma5, min_periods=1).mean()
            
            # 计算最高价
            df['highest_60'] = df['high'].rolling(breakout_period, min_periods=1).max()
            
            # 计算近期涨幅
            df['recent_gain'] = df['close'].pct_change(recent_gain_period)
            
            # 洗盘信号: 放量阴线
            df['avg_volume'] = df['vol'].rolling(20, min_periods=1).mean()
            df['daily_change'] = df['close'].pct_change()
            df['wash_signal'] = (
                (df['daily_change'] < -0.05) &
                (df['vol'] > df['avg_volume'] * 2)
            ).astype(float)
            
            ma5_dict[code] = df['ma5']
            highest_60_dict[code] = df['highest_60']
            recent_gain_dict[code] = df['recent_gain']
            wash_signal_dict[code] = df['wash_signal']
        
        # 转换为DataFrame
        all_dates = set()
        for s in ma5_dict.values():
            all_dates.update(s.index)
        for s in highest_60_dict.values():
            all_dates.update(s.index)
        for s in recent_gain_dict.values():
            all_dates.update(s.index)
        for s in wash_signal_dict.values():
            all_dates.update(s.index)
        
        all_dates = sorted(list(all_dates))
        
        ma5_df = pd.DataFrame(index=all_dates)
        highest_60_df = pd.DataFrame(index=all_dates)
        recent_gain_df = pd.DataFrame(index=all_dates)
        wash_signal_df = pd.DataFrame(index=all_dates)
        
        for code, s in ma5_dict.items():
            ma5_df[code] = s
        
        for code, s in highest_60_dict.items():
            highest_60_df[code] = s
        
        for code, s in recent_gain_dict.items():
            recent_gain_df[code] = s
        
        for code, s in wash_signal_dict.items():
            wash_signal_df[code] = s
        
        return {
            'ma5': ma5_df,
            'highest_60': highest_60_df,
            'recent_gain_20': recent_gain_df,
            'wash_signal': wash_signal_df,
        }
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        
        df = context.current_data.copy()
        
        # 计算当日涨幅
        df['daily_gain'] = df.groupby('code')['close'].pct_change().fillna(0)
        
        # 过滤ST股票
        if self.get_param('avoid_st'):
            df = df[~df['code'].str.contains('ST')]
        
        # 过滤近期涨幅过大的股票
        df['recent_gain'] = df['code'].apply(
            lambda x: context.get_factor('recent_gain_20', x) or 0
        )
        df = df[df['recent_gain'] <= self.get_param('max_recent_gain')]
        
        # 获取因子值
        df['ma5'] = df['code'].apply(lambda x: context.get_factor('ma5', x))
        df['highest_60'] = df['code'].apply(lambda x: context.get_factor('highest_60', x))
        df['rsrs'] = df['code'].apply(lambda x: context.get_factor('rsrs', x))
        df['atr_pct'] = df['code'].apply(lambda x: context.get_factor('atr_pct', x))
        df['momentum'] = df['code'].apply(lambda x: context.get_factor('momentum', x))
        
        # 计算成交量比率 (使用当日成交量与平均成交量的比率)
        df['avg_volume'] = df.groupby('code')['vol'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['volume_ratio'] = df['vol'] / df['avg_volume']
        
        # 过滤有效数据
        valid_mask = (
            df['ma5'].notna() &
            df['highest_60'].notna() &
            df['volume_ratio'].notna()
        )
        
        df = df[valid_mask].copy()
        
        if df.empty:
            return signals
        
        # ========== 检查现有持仓 ==========
        for code in list(self._positions.keys()):
            pos = self._positions[code]
            
            if code not in df['code'].values:
                # 平仓
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason="股票不在交易列表"
                ))
                del self._positions[code]
                continue
            
            row = df[df['code'] == code].iloc[0]
            current_price = row['close']
            daily_gain = row['daily_gain']
            
            pos.update_price(current_price)
            
            # 检查连板
            if daily_gain >= 0.099:  # 涨停
                pos.consecutive_boards += 1
                pos.highest_board = max(pos.highest_board, pos.consecutive_boards)
            
            # 检查离场条件
            exit_signal = self._check_exit(pos, current_price, daily_gain, row)
            if exit_signal:
                signals.append(exit_signal)
                del self._positions[code]
        
        # ========== 检查当日涨停板 ==========
        limit_up_mask = (
            (df['daily_gain'] >= self.get_param('min_gain')) &
            (df['daily_gain'] <= self.get_param('max_gain'))
        )
        
        limit_up_stocks = df[limit_up_mask]
        self._today_limit_ups = {
            row['code']: LimitUpRecord(
                code=row['code'],
                limit_up_date=context.current_date,
                limit_up_price=row['close']
            )
            for _, row in limit_up_stocks.iterrows()
        }
        
        # ========== 生成新信号 ==========
        
        # 检查最大持仓数
        if len(self._positions) >= self.get_param('max_position_count'):
            return signals
        
        # 涨停预测候选筛选
        candidates_mask = (
            (df['daily_gain'] >= self.get_param('min_gain')) &
            (df['daily_gain'] <= self.get_param('max_gain')) &
            (df['volume_ratio'] >= self.get_param('volume_multiplier')) &
            (df['rsrs'] >= self.get_param('rsrs_threshold'))
        )
        
        candidates = df[candidates_mask]
        
        if not candidates.empty:
            # 按涨幅和成交量排序
            candidates['score'] = candidates['daily_gain'] * candidates['vol']
            top_candidates = candidates.nlargest(5, 'score')
            
            for _, row in top_candidates.iterrows():
                code = row['code']
                
                # 检查是否已持仓
                if code in self._positions:
                    # 可能是加仓 (连板)
                    pos = self._positions[code]
                    if pos.consecutive_boards >= 2 and pos.consecutive_boards <= 3:
                        # 2-3板加仓
                        weight = self.get_param('position_2_3_board')
                        signals.append(Signal(
                            code=code,
                            side=OrderSide.BUY,
                            weight=weight,
                            price=row['close'],
                            reason=f"连板{pos.consecutive_boards}板加仓"
                        ))
                    continue
                
                # 新建仓
                weight = self.get_param('position_first_board')
                self._positions[code] = LimitUpPosition(
                    code=code,
                    entry_date=context.current_date,
                    entry_price=row['close'],
                    quantity=0,
                    consecutive_boards=1,
                    highest_price=row['close']
                )
                
                signals.append(Signal(
                    code=code,
                    side=OrderSide.BUY,
                    weight=weight,
                    price=row['close'],
                    reason=f"首板涨停 涨幅={row['daily_gain']:.2%}"
                ))
                
                if len(self._positions) >= self.get_param('max_position_count'):
                    break
        
        return signals
    
    def _check_exit(self, pos: LimitUpPosition, current_price: float, 
                     daily_gain: float, row: pd.Series) -> Optional[Signal]:
        """检查是否需要离场"""
        reason = None
        
        # 开板止损: 涨停板打开且下跌 > 2%
        if pos.consecutive_boards >= 1 and daily_gain < -0.02:
            reason = f"开板止损 下跌{daily_gain:.2%}"
        
        # 连板失败: 次日未能再次涨停
        elif pos.consecutive_boards >= 2 and daily_gain < 0.09:
            reason = f"连板失败 涨幅{daily_gain:.2%}"
        
        # 高位止盈: 连续3板后打开
        elif pos.highest_board >= 3 and daily_gain < 0.09:
            reason = f"高位止盈 最高{pos.highest_board}板"
        
        # 趋势反转: 跌破5日均线
        elif current_price < row['ma5']:
            reason = f"趋势反转 跌破MA5"
        
        if reason:
            return Signal(
                code=pos.code,
                side=OrderSide.SELL,
                weight=0.0,
                reason=reason
            )
        
        return None
    
    def on_order_filled(self, order: 'Order') -> None:
        """订单成交回调"""
        # 可以在这里添加持仓数量更新逻辑
        pass
    
    def on_day_end(self, context: StrategyContext) -> None:
        """日终回调 - 更新涨停记录"""
        # 将当日涨停记录保存到历史记录
        self._limit_up_records.extend(list(self._today_limit_ups.values()))
        self._today_limit_ups.clear()
        
        # 清理超过30天的记录
        cutoff_date = (pd.Timestamp(context.current_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        self._limit_up_records = [
            r for r in self._limit_up_records if r.limit_up_date >= cutoff_date
        ]
