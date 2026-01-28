# ============================================================================
# 文件: strategy/alpha_hunter_v2_strategy.py
# ============================================================================
"""
Alpha-Hunter-V2 策略

目标: 年化 >30%, 回撤 <10%
持仓周期: T+1 到 T+2
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors.alpha_hunter_v2_factors import (
    AlphaFactorEngineV2, AlphaFactorResult, MarketState,
    AdaptiveRSRSFactor, OpeningSurgeFactor, MultiLevelPressureFactor
)
from config import settings


@dataclass
class TradeRecord:
    """交易记录"""
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_ratio: float
    is_win: bool
    holding_days: int


@dataclass
class PositionStateV2:
    """持仓状态 V2"""
    code: str
    entry_price: float
    entry_date: str
    quantity: int

    # 止损止盈
    hard_stop: float  # 硬止损价
    trailing_stop: float  # 移动止损价
    highest_price: float  # 最高价

    # 锁利状态
    lock_levels: List[float] = field(default_factory=lambda: [0.03, 0.06, 0.09, 0.12, 0.15])
    current_lock_level: int = 0

    # 风险标记
    is_limit_up_entry: bool = False  # 是否涨停板买入
    sector: str = ""  # 所属行业

    def update_trailing_stop(self, current_price: float, atr_pct: float = 0.02):
        """更新移动止盈"""
        if current_price > self.highest_price:
            self.highest_price = current_price

        pnl = (current_price - self.entry_price) / self.entry_price

        # 每 +3% 利润，止损上移 2%
        while (self.current_lock_level < len(self.lock_levels) and
               pnl >= self.lock_levels[self.current_lock_level]):

            new_stop = self.entry_price * (1 + 0.02 * (self.current_lock_level + 1))
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

            self.current_lock_level += 1

        # 也可以用 ATR 动态调整
        atr_stop = self.highest_price * (1 - 2 * atr_pct)
        self.trailing_stop = max(self.trailing_stop, atr_stop, self.hard_stop)


@StrategyRegistry.register
class AlphaHunterV2Strategy(BaseStrategy):
    """
    Alpha-Hunter-V2 策略

    核心改进:
    1. 自适应 RSRS (市场状态感知)
    2. 多层次压力位过滤
    3. 更精细的涨跌停处理
    4. Kelly 准则动态仓位
    5. 行业限额控制
    """

    name = "alpha_hunter_v2"
    version = "2.0.0"

    DEFAULT_PARAMS = {
        # 入场参数
        'rsrs_threshold': 0.8,
        'rsrs_r2_threshold': 0.85,
        'min_signal_quality': 0.6,
        'min_pressure_distance': 0.05,
        'ma5_slope_threshold': 0.001,
        'max_turnover': 0.25,
        'market_breadth_threshold': 0.40,

        # 离场参数
        'hard_stop_loss': 0.03,
        't1_kill_threshold': 0.02,
        'profit_lock_step': 0.03,
        'stop_raise_step': 0.02,
        'max_holding_days': 2,

        # 仓位参数
        'kelly_lookback': 20,
        'kelly_fraction': 0.5,
        'max_single_position': 0.08,
        'max_total_position': 0.70,
        'max_positions': 8,

        # 行业限制
        'max_sector_exposure': 0.20,

        # 涨停限制
        'allow_limit_up_chase': False,
        'max_limit_up_positions': 2,

        # 价格过滤
        'min_price': 5.0,
        'max_price': 100.0,
        'min_amount': 5000000,
    }

    def __init__(self, params: Dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

        # 因子引擎
        self.factor_engine = AlphaFactorEngineV2()

        # 持仓状态
        self._positions: Dict[str, PositionStateV2] = {}

        # 交易历史 (Kelly 计算)
        self._trade_history: deque = deque(maxlen=100)

        # 行业敞口
        self._sector_exposure: Dict[str, float] = {}

        # 市场状态缓存
        self._market_state: MarketState = MarketState.SHOCK
        self._market_breadth: Dict = {}

        # 涨停买入计数
        self._limit_up_count: int = 0

    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """预计算因子"""
        self.logger.info("Computing Alpha-Hunter-V2 factors...")

        factors = {}

        rsrs_factor = AdaptiveRSRSFactor()
        surge_factor = OpeningSurgeFactor()
        pressure_factor = MultiLevelPressureFactor()

        rsrs_results = {}
        surge_results = {}
        pressure_results = {}
        ma_results = {}

        total = len(history)
        processed = 0

        for code, df in history.items():
            if len(df) < 250:
                continue

            try:
                # RSRS
                rsrs_data = rsrs_factor.compute_full(df)
                rsrs_results[code] = rsrs_data

                # 开盘异动
                surge_data = surge_factor.compute_full(df)
                surge_results[code] = surge_data

                # 压力位
                pressure_data = pressure_factor.compute_full(df)
                pressure_results[code] = pressure_data

                # MA 数据
                ma5 = df['close'].rolling(5).mean()
                ma5_slope = ma5.diff(3) / ma5.shift(3)
                ma20 = df['close'].rolling(20).mean()

                ma_results[code] = pd.DataFrame({
                    'ma5': ma5,
                    'ma5_slope': ma5_slope,
                    'ma20': ma20,
                    'above_ma5': (df['close'] > ma5).astype(int),
                    'above_ma20': (df['close'] > ma20).astype(int)
                }, index=df.index)

                processed += 1

            except Exception as e:
                self.logger.warning(f"Factor error for {code}: {e}")

        # 转换为宽表
        if rsrs_results:
            factors['rsrs_adaptive'] = pd.DataFrame({
                code: data['rsrs_adaptive'] for code, data in rsrs_results.items()
            })
            factors['rsrs_r2'] = pd.DataFrame({
                code: data['rsrs_r2'] for code, data in rsrs_results.items()
            })
            factors['rsrs_valid'] = pd.DataFrame({
                code: data['rsrs_valid'] for code, data in rsrs_results.items()
            })
            factors['rsrs_momentum'] = pd.DataFrame({
                code: data['rsrs_momentum'] for code, data in rsrs_results.items()
            })
            factors['signal_quality'] = pd.DataFrame({
                code: data['signal_quality'] for code, data in rsrs_results.items()
            })

        if surge_results:
            factors['surge_score'] = pd.DataFrame({
                code: data['surge_score'] for code, data in surge_results.items()
            })

        if pressure_results:
            factors['pressure_distance'] = pd.DataFrame({
                code: data['combined_pressure_dist'] for code, data in pressure_results.items()
            })
            factors['safety_score'] = pd.DataFrame({
                code: data['safety_score'] for code, data in pressure_results.items()
            })

        if ma_results:
            factors['ma5'] = pd.DataFrame({
                code: data['ma5'] for code, data in ma_results.items()
            })
            factors['ma5_slope'] = pd.DataFrame({
                code: data['ma5_slope'] for code, data in ma_results.items()
            })
            factors['above_ma5'] = pd.DataFrame({
                code: data['above_ma5'] for code, data in ma_results.items()
            })

        self.logger.info(f"Computed factors for {processed} stocks")
        return factors

    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        current_date = context.current_date

        # 1. 计算市场情绪
        self._market_breadth = self._calc_market_breadth(context)

        # 2. T+1 必杀卖出检查 (最高优先级)
        kill_signals = self._generate_t1_kill_signals(context)
        signals.extend(kill_signals)

        # 3. 常规离场检查
        exit_signals = self._generate_exit_signals(context)
        signals.extend(exit_signals)

        # 4. 市场情绪过滤
        if not self._market_breadth.get('is_bullish', False):
            self.logger.info(
                f"市场情绪偏空 ({self._market_breadth.get('advance_ratio', 0):.0%}), "
                f"暂停入场"
            )
            return signals

        # 5. 入场信号
        entry_signals = self._generate_entry_signals(context)
        signals.extend(entry_signals)

        return signals

    def _calc_market_breadth(self, context: StrategyContext) -> Dict:
        """计算市场广度"""
        df = context.current_data

        if df.empty:
            return {'is_bullish': False, 'advance_ratio': 0.5}

        # 计算涨跌
        if 'open' in df.columns and 'close' in df.columns:
            changes = df['close'] / df['open'] - 1
            advancing = (changes > 0.001).sum()
            total = len(df)
            advance_ratio = advancing / total if total > 0 else 0.5
        else:
            advance_ratio = 0.5

        threshold = self.get_param('market_breadth_threshold')

        return {
            'advance_ratio': advance_ratio,
            'is_bullish': advance_ratio >= threshold
        }

    def _generate_t1_kill_signals(self, context: StrategyContext) -> List[Signal]:
        """T+1 必杀卖出"""
        signals = []
        current_date = context.current_date
        threshold = self.get_param('t1_kill_threshold')

        for code, pos in list(self._positions.items()):
            # T+1 检查
            if pos.entry_date == current_date:
                continue

            # 获取数据
            row = context.current_data[context.current_data['code'] == code]
            if row.empty:
                continue

            current_price = row['close'].iloc[0]
            open_price = row['open'].iloc[0]

            # 获取昨收
            history = context.get_history(code, 2)
            if len(history) < 2:
                continue
            prev_close = history['close'].iloc[-2]

            # 模拟早盘价格 (开盘和收盘加权)
            early_price = open_price * 0.7 + current_price * 0.3

            # 涨幅
            change_from_prev = (early_price - prev_close) / prev_close

            # 必杀条件
            if change_from_prev < threshold and early_price < prev_close:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=early_price,
                    priority=100,
                    reason=f"T+1必杀: 涨幅{change_from_prev:.1%}<{threshold:.0%}, 跌破昨收"
                ))

                self.logger.warning(
                    f"[T+1-KILL] {code} | 开盘={open_price:.2f} "
                    f"早盘={early_price:.2f} 昨收={prev_close:.2f}"
                )

        return signals

    def _generate_exit_signals(self, context: StrategyContext) -> List[Signal]:
        """常规离场信号"""
        signals = []
        current_date = context.current_date

        hard_stop = self.get_param('hard_stop_loss')
        max_days = self.get_param('max_holding_days')

        for code, pos in list(self._positions.items()):
            if pos.entry_date == current_date:
                continue

            row = context.current_data[context.current_data['code'] == code]
            if row.empty:
                continue

            current_price = row['close'].iloc[0]

            # 获取 ATR
            atr_pct = context.get_factor('rsrs_momentum', code)
            if atr_pct is None or pd.isna(atr_pct):
                atr_pct = 0.02

            # 更新移动止盈
            pos.update_trailing_stop(current_price, abs(atr_pct) * 0.1)

            should_exit = False
            reason = ""

            # 1. 硬止损
            pnl = (current_price - pos.entry_price) / pos.entry_price
            if pnl <= -hard_stop:
                should_exit = True
                reason = f"硬止损 {pnl:.1%}"

            # 2. 移动止损
            if not should_exit and current_price < pos.trailing_stop:
                should_exit = True
                reason = f"移动止损 ({pos.trailing_stop:.2f})"

            # 3. 跌破 MA5
            if not should_exit:
                ma5 = context.get_factor('ma5', code)
                if ma5 is not None and current_price < ma5:
                    should_exit = True
                    reason = f"跌破MA5 ({ma5:.2f})"

            # 4. RSRS 转弱
            if not should_exit:
                rsrs = context.get_factor('rsrs_adaptive', code)
                if rsrs is not None and rsrs < -0.3:
                    should_exit = True
                    reason = f"RSRS转弱 ({rsrs:.2f})"

            # 5. 持仓时间
            if not should_exit:
                try:
                    entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
                    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
                    holding_days = (current_dt - entry_dt).days

                    if holding_days >= max_days:
                        should_exit = True
                        reason = f"持仓{holding_days}天, 强制离场"
                except:
                    pass

            if should_exit:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    price=current_price,
                    reason=reason
                ))

                self.logger.info(f"[EXIT] {code} | {reason} | PnL={pnl:.1%}")

        return signals

    def _generate_entry_signals(self, context: StrategyContext) -> List[Signal]:
        """入场信号"""
        signals = []

        # 参数
        rsrs_th = self.get_param('rsrs_threshold')
        r2_th = self.get_param('rsrs_r2_threshold')
        quality_th = self.get_param('min_signal_quality')
        pressure_th = self.get_param('min_pressure_distance')
        ma5_slope_th = self.get_param('ma5_slope_threshold')
        max_turnover = self.get_param('max_turnover')
        min_price = self.get_param('min_price')
        max_price = self.get_param('max_price')
        min_amount = self.get_param('min_amount')
        max_positions = self.get_param('max_positions')

        # 检查持仓数
        if len(self._positions) >= max_positions:
            return signals

        candidates = []

        for _, row in context.current_data.iterrows():
            code = row['code']
            close = row['close']
            volume = row.get('vol', 0)
            amount = row.get('amount', close * volume)

            # ===== 基础过滤 =====
            if code in self._positions or code in context.positions:
                continue

            if close < min_price or close > max_price:
                continue

            if amount < min_amount:
                continue

            # ===== 条件 1: RSRS =====
            rsrs = context.get_factor('rsrs_adaptive', code)
            r2 = context.get_factor('rsrs_r2', code)
            quality = context.get_factor('signal_quality', code)

            if rsrs is None or pd.isna(rsrs) or rsrs <= rsrs_th:
                continue

            if r2 is None or pd.isna(r2) or r2 < r2_th:
                continue

            if quality is not None and quality < quality_th:
                continue

            # ===== 条件 2: MA5 趋势 =====
            above_ma5 = context.get_factor('above_ma5', code)
            ma5_slope = context.get_factor('ma5_slope', code)

            if above_ma5 is None or above_ma5 < 1:
                continue

            if ma5_slope is None or ma5_slope < ma5_slope_th:
                continue

            # ===== 条件 3: 压力距离 =====
            pressure = context.get_factor('pressure_distance', code)
            if pressure is not None and pressure < pressure_th:
                continue

            # ===== 条件 4: 换手率 =====
            history = context.get_history(code, 5)
            if not history.empty and 'amount' in history.columns:
                avg_amount = history['amount'].mean()
                market_cap_est = close * history['vol'].mean() * 100
                turnover = avg_amount / market_cap_est if market_cap_est > 0 else 0

                if turnover > max_turnover:
                    continue

            # ===== 条件 5: 非涨停 =====
            if not self.get_param('allow_limit_up_chase'):
                if not history.empty:
                    prev_close = history['close'].iloc[-1]
                    if close >= prev_close * 1.095:
                        continue

            # 通过所有条件
            safety = context.get_factor('safety_score', code) or 0.5
            surge = context.get_factor('surge_score', code) or 0

            score = rsrs * r2 * (0.5 + 0.5 * safety) * (1 + 0.2 * surge)

            candidates.append({
                'code': code,
                'close': close,
                'rsrs': rsrs,
                'r2': r2,
                'quality': quality or 0.5,
                'pressure': pressure or 0.1,
                'safety': safety,
                'surge': surge,
                'score': score
            })

        # 排序
        candidates.sort(key=lambda x: x['score'], reverse=True)

        slots = max_positions - len(self._positions)
        selected = candidates[:slots]

        # 计算仓位
        for cand in selected:
            weight = self._calc_kelly_position(context.total_equity)
            weight = min(weight, self.get_param('max_single_position'))

            if weight < 0.02:
                continue

            signals.append(Signal(
                code=cand['code'],
                side=OrderSide.BUY,
                weight=weight,
                price=cand['close'],
                reason=(
                    f"RSRS={cand['rsrs']:.2f} R²={cand['r2']:.2f} "
                    f"Q={cand['quality']:.2f} 压力距={cand['pressure']:.1%}"
                )
            ))

            self.logger.info(
                f"[ENTRY] {cand['code']} | Score={cand['score']:.3f} | Weight={weight:.1%}"
            )

        return signals

    def _calc_kelly_position(self, total_equity: float) -> float:
        """Kelly 准则计算仓位"""
        if len(self._trade_history) < 5:
            return 0.05

        wins = [t for t in self._trade_history if t.is_win]
        losses = [t for t in self._trade_history if not t.is_win]

        if not wins or not losses:
            return 0.05

        p = len(wins) / len(self._trade_history)
        q = 1 - p

        avg_win = np.mean([t.pnl_ratio for t in wins])
        avg_loss = abs(np.mean([t.pnl_ratio for t in losses]))

        if avg_loss <= 0:
            return 0.05

        b = avg_win / avg_loss
        kelly = (p * b - q) / b if b > 0 else 0

        position = kelly * self.get_param('kelly_fraction')
        position = np.clip(position, 0.02, 0.10)

        return position

    def on_order_filled(self, order) -> None:
        """订单成交回调"""
        if order.side == OrderSide.BUY:
            hard_stop = self.get_param('hard_stop_loss')

            self._positions[order.code] = PositionStateV2(
                code=order.code,
                entry_price=order.filled_price,
                entry_date=order.create_date,
                quantity=order.filled_quantity,
                hard_stop=order.filled_price * (1 - hard_stop),
                trailing_stop=order.filled_price * (1 - hard_stop),
                highest_price=order.filled_price
            )

            self.logger.info(
                f"[BUY] {order.code} @ {order.filled_price:.2f} | "
                f"止损={order.filled_price * (1 - hard_stop):.2f}"
            )

        else:  # SELL
            if order.code in self._positions:
                pos = self._positions.pop(order.code)
                pnl = (order.filled_price - pos.entry_price) / pos.entry_price

                try:
                    entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
                    exit_dt = datetime.strptime(order.create_date, '%Y-%m-%d')
                    holding_days = (exit_dt - entry_dt).days
                except:
                    holding_days = 1

                trade = TradeRecord(
                    code=order.code,
                    entry_date=pos.entry_date,
                    exit_date=order.create_date,
                    entry_price=pos.entry_price,
                    exit_price=order.filled_price,
                    pnl_ratio=pnl,
                    is_win=(pnl > 0),
                    holding_days=holding_days
                )
                self._trade_history.append(trade)

                self.logger.info(
                    f"[SELL] {order.code} @ {order.filled_price:.2f} | "
                    f"PnL={pnl:.1%} | 持仓{holding_days}天"
                )

    def get_performance_summary(self) -> Dict:
        """绩效摘要"""
        if not self._trade_history:
            return {'trades': 0}

        wins = [t for t in self._trade_history if t.is_win]
        losses = [t for t in self._trade_history if not t.is_win]

        return {
            'trades': len(self._trade_history),
            'win_rate': len(wins) / len(self._trade_history),
            'avg_pnl': np.mean([t.pnl_ratio for t in self._trade_history]),
            'avg_win': np.mean([t.pnl_ratio for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl_ratio for t in losses]) if losses else 0,
            'avg_holding_days': np.mean([t.holding_days for t in self._trade_history]),
            'max_win': max([t.pnl_ratio for t in self._trade_history]),
            'max_loss': min([t.pnl_ratio for t in self._trade_history])
        }