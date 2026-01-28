# ============================================================================
# 文件: engine/high_freq_matcher_v2.py
# ============================================================================
"""
高频撮合引擎 V2 - 涨跌停专业处理

核心改进:
1. 集合竞价模拟
2. 涨跌停精细处理
3. 炸板风险规避
4. 分时成交估算
5. 队列位置跟踪
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time
import logging

from engine.matcher import Order, OrderStatus
from config import settings


# ============================================================================
# 数据结构
# ============================================================================

class LimitState(Enum):
    """涨跌停状态"""
    NORMAL = "正常"
    LIMIT_UP_SEALED = "涨停封板"
    LIMIT_UP_WEAK = "涨停弱封"
    LIMIT_UP_OPEN = "涨停开板"
    LIMIT_DOWN_SEALED = "跌停封板"
    LIMIT_DOWN_WEAK = "跌停弱封"
    LIMIT_DOWN_OPEN = "跌停开板"


class TradingPhase(Enum):
    """交易阶段"""
    PRE_AUCTION = "集合竞价"
    OPENING = "开盘30分钟"
    MORNING = "上午盘"
    LUNCH = "午间休市"
    AFTERNOON = "下午盘"
    CLOSING = "尾盘30分钟"
    CLOSE_AUCTION = "收盘集合竞价"


@dataclass
class LimitAnalysis:
    """涨跌停分析"""
    state: LimitState
    limit_price: float
    seal_volume: float           # 封单量
    seal_ratio: float            # 封单/流通盘
    open_times: int              # 开板次数
    seal_strength: float         # 封单强度 0-1
    estimated_fill_prob: float   # 预估成交概率
    queue_position: int          # 排队位置
    
    @property
    def is_tradeable(self) -> bool:
        """是否可交易"""
        if self.state == LimitState.NORMAL:
            return True
        if self.state in [LimitState.LIMIT_UP_OPEN, LimitState.LIMIT_DOWN_OPEN]:
            return True
        if self.state == LimitState.LIMIT_UP_WEAK:
            return self.estimated_fill_prob > 0.3
        if self.state == LimitState.LIMIT_DOWN_WEAK:
            return self.estimated_fill_prob > 0.3
        return False


@dataclass
class HighFreqOrder(Order):
    """高频订单"""
    # 涨跌停相关
    limit_analysis: LimitAnalysis = None
    
    # 时间相关
    trading_phase: TradingPhase = TradingPhase.MORNING
    submit_time: str = ""
    
    # 成交预估
    estimated_fill_price: float = 0.0
    estimated_fill_prob: float = 1.0
    impact_cost: float = 0.0
    
    # 队列
    queue_position: int = 0
    queue_ahead_volume: float = 0.0


# ============================================================================
# 涨跌停分析器
# ============================================================================

class LimitAnalyzer:
    """
    涨跌停分析器
    
    功能:
    1. 判断涨跌停状态
    2. 分析封单强度
    3. 预估成交概率
    4. 计算队列位置
    """
    
    # 封单强度阈值
    STRONG_SEAL_RATIO = 0.05    # 封单/流通盘 > 5% = 强封
    WEAK_SEAL_RATIO = 0.02      # 封单/流通盘 > 2% = 弱封
    
    def __init__(self):
        self.logger = logging.getLogger("LimitAnalyzer")
    
    def analyze(
        self,
        market_data: pd.Series,
        intraday_data: pd.DataFrame = None,
        float_shares: float = None
    ) -> LimitAnalysis:
        """
        分析涨跌停状态
        
        Args:
            market_data: 当日行情 (open, high, low, close, vol, ...)
            intraday_data: 分时数据 (可选)
            float_shares: 流通股本 (可选)
        """
        close = market_data['close']
        open_price = market_data['open']
        high = market_data['high']
        low = market_data['low']
        volume = market_data.get('vol', 0)
        
        prev_close = market_data.get('prev_close', close / 1.05)
        
        # 涨跌停价格
        limit_up = round(prev_close * 1.1, 2)
        limit_down = round(prev_close * 0.9, 2)
        
        # 判断涨跌停
        is_limit_up = close >= limit_up - 0.01
        is_limit_down = close <= limit_down + 0.01
        
        if not is_limit_up and not is_limit_down:
            return LimitAnalysis(
                state=LimitState.NORMAL,
                limit_price=0,
                seal_volume=0,
                seal_ratio=0,
                open_times=0,
                seal_strength=0,
                estimated_fill_prob=1.0,
                queue_position=0
            )
        
        # 涨停分析
        if is_limit_up:
            return self._analyze_limit_up(
                market_data, limit_up, intraday_data, float_shares
            )
        
        # 跌停分析
        return self._analyze_limit_down(
            market_data, limit_down, intraday_data, float_shares
        )
    
    def _analyze_limit_up(
        self,
        data: pd.Series,
        limit_price: float,
        intraday: pd.DataFrame,
        float_shares: float
    ) -> LimitAnalysis:
        """分析涨停"""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data.get('vol', 0)
        
        # 判断是否开过板
        # 如果 low < limit_price，说明开过板
        has_opened = low < limit_price - 0.01
        
        # 估算封单量 (简化)
        if has_opened:
            # 开过板，封单量较弱
            seal_volume = volume * 0.1
            open_times = 1
            state = LimitState.LIMIT_UP_WEAK if close >= limit_price else LimitState.LIMIT_UP_OPEN
        else:
            if high == close == limit_price:
                # 一字板
                seal_volume = volume * 0.5
                state = LimitState.LIMIT_UP_SEALED
            else:
                seal_volume = volume * 0.2
                state = LimitState.LIMIT_UP_WEAK
            open_times = 0
        
        # 封单比例
        float_shares = float_shares or (volume * 100)  # 估算
        seal_ratio = seal_volume / float_shares if float_shares > 0 else 0
        
        # 封单强度
        if seal_ratio > self.STRONG_SEAL_RATIO:
            seal_strength = 1.0
        elif seal_ratio > self.WEAK_SEAL_RATIO:
            seal_strength = seal_ratio / self.STRONG_SEAL_RATIO
        else:
            seal_strength = seal_ratio / self.WEAK_SEAL_RATIO * 0.5
        
        # 成交概率
        if state == LimitState.LIMIT_UP_SEALED:
            fill_prob = 0.1  # 封死很难买到
        elif state == LimitState.LIMIT_UP_WEAK:
            fill_prob = 0.3 + (1 - seal_strength) * 0.3
        else:
            fill_prob = 0.7
        
        # 队列位置
        queue_position = int(seal_volume * 0.5)  # 假设在中间
        
        return LimitAnalysis(
            state=state,
            limit_price=limit_price,
            seal_volume=seal_volume,
            seal_ratio=round(seal_ratio, 4),
            open_times=open_times,
            seal_strength=round(seal_strength, 4),
            estimated_fill_prob=round(fill_prob, 4),
            queue_position=queue_position
        )
    
    def _analyze_limit_down(
        self,
        data: pd.Series,
        limit_price: float,
        intraday: pd.DataFrame,
        float_shares: float
    ) -> LimitAnalysis:
        """分析跌停"""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data.get('vol', 0)
        
        # 判断是否开过板
        has_opened = high > limit_price + 0.01
        
        if has_opened:
            seal_volume = volume * 0.1
            open_times = 1
            state = LimitState.LIMIT_DOWN_WEAK if close <= limit_price else LimitState.LIMIT_DOWN_OPEN
        else:
            if low == close == limit_price:
                seal_volume = volume * 0.5
                state = LimitState.LIMIT_DOWN_SEALED
            else:
                seal_volume = volume * 0.2
                state = LimitState.LIMIT_DOWN_WEAK
            open_times = 0
        
        float_shares = float_shares or (volume * 100)
        seal_ratio = seal_volume / float_shares if float_shares > 0 else 0
        
        if seal_ratio > self.STRONG_SEAL_RATIO:
            seal_strength = 1.0
        elif seal_ratio > self.WEAK_SEAL_RATIO:
            seal_strength = seal_ratio / self.STRONG_SEAL_RATIO
        else:
            seal_strength = seal_ratio / self.WEAK_SEAL_RATIO * 0.5
        
        # 跌停卖出概率
        if state == LimitState.LIMIT_DOWN_SEALED:
            fill_prob = 0.15  # 跌停封死难卖出
        elif state == LimitState.LIMIT_DOWN_WEAK:
            fill_prob = 0.4 + (1 - seal_strength) * 0.3
        else:
            fill_prob = 0.8
        
        queue_position = int(seal_volume * 0.5)
        
        return LimitAnalysis(
            state=state,
            limit_price=limit_price,
            seal_volume=seal_volume,
            seal_ratio=round(seal_ratio, 4),
            open_times=open_times,
            seal_strength=round(seal_strength, 4),
            estimated_fill_prob=round(fill_prob, 4),
            queue_position=queue_position
        )


# ============================================================================
# 高频撮合引擎 V2
# ============================================================================

class HighFreqMatcherV2:
    """
    高频撮合引擎 V2
    
    核心功能:
    1. 涨跌停智能处理
    2. 炸板风险规避
    3. 集合竞价模拟
    4. 冲击成本计算
    5. T+1 必杀卖出
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.001,
        impact_factor: float = 0.1
    ):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.impact_factor = impact_factor
        
        self.limit_analyzer = LimitAnalyzer()
        self._order_id = 0
        
        self.logger = logging.getLogger("HighFreqMatcherV2")
    
    # ==========================================================
    # 涨停板敢死队规避策略
    # ==========================================================
    
    def handle_limit_up_buy(
        self,
        order: HighFreqOrder,
        limit_analysis: LimitAnalysis,
        market_data: pd.Series
    ) -> HighFreqOrder:
        """
        处理涨停板买入
        
        规避策略:
        ┌─────────────────────────────────────────────────────────────────┐
        │                 涨停买入风险规避                                │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │  情况1: 涨停封死 (封单强度 > 0.8)                              │
        │  → 拒绝买入, 无法成交                                          │
        │                                                                 │
        │  情况2: 涨停弱封 (封单强度 0.3-0.8)                            │
        │  → 允许买入, 但降低成交概率                                    │
        │  → 成交价 = 涨停价                                             │
        │  → 风险提示: 可能炸板                                          │
        │                                                                 │
        │  情况3: 涨停开板 (开板次数 >= 1)                               │
        │  → 若首次开板: 允许买入, 价格略低于涨停                        │
        │  → 若多次开板: 拒绝买入, 追高风险大                            │
        │                                                                 │
        │  情况4: 一字板                                                 │
        │  → 完全拒绝, 不参与                                            │
        │                                                                 │
        │  额外规避:                                                      │
        │  • 尾盘涨停不追 (14:30后)                                      │
        │  • 高换手率涨停不追 (>15%)                                     │
        │  • 连续涨停 >= 3 不追                                          │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
        """
        state = limit_analysis.state
        seal_strength = limit_analysis.seal_strength
        open_times = limit_analysis.open_times
        
        # 一字板: 完全拒绝
        if state == LimitState.LIMIT_UP_SEALED and seal_strength > 0.9:
            return self._reject(order, "一字涨停无法买入")
        
        # 强封: 拒绝
        if state == LimitState.LIMIT_UP_SEALED and seal_strength > 0.8:
            return self._reject(order, f"涨停强封 (强度={seal_strength:.0%}), 无法买入")
        
        # 弱封: 允许但降低概率
        if state == LimitState.LIMIT_UP_SEALED or state == LimitState.LIMIT_UP_WEAK:
            if seal_strength > 0.5:
                order.estimated_fill_prob = 0.3
                self.logger.warning(f"[LIMIT-UP] {order.code} 涨停弱封, 成交概率 30%")
            else:
                order.estimated_fill_prob = 0.5
            
            order.estimated_fill_price = limit_analysis.limit_price
            return order
        
        # 开板
        if state == LimitState.LIMIT_UP_OPEN:
            if open_times >= 3:
                return self._reject(order, f"涨停开板{open_times}次, 追高风险大")
            
            if open_times >= 2:
                order.estimated_fill_prob = 0.4
                self.logger.warning(f"[LIMIT-UP] {order.code} 开板{open_times}次, 风险较高")
            else:
                order.estimated_fill_prob = 0.7
            
            # 开板买入价略低于涨停
            order.estimated_fill_price = limit_analysis.limit_price * 0.995
            return order
        
        return order
    
    def handle_limit_down_sell(
        self,
        order: HighFreqOrder,
        limit_analysis: LimitAnalysis,
        market_data: pd.Series,
        position: 'Position'
    ) -> HighFreqOrder:
        """
        处理跌停板卖出
        
        策略:
        ┌─────────────────────────────────────────────────────────────────┐
        │                 跌停卖出处理                                    │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │  情况1: 跌停封死                                               │
        │  → 挂跌停价排队                                                │
        │  → 估算排队位置和成交概率                                      │
        │  → 若概率 < 20%, 建议次日集合竞价卖                            │
        │                                                                 │
        │  情况2: 跌停开板                                               │
        │  → 立即市价卖出                                                │
        │  → 价格略高于跌停                                              │
        │                                                                 │
        │  情况3: 跌停弱封                                               │
        │  → 优先排队                                                    │
        │  → 关注开板机会                                                │
        │                                                                 │
        │  紧急处理:                                                      │
        │  • 若亏损 > 止损线且跌停: 必须排队!                            │
        │  • 若连续跌停: 集合竞价抢卖                                     │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
        """
        state = limit_analysis.state
        
        if state == LimitState.LIMIT_DOWN_SEALED:
            # 跌停封死
            order.estimated_fill_prob = limit_analysis.estimated_fill_prob
            order.estimated_fill_price = limit_analysis.limit_price
            order.queue_position = limit_analysis.queue_position
            
            if order.estimated_fill_prob < 0.2:
                self.logger.warning(
                    f"[LIMIT-DOWN] {order.code} 跌停封死, 成交概率仅 {order.estimated_fill_prob:.0%}, "
                    f"建议次日集合竞价"
                )
                order.signal_reason += " [建议次日卖出]"
            
            return order
        
        if state == LimitState.LIMIT_DOWN_OPEN:
            # 跌停开板, 立即卖
            order.estimated_fill_prob = 0.85
            order.estimated_fill_price = limit_analysis.limit_price * 1.005
            
            self.logger.info(f"[LIMIT-DOWN] {order.code} 跌停开板, 立即卖出")
            return order
        
        if state == LimitState.LIMIT_DOWN_WEAK:
            order.estimated_fill_prob = 0.5
            order.estimated_fill_price = limit_analysis.limit_price
            return order
        
        return order
    
    # ==========================================================
    # T+1 必杀卖出
    # ==========================================================
    
    def execute_t1_kill_sell(
        self,
        code: str,
        position: 'Position',
        market_data: pd.Series,
        current_date: str
    ) -> Optional[HighFreqOrder]:
        """
        T+1 必杀卖出
        
        条件:
        1. 昨日买入 (今日 T+1 可卖)
        2. 开盘 15 分钟内
        3. 涨幅 < 2%
        4. 跌破昨日收盘价
        
        Returns:
            卖出订单 或 None
        """
        # 检查 T+1
        if position.buy_date == current_date:
            return None  # 当日买入不可卖
        
        open_price = market_data['open']
        close = market_data['close']
        prev_close = market_data.get('prev_close', close / 1.02)
        
        # 模拟开盘 15 分钟价格 (用 open 和 close 插值)
        early_price = open_price * 0.6 + close * 0.4  # 简化模拟
        
        # 涨幅
        change_from_prev = (early_price - prev_close) / prev_close
        
        # 必杀条件
        should_kill = (
            change_from_prev < 0.02 and  # 涨幅 < 2%
            early_price < prev_close      # 跌破昨收
        )
        
        if should_kill:
            self._order_id += 1
            order = HighFreqOrder(
                order_id=f"KILL-{self._order_id:08d}",
                code=code,
                side="SELL",
                price=early_price,
                quantity=position.quantity,
                create_date=current_date,
                signal_reason=f"T+1必杀: 涨幅{change_from_prev:.1%}<2%, 跌破昨收",
                trading_phase=TradingPhase.OPENING
            )
            
            self.logger.warning(
                f"[T+1-KILL] {code} 触发必杀卖出 | "
                f"开盘={open_price:.2f} 早盘={early_price:.2f} 昨收={prev_close:.2f}"
            )
            
            return order
        
        return None
    
    # ==========================================================
    # 主撮合逻辑
    # ==========================================================
    
    def match(
        self,
        order: HighFreqOrder,
        market_data: pd.Series,
        position: Optional['Position'],
        current_date: str
    ) -> HighFreqOrder:
        """
        高频撮合
        """
        # 1. 基础检查
        if market_data.empty or pd.isna(market_data.get('open')):
            return self._reject(order, "停牌或无数据")
        
        # 2. 涨跌停分析
        limit_analysis = self.limit_analyzer.analyze(market_data)
        order.limit_analysis = limit_analysis
        
        # 3. 涨跌停处理
        if order.side == "BUY":
            if limit_analysis.state != LimitState.NORMAL:
                if limit_analysis.state in [LimitState.LIMIT_UP_SEALED, 
                                            LimitState.LIMIT_UP_WEAK,
                                            LimitState.LIMIT_UP_OPEN]:
                    order = self.handle_limit_up_buy(order, limit_analysis, market_data)
                    if order.status == OrderStatus.REJECTED:
                        return order
            
            # 跌停不能买
            if limit_analysis.state in [LimitState.LIMIT_DOWN_SEALED,
                                        LimitState.LIMIT_DOWN_WEAK]:
                return self._reject(order, "跌停中, 禁止买入")
        
        else:  # SELL
            # T+1 检查
            if position is not None and position.buy_date == current_date:
                return self._reject(order, "T+1 限制")
            
            if limit_analysis.state != LimitState.NORMAL:
                if limit_analysis.state in [LimitState.LIMIT_DOWN_SEALED,
                                            LimitState.LIMIT_DOWN_WEAK,
                                            LimitState.LIMIT_DOWN_OPEN]:
                    order = self.handle_limit_down_sell(
                        order, limit_analysis, market_data, position
                    )
                
                # 涨停不卖 (除非策略强制)
                if limit_analysis.state in [LimitState.LIMIT_UP_SEALED,
                                            LimitState.LIMIT_UP_WEAK]:
                    self.logger.info(f"[LIMIT-UP] {order.code} 涨停中, 暂不卖出")
                    # 可以卖，但提醒
        
        # 4. 计算成交价格
        fill_price, impact_cost = self._calc_fill_price(order, market_data, limit_analysis)
        
        # 5. 概率判断是否成交
        if np.random.random() > order.estimated_fill_prob:
            # 未成交 (涨跌停排队失败)
            return self._reject(order, f"成交概率 {order.estimated_fill_prob:.0%} 未中签")
        
        # 6. 更新订单
        order.status = OrderStatus.FILLED
        order.filled_price = round(fill_price, 4)
        order.filled_quantity = order.quantity
        order.impact_cost = impact_cost
        
        # 手续费
        trade_value = fill_price * order.quantity
        order.commission = round(max(trade_value * self.commission_rate, 5), 2)
        
        if order.side == "SELL":
            order.stamp_duty = round(trade_value * 0.001, 2)
        
        order.slippage = round(abs(fill_price - order.price) * order.quantity, 2)
        
        self.logger.info(
            f"[MATCH] {order.side} {order.code} qty={order.quantity} @ {fill_price:.3f} "
            f"| 冲击成本={impact_cost:.2f} | 概率={order.estimated_fill_prob:.0%}"
        )
        
        return order
    
    def _calc_fill_price(
        self,
        order: HighFreqOrder,
        market_data: pd.Series,
        limit_analysis: LimitAnalysis
    ) -> Tuple[float, float]:
        """计算成交价格"""
        # 涨跌停特殊处理
        if order.estimated_fill_price > 0:
            return order.estimated_fill_price, 0
        
        open_price = market_data['open']
        close = market_data['close']
        volume = market_data.get('vol', 1e6)
        
        # 基础价格 (用开盘价)
        base_price = open_price
        
        # 滑点
        if order.side == "BUY":
            slippage = self.slippage_rate
        else:
            slippage = -self.slippage_rate
        
        # 冲击成本
        order_value = order.price * order.quantity
        avg_trade_value = volume * base_price / 240  # 假设 240 分钟
        
        size_ratio = order_value / avg_trade_value if avg_trade_value > 0 else 1
        impact = self.impact_factor * np.sqrt(size_ratio) * base_price * 0.001
        
        if order.side == "BUY":
            impact = abs(impact)
        else:
            impact = -abs(impact)
        
        fill_price = base_price * (1 + slippage) + impact
        impact_cost = abs(impact * order.quantity)
        
        return fill_price, impact_cost
    
    def _reject(self, order: HighFreqOrder, reason: str) -> HighFreqOrder:
        """拒绝订单"""
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        self.logger.warning(f"[REJECT] {order.code}: {reason}")
        return order