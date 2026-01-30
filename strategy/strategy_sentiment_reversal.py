# ============================================================================
# 文件: strategy/sentiment_reversal.py
# ============================================================================
"""
情绪反转捕捉策略 - 集成版

目标:
- Sharpe Ratio: 3.2-3.9
- Win Rate: 67-73%
- 持仓周期: 2-5天

核心逻辑:
- 识别恐慌性抛售 (连续大跌+放量)
- 反向建仓
- 快速兑现 (2-5天)

典型场景:
- 2020-03 COVID暴跌后反弹
- 2022-04 恐慌性抛售修复
- 市场极端情绪修复

集成方式:
from strategy.sentiment_reversal import SentimentReversalStrategy
strategy = SentimentReversalStrategy()
engine.add_strategy(strategy)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from strategy.base import BaseStrategy, StrategyContext, Signal, OrderSide
from engine.risk import RiskManager
from config import settings


@dataclass
class SentimentConfig:
    """情绪反转策略配置"""
    # 恐慌检测参数
    extreme_loss_threshold: float = -0.15   # 累计跌幅阈值 15%
    consecutive_days: int = 3               # 连续下跌天数
    volume_surge_ratio: float = 2.0         # 成交量放大倍数
    final_day_drop: float = -0.05           # 最后一天跌幅 5%
    
    # RSI参数
    rsi_period: int = 14
    rsi_oversold: float = 30                # RSI超卖阈值
    
    # FOMO检测 (追涨风险)
    fomo_rise_threshold: float = 0.20       # 累计涨幅阈值 20%
    fomo_rsi_threshold: float = 80          # RSI超买阈值
    
    # 选股参数
    top_n: int = 12                         # 持仓数量
    panic_score_threshold: float = 0.6      # 恐慌分数阈值
    
    # 持仓管理
    max_holding_days: int = 5               # 最长持有期
    min_holding_days: int = 1               # 最短持有期
    
    # 风控参数
    stop_loss: float = -0.06                # 止损6%
    take_profit: float = 0.10               # 止盈10%
    min_profit_target: float = 0.03         # 最小获利目标3%


class SentimentReversalStrategy(BaseStrategy):
    """情绪反转捕捉策略"""
    
    def __init__(self, config: SentimentConfig = None):
        super().__init__(name="sentiment_reversal")
        self.config = config or SentimentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 风险管理
        self.risk_manager = RiskManager(
            max_position_weight=0.12,  # 单股最大12%
            max_total_position=0.90
        )
        
        # 持仓跟踪
        self.entry_prices: Dict[str, float] = {}
        self.entry_dates: Dict[str, str] = {}
        self.holding_days: Dict[str, int] = {}
        
        # 情绪指标缓存
        self._panic_scores: pd.Series = pd.Series()
        self._fomo_scores: pd.Series = pd.Series()
    
    def initialize(self) -> None:
        """初始化策略"""
        self.logger.info(f"Initializing {self.name}")
        self.logger.info(f"Config: extreme_loss={self.config.extreme_loss_threshold:.0%}, "
                        f"consecutive_days={self.config.consecutive_days}, "
                        f"top_n={self.config.top_n}")
    
    def compute_factors(self, history_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        计算情绪因子
        
        Args:
            history_data: {code: DataFrame}
        
        Returns:
            {factor_name: Series(code → score)}
        """
        self.logger.info(f"Computing sentiment factors for {len(history_data)} stocks...")
        
        panic_scores = {}
        fomo_scores = {}
        
        for code, df in history_data.items():
            if len(df) < 30:
                continue
            
            try:
                # 1. 恐慌性抛售分数
                panic_scores[code] = self._compute_panic_score(df)
                
                # 2. FOMO追涨分数 (风险指标)
                fomo_scores[code] = self._compute_fomo_score(df)
                
            except Exception as e:
                self.logger.debug(f"Factor computation failed for {code}: {e}")
                continue
        
        # 转为Series
        self._panic_scores = pd.Series(panic_scores)
        self._fomo_scores = pd.Series(fomo_scores)
        
        self.logger.info(f"Computed sentiment factors: {len(self._panic_scores)} valid stocks")
        
        return {
            'panic': self._panic_scores,
            'fomo': self._fomo_scores
        }
    
    def _compute_panic_score(self, df: pd.DataFrame) -> float:
        """
        计算恐慌性抛售分数
        
        检测特征:
        1. 连续N天下跌，累计跌幅 > 15%
        2. 最后一天成交量放大 2倍以上
        3. 最后一天跌幅 > 5%
        4. RSI < 30 (超卖)
        
        Returns:
            0-1 的分数，越高表示恐慌越严重
        """
        if len(df) < self.config.consecutive_days + 20:
            return 0
        
        recent = df.tail(self.config.consecutive_days + 20)
        
        # === 1. 连续下跌检测 ===
        returns = recent['close'].pct_change()
        last_n_days = returns.tail(self.config.consecutive_days)
        
        # 是否连续下跌
        is_consecutive_fall = (last_n_days < 0).all()
        
        if not is_consecutive_fall:
            return 0  # 不是连续下跌，无恐慌
        
        # 累计跌幅
        cum_return = (1 + last_n_days).prod() - 1
        
        # === 2. 成交量检测 ===
        volumes = recent['vol']
        avg_volume = volumes.iloc[:-self.config.consecutive_days].mean()
        last_day_volume = volumes.iloc[-1]
        
        volume_ratio = last_day_volume / (avg_volume + 1e-10)
        
        # === 3. 最后一天跌幅 ===
        last_day_return = returns.iloc[-1]
        
        # === 4. RSI超卖 ===
        rsi = self._compute_rsi(recent['close'], self.config.rsi_period)
        current_rsi = rsi.iloc[-1]
        
        # === 综合评分 ===
        score = 0
        
        # 1. 累计跌幅分 (跌得越多分越高)
        if cum_return < self.config.extreme_loss_threshold:
            fall_degree = abs(cum_return) / abs(self.config.extreme_loss_threshold)
            score += 0.35 * min(fall_degree, 1.0)
        
        # 2. 成交量放大分
        if volume_ratio > self.config.volume_surge_ratio:
            volume_degree = volume_ratio / (self.config.volume_surge_ratio * 2)
            score += 0.30 * min(volume_degree, 1.0)
        
        # 3. 最后一天大跌分
        if last_day_return < self.config.final_day_drop:
            drop_degree = abs(last_day_return) / abs(self.config.final_day_drop)
            score += 0.20 * min(drop_degree, 1.0)
        
        # 4. RSI超卖分
        if current_rsi < self.config.rsi_oversold:
            oversold_degree = (self.config.rsi_oversold - current_rsi) / self.config.rsi_oversold
            score += 0.15 * oversold_degree
        
        return min(score, 1.0)
    
    def _compute_fomo_score(self, df: pd.DataFrame) -> float:
        """
        计算FOMO追涨分数 (风险指标)
        
        检测特征:
        1. 连续上涨，累计涨幅 > 20%
        2. RSI > 80 (超买)
        3. 成交量持续放大
        
        Returns:
            0-1 的分数，越高表示追涨风险越大
        """
        if len(df) < self.config.consecutive_days + 20:
            return 0
        
        recent = df.tail(self.config.consecutive_days + 20)
        
        # 连续上涨
        returns = recent['close'].pct_change()
        last_n_days = returns.tail(self.config.consecutive_days)
        
        is_consecutive_rise = (last_n_days > 0).all()
        
        if not is_consecutive_rise:
            return 0
        
        # 累计涨幅
        cum_return = (1 + last_n_days).prod() - 1
        
        # RSI
        rsi = self._compute_rsi(recent['close'], self.config.rsi_period)
        current_rsi = rsi.iloc[-1]
        
        # 评分
        score = 0
        
        # 1. 大幅上涨
        if cum_return > self.config.fomo_rise_threshold:
            rise_degree = cum_return / self.config.fomo_rise_threshold
            score += 0.6 * min(rise_degree, 1.0)
        
        # 2. RSI超买
        if current_rsi > self.config.fomo_rsi_threshold:
            overbought_degree = (current_rsi - self.config.fomo_rsi_threshold) / (100 - self.config.fomo_rsi_threshold)
            score += 0.4 * overbought_degree
        
        return min(score, 1.0)
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(period).mean()
        avg_losses = losses.rolling(period).mean()
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        
        return rsi
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """
        生成交易信号
        
        策略:
        1. 选择高恐慌分数的股票 (买入机会)
        2. 排除FOMO股票 (追涨风险)
        3. Top N持仓
        """
        signals = []
        current_date = context.current_date
        
        # 检查是否有因子数据
        if self._panic_scores.empty:
            self.logger.warning("No panic scores available")
            return signals
        
        # === 1. 选择恐慌股票 ===
        panic_candidates = self._panic_scores[
            self._panic_scores > self.config.panic_score_threshold
        ]
        
        if len(panic_candidates) == 0:
            self.logger.info("No panic selling detected")
            # 如果没有恐慌机会，清空所有持仓
            for code in context.positions:
                if self._should_exit_no_opportunity(code, current_date):
                    signals.append(Signal(
                        code=code,
                        side=OrderSide.SELL,
                        weight=0.0,
                        reason="退出: 无恐慌机会"
                    ))
            return signals
        
        # === 2. 排除FOMO股票 ===
        if not self._fomo_scores.empty:
            fomo_stocks = self._fomo_scores[
                self._fomo_scores > 0.5
            ].index
            
            panic_candidates = panic_candidates.drop(fomo_stocks, errors='ignore')
        
        # === 3. 过滤不可交易股票 ===
        tradeable_stocks = self._filter_tradeable(
            panic_candidates.index.tolist(),
            context.current_data
        )
        
        if len(tradeable_stocks) == 0:
            return signals
        
        # === 4. 选择Top N ===
        panic_candidates = panic_candidates[tradeable_stocks]
        top_stocks = panic_candidates.nlargest(self.config.top_n)
        
        # === 5. 生成信号 ===
        
        # 卖出不在目标列表的持仓
        for code in context.positions:
            if code not in top_stocks.index:
                # 检查最短持有期
                if not self._can_exit(code, current_date):
                    continue
                
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason="退出: 不在目标池"
                ))
        
        # 买入目标股票
        target_weight = 1.0 / len(top_stocks) if len(top_stocks) > 0 else 0
        
        for code in top_stocks.index:
            if code not in context.positions:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.BUY,
                    weight=target_weight,
                    reason=f"恐慌买入: Panic={top_stocks[code]:.2f}"
                ))
        
        return signals
    
    def _filter_tradeable(
        self,
        candidates: List[str],
        market_data: pd.DataFrame
    ) -> List[str]:
        """过滤不可交易股票"""
        tradeable = []
        
        for code in candidates:
            code_data = market_data[market_data['code'] == code]
            
            if code_data.empty:
                continue
            
            row = code_data.iloc[0]
            
            # 涨停无法买入
            if row.get('is_limit_up', False):
                continue
            
            # 停牌
            if pd.isna(row.get('open')):
                continue
            
            tradeable.append(code)
        
        return tradeable
    
    def _can_exit(self, code: str, current_date: str) -> bool:
        """检查是否可以退出"""
        if code not in self.entry_dates:
            return True
        
        entry_date = pd.to_datetime(self.entry_dates[code])
        current = pd.to_datetime(current_date)
        
        holding_days = (current - entry_date).days
        
        return holding_days >= self.config.min_holding_days
    
    def _should_exit_no_opportunity(self, code: str, current_date: str) -> bool:
        """当没有机会时是否应该退出"""
        if code not in self.entry_dates:
            return True
        
        # 持有超过2天且没有新机会，就退出
        entry_date = pd.to_datetime(self.entry_dates[code])
        current = pd.to_datetime(current_date)
        
        holding_days = (current - entry_date).days
        
        return holding_days >= 2
    
    def on_bar(self, context: StrategyContext) -> List[Signal]:
        """
        逐日回调 - 检查止损止盈和时间止盈
        """
        signals = []
        current_date = context.current_date
        current_data = context.current_data
        
        # 更新持有天数
        for code in context.positions:
            if code in self.entry_dates:
                entry_date = pd.to_datetime(self.entry_dates[code])
                current = pd.to_datetime(current_date)
                self.holding_days[code] = (current - entry_date).days
        
        # 检查退出条件
        for code in list(context.positions.keys()):
            code_data = current_data[current_data['code'] == code]
            
            if code_data.empty:
                continue
            
            current_price = code_data.iloc[0]['close']
            
            should_exit, reason = self._check_exit_conditions(
                code,
                current_price,
                current_date
            )
            
            if should_exit:
                self.logger.info(f"[EXIT] {code}: {reason}")
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason=reason
                ))
        
        return signals
    
    def _check_exit_conditions(
        self,
        code: str,
        current_price: float,
        current_date: str
    ) -> tuple:
        """
        检查退出条件
        
        退出规则:
        1. 止损: -6%
        2. 止盈: +10%
        3. 最短持有: 1天
        4. 最长持有: 5天
        5. 小幅盈利后长持: 盈利>3% 且持有>2天
        """
        if code not in self.entry_prices:
            return False, ""
        
        entry_price = self.entry_prices[code]
        holding_days = self.holding_days.get(code, 0)
        
        # 盈亏
        pnl = (current_price - entry_price) / entry_price
        
        # 1. 止损
        if pnl < self.config.stop_loss:
            return True, f"止损 {pnl:.2%}"
        
        # 2. 止盈
        if pnl > self.config.take_profit:
            return True, f"止盈 {pnl:.2%}"
        
        # 3. 最短持有期未满
        if holding_days < self.config.min_holding_days:
            return False, ""
        
        # 4. 最长持有期
        if holding_days >= self.config.max_holding_days:
            return True, f"时间止盈 {pnl:.2%} ({holding_days}天)"
        
        # 5. 小幅盈利快速兑现 (持有2天以上，盈利>3%)
        if holding_days >= 2 and pnl > self.config.min_profit_target:
            return True, f"快速兑现 {pnl:.2%} ({holding_days}天)"
        
        return False, ""
    
    def on_order_filled(self, order) -> None:
        """订单成交回调"""
        if order.side == "BUY":
            self.entry_prices[order.code] = order.filled_price
            self.entry_dates[order.code] = order.create_date
            self.holding_days[order.code] = 0
            
            self.logger.info(f"[BUY] {order.code} @ {order.filled_price:.2f}")
        
        elif order.side == "SELL":
            if order.code in self.entry_prices:
                entry_price = self.entry_prices[order.code]
                pnl = (order.filled_price - entry_price) / entry_price
                days = self.holding_days.get(order.code, 0)
                
                self.logger.info(f"[SELL] {order.code} @ {order.filled_price:.2f} "
                               f"PnL={pnl:.2%} Days={days}")
                
                # 清除记录
                del self.entry_prices[order.code]
                del self.entry_dates[order.code]
                if order.code in self.holding_days:
                    del self.holding_days[order.code]
    
    def on_order_rejected(self, order, reason: str) -> None:
        """订单拒绝回调"""
        self.logger.warning(f"[REJECTED] {order.code} {order.side}: {reason}")


# ============================================================================
# 辅助函数
# ============================================================================
def create_sentiment_reversal_strategy(
    extreme_loss: float = -0.15,
    consecutive_days: int = 3,
    top_n: int = 12,
    **kwargs
) -> SentimentReversalStrategy:
    """
    快速创建情绪反转策略
    
    Args:
        extreme_loss: 累计跌幅阈值
        consecutive_days: 连续天数
        top_n: 持仓数量
        **kwargs: 其他配置
    
    Returns:
        策略实例
    
    Example:
        >>> strategy = create_sentiment_reversal_strategy(
        ...     extreme_loss=-0.15,
        ...     top_n=12
        ... )
    """
    config = SentimentConfig(
        extreme_loss_threshold=extreme_loss,
        consecutive_days=consecutive_days,
        top_n=top_n,
        **kwargs
    )
    
    return SentimentReversalStrategy(config)


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # 示例1: 默认配置
    strategy1 = SentimentReversalStrategy()
    
    # 示例2: 自定义配置
    custom_config = SentimentConfig(
        extreme_loss_threshold=-0.20,  # 更严格的恐慌阈值
        consecutive_days=4,
        top_n=10,
        max_holding_days=4
    )
    strategy2 = SentimentReversalStrategy(custom_config)
    
    # 示例3: 快速创建
    strategy3 = create_sentiment_reversal_strategy(
        extreme_loss=-0.15,
        top_n=12
    )
    
    print(f"策略已创建: {strategy3.name}")
    print(f"配置: {strategy3.config}")
