# ============================================================================
# 文件: strategy/momentum_reversal_combo.py
# ============================================================================
"""
动量反转组合策略 - 集成版

目标:
- Sharpe Ratio: 3.2-3.8
- Win Rate: 68-72%
- 持仓周期: 3-7天

核心逻辑:
- 中期动量(40日) 捕捉趋势
- 短期反转(5日) 捕捉超跌
- 质量过滤 保证流动性
- 动态止损止盈

集成方式:
from strategy.momentum_reversal_combo import MomentumReversalStrategy
strategy = MomentumReversalStrategy()
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
class MRConfig:
    """动量反转策略配置"""
    # 因子参数
    momentum_window: int = 40       # 中期动量窗口
    reversal_window: int = 5        # 短期反转窗口
    rsi_period: int = 14            # RSI周期
    
    # 因子权重
    momentum_weight: float = 0.45
    reversal_weight: float = 0.35
    quality_weight: float = 0.20
    
    # 选股参数
    top_n: int = 15                 # 持仓数量
    min_volume_rank: float = 0.3    # 成交量分位数下限
    max_volatility: float = 0.50    # 最大波动率
    rsi_oversold: float = 35        # RSI超卖阈值
    reversal_threshold: float = -0.05  # 反转跌幅阈值
    
    # 持仓管理
    min_holding_days: int = 3       # 最短持有期
    
    # 风控参数
    stop_loss: float = -0.08        # 止损8%
    take_profit: float = 0.15       # 止盈15%
    trailing_stop: float = 0.05     # 移动止损5%


class MomentumReversalStrategy(BaseStrategy):
    """动量反转组合策略"""
    
    name = "momentum_reversal_combo"
    
    def __init__(self, config: MRConfig = None):
        super().__init__()
        self.config = config or MRConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 风险管理
        self.risk_manager = RiskManager(
            max_single_weight=0.10,  # 单股最大10%
            max_total_weight=0.95
        )
        
        # 持仓跟踪
        self.entry_prices: Dict[str, float] = {}
        self.entry_dates: Dict[str, str] = {}
        self.highest_prices: Dict[str, float] = {}
        
        # 因子缓存
        self._factors: Dict[str, pd.Series] = {}
    
    def initialize(self) -> None:
        """初始化策略"""
        self.logger.info(f"Initializing {self.name}")
        self.logger.info(f"Config: momentum={self.config.momentum_window}, "
                        f"reversal={self.config.reversal_window}, "
                        f"top_n={self.config.top_n}")
    
    def compute_factors(self, history_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        计算因子 - 向量化批量实现

        Args:
            history_data: {code: DataFrame(date, open, high, low, close, vol)}

        Returns:
            {factor_name: Series(code → score)}
        """
        self.logger.info(f"Computing factors for {len(history_data)} stocks (vectorized)...")

        # 向量化批量计算所有股票的因子
        momentum_scores = self._compute_momentum_batch(history_data)
        reversal_scores = self._compute_reversal_batch(history_data)
        quality_scores = self._compute_quality_batch(history_data)

        # 转为Series并标准化
        momentum = pd.Series(momentum_scores)
        reversal = pd.Series(reversal_scores)
        quality = pd.Series(quality_scores)

        # 标准化到[-1, 1]
        momentum = self._standardize(momentum)
        reversal = self._standardize(reversal)
        quality = self._standardize(quality)

        self.logger.info(f"Computed factors: {len(momentum)} valid stocks")

        return {
            'momentum': momentum,
            'reversal': reversal,
            'quality': quality
        }

    def _compute_momentum_batch(self, history_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """向量化批量计算动量因子"""
        momentum_scores = {}
        window = self.config.momentum_window

        for code, df in history_data.items():
            if len(df) < window:
                continue

            try:
                # 使用向量化操作计算动量
                closes = df['close'].values
                returns = np.diff(closes) / np.where(closes[:-1] > 0, closes[:-1], 1)

                # 计算窗口内的累计收益
                cum_return = np.prod(1 + returns[-window:]) - 1

                # 计算波动率
                volatility = np.std(returns[-window:]) * np.sqrt(252) if len(returns) >= window else 0.01

                # Sharpe动量
                sharpe_momentum = cum_return / volatility if volatility > 0.01 else 0

                # 趋势强度: 线性回归R² (向量化)
                prices = closes[-window:]
                x = np.arange(len(prices), dtype=np.float64)
                x_mean = x.mean()
                y_mean = prices.mean()

                if len(prices) > 5:
                    cov_xy = ((x - x_mean) * (prices - y_mean)).sum()
                    var_x = ((x - x_mean) ** 2).sum()
                    var_y = ((prices - y_mean) ** 2).sum()

                    slope = cov_xy / var_x if var_x > 1e-10 else 0
                    r2 = (cov_xy ** 2) / (var_x * var_y) if var_x * var_y > 1e-10 else 0

                    trend_strength = slope * r2 / (y_mean + 1e-10)
                else:
                    trend_strength = 0

                # 综合动量分数
                score = 0.6 * sharpe_momentum + 0.4 * trend_strength
                momentum_scores[code] = score

            except Exception:
                continue

        return momentum_scores

    def _compute_reversal_batch(self, history_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """向量化批量计算反转因子"""
        reversal_scores = {}
        window = self.config.reversal_window

        for code, df in history_data.items():
            if len(df) < window + 14:
                continue

            try:
                closes = df['close'].values
                returns = np.diff(closes) / np.where(closes[:-1] > 0, closes[:-1], 1)

                # 短期收益 (向量化)
                short_return = np.prod(1 + returns[-window:]) - 1

                # RSI (向量化计算)
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)

                avg_gains = np.mean(gains[-14:]) if len(gains) >= 14 else 0
                avg_losses = np.mean(losses[-14:]) if len(losses) >= 14 else 0

                rs = avg_gains / (avg_losses + 1e-10)
                current_rsi = 100 - 100 / (1 + rs)

                # 反转分数
                score = 0

                # 1. 显著下跌 (跌幅>5%)
                if short_return < self.config.reversal_threshold:
                    score += -short_return  # 跌得越多分数越高

                    # 2. RSI超卖加成
                    if current_rsi < self.config.rsi_oversold:
                        oversold_degree = (self.config.rsi_oversold - current_rsi) / self.config.rsi_oversold
                        score *= (1 + oversold_degree)
                else:
                    score = 0  # 不超卖则无反转机会

                reversal_scores[code] = score

            except Exception:
                continue

        return reversal_scores

    def _compute_quality_batch(self, history_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """向量化批量计算质量因子"""
        quality_scores = {}

        for code, df in history_data.items():
            if len(df) < 60:
                continue

            try:
                volumes = df['vol'].values[-60:]
                closes = df['close'].values

                # 1. 成交量分位数 (向量化)
                current_vol = volumes[-1]
                volume_rank = np.mean(volumes[:-1] < current_vol) if len(volumes) > 1 else 0.5

                # 2. 波动率 (向量化)
                returns = np.diff(closes[-21:]) / np.where(closes[-21:-1] > 0, closes[-21:-1], 1)
                volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.25

                # 流动性评分 (倒U型: 50-70分位最优)
                liquidity_score = 1 - 4 * (volume_rank - 0.6) ** 2
                liquidity_score = max(0, min(1, liquidity_score))

                # 波动率评分 (适中最好: 15-35%)
                vol_optimal = 0.25
                vol_score = 1 - ((volatility - vol_optimal) / 0.2) ** 2
                vol_score = max(0, min(1, vol_score))

                # 综合质量分数
                quality = 0.6 * liquidity_score + 0.4 * vol_score
                quality_scores[code] = quality

            except Exception:
                continue

        return quality_scores
    
    def _compute_momentum(self, df: pd.DataFrame) -> float:
        """
        计算动量因子
        
        方法: Sharpe动量 (收益/波动率)
        """
        window = self.config.momentum_window
        
        if len(df) < window:
            return 0
        
        # 收益率
        returns = df['close'].pct_change()
        
        # 累计收益
        cum_return = (1 + returns.tail(window)).prod() - 1
        
        # 波动率
        volatility = returns.tail(window).std() * np.sqrt(252)
        
        # Sharpe动量
        if volatility > 0.01:
            sharpe_momentum = cum_return / volatility
        else:
            sharpe_momentum = 0
        
        # 趋势强度: 线性回归R²
        prices = df['close'].tail(window).values
        x = np.arange(len(prices))
        
        if len(prices) > 5:
            slope = np.polyfit(x, prices, 1)[0]
            y_pred = np.poly1d(np.polyfit(x, prices, 1))(x)
            ss_res = ((prices - y_pred) ** 2).sum()
            ss_tot = ((prices - prices.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            trend_strength = slope * r2 / (prices.mean() + 1e-10)
        else:
            trend_strength = 0
        
        # 综合动量分数
        score = 0.6 * sharpe_momentum + 0.4 * trend_strength
        
        return score
    
    def _compute_reversal(self, df: pd.DataFrame) -> float:
        """
        计算反转因子
        
        逻辑: 短期超跌 + RSI超卖
        """
        window = self.config.reversal_window
        
        if len(df) < window + 14:
            return 0
        
        # 短期收益
        returns = df['close'].pct_change()
        short_return = (1 + returns.tail(window)).prod() - 1
        
        # RSI
        rsi = self._compute_rsi(df['close'], self.config.rsi_period)
        current_rsi = rsi.iloc[-1]
        
        # 反转分数
        score = 0
        
        # 1. 显著下跌 (跌幅>5%)
        if short_return < self.config.reversal_threshold:
            score += -short_return  # 跌得越多分数越高
        
        # 2. RSI超卖加成
        if current_rsi < self.config.rsi_oversold:
            oversold_degree = (self.config.rsi_oversold - current_rsi) / self.config.rsi_oversold
            score *= (1 + oversold_degree)
        else:
            score = 0  # 不超卖则无反转机会
        
        return score
    
    def _compute_quality(self, df: pd.DataFrame) -> float:
        """
        计算质量因子
        
        考虑: 流动性 + 波动率适中
        """
        if len(df) < 60:
            return 0
        
        # 1. 成交量分位数
        volumes = df['vol'].tail(60)
        current_vol = volumes.iloc[-1]
        volume_rank = (current_vol > volumes[:-1]).sum() / (len(volumes) - 1)
        
        # 2. 波动率
        returns = df['close'].pct_change()
        volatility = returns.tail(20).std() * np.sqrt(252)
        
        # 流动性评分 (倒U型: 50-70分位最优)
        liquidity_score = 1 - 4 * (volume_rank - 0.6) ** 2
        liquidity_score = max(0, min(1, liquidity_score))
        
        # 波动率评分 (适中最好: 15-35%)
        vol_optimal = 0.25
        vol_score = 1 - ((volatility - vol_optimal) / 0.2) ** 2
        vol_score = max(0, min(1, vol_score))
        
        # 综合质量分数
        quality = 0.6 * liquidity_score + 0.4 * vol_score
        
        return quality
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(period).mean()
        avg_losses = losses.rolling(period).mean()
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        
        return rsi
    
    def _standardize(self, series: pd.Series) -> pd.Series:
        """标准化到[-1, 1]"""
        if len(series) == 0:
            return series
        
        # Rank百分位
        ranked = series.rank(pct=True)
        
        # 映射到[-1, 1]
        standardized = ranked * 2 - 1
        
        return standardized
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            context: 策略上下文 (包含当前数据、历史数据、持仓等)
        
        Returns:
            信号列表
        """
        signals = []
        current_date = context.current_date
        
        # 获取因子
        if not self._factors:
            self.logger.warning("No factors available")
            return signals
        
        momentum = self._factors.get('momentum', pd.Series())
        reversal = self._factors.get('reversal', pd.Series())
        quality = self._factors.get('quality', pd.Series())
        
        # 综合评分
        common_codes = momentum.index.intersection(reversal.index).intersection(quality.index)
        
        if len(common_codes) == 0:
            return signals
        
        alpha_scores = (
            self.config.momentum_weight * momentum[common_codes] +
            self.config.reversal_weight * reversal[common_codes] +
            self.config.quality_weight * quality[common_codes]
        )
        
        # 只选正分数 (看好的股票)
        alpha_scores = alpha_scores[alpha_scores > 0]
        
        # 排序选Top N
        top_stocks = alpha_scores.nlargest(self.config.top_n * 2)  # 多选些备用
        
        # 过滤不可交易股票
        tradeable_stocks = self._filter_tradeable(
            top_stocks.index.tolist(),
            context.current_data
        )
        
        # 最终选择
        final_stocks = tradeable_stocks[:self.config.top_n]
        
        # === 生成信号 ===
        
        # 1. 卖出信号 (不在目标列表中的持仓)
        for code in context.positions:
            if code not in final_stocks:
                # 检查最短持有期
                if self._should_hold(code, current_date):
                    continue
                
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason="退出: 不在目标股票池"
                ))
        
        # 2. 买入信号 (目标股票)
        target_weight = 1.0 / len(final_stocks) if final_stocks else 0
        
        for code in final_stocks:
            if code not in context.positions:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.BUY,
                    weight=target_weight,
                    reason=f"买入: Alpha={alpha_scores[code]:.3f}"
                ))
        
        return signals
    
    def _filter_tradeable(
        self,
        candidates: List[str],
        market_data: pd.DataFrame
    ) -> List[str]:
        """过滤不可交易股票 - 向量化实现"""
        if market_data.empty or not candidates:
            return []

        # 向量化过滤条件
        # 1. 在候选列表中
        mask = market_data['code'].isin(candidates)

        # 2. 非涨停
        if 'is_limit_up' in market_data.columns:
            mask &= ~market_data['is_limit_up']

        # 3. 非停牌 (open不为NaN)
        if 'open' in market_data.columns:
            mask &= market_data['open'].notna()

        # 获取可交易股票列表
        tradeable = market_data.loc[mask, 'code'].tolist()

        # 保持原始排序
        tradeable_sorted = [code for code in candidates if code in tradeable]

        return tradeable_sorted
    
    def _should_hold(self, code: str, current_date: str) -> bool:
        """检查是否应继续持有 (最短持有期)"""
        if code not in self.entry_dates:
            return False
        
        entry_date = pd.to_datetime(self.entry_dates[code])
        current = pd.to_datetime(current_date)
        
        holding_days = (current - entry_date).days
        
        return holding_days < self.config.min_holding_days
    
    def on_bar(self, context: StrategyContext) -> List[Signal]:
        """
        逐日回调 - 检查止损止盈
        
        Args:
            context: 策略上下文
        
        Returns:
            额外信号 (止损/止盈)
        """
        signals = []
        current_date = context.current_date
        current_data = context.current_data
        
        for code in list(context.positions.keys()):
            # 获取当前价格
            code_data = current_data[current_data['code'] == code]
            
            if code_data.empty:
                continue
            
            current_price = code_data.iloc[0]['close']
            
            # 检查止损止盈
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
        
        Returns:
            (should_exit, reason)
        """
        if code not in self.entry_prices:
            return False, ""
        
        entry_price = self.entry_prices[code]
        highest = self.highest_prices.get(code, entry_price)
        
        # 更新最高价
        if current_price > highest:
            self.highest_prices[code] = current_price
            highest = current_price
        
        # 盈亏计算
        pnl = (current_price - entry_price) / entry_price
        
        # 1. 止损
        if pnl < self.config.stop_loss:
            return True, f"止损 {pnl:.2%}"
        
        # 2. 止盈
        if pnl > self.config.take_profit:
            return True, f"止盈 {pnl:.2%}"
        
        # 3. 移动止损
        drawdown = (current_price - highest) / highest
        if drawdown < -self.config.trailing_stop:
            return True, f"移动止损 {drawdown:.2%} (最高{highest:.2f})"
        
        return False, ""
    
    def on_order_filled(self, order) -> None:
        """订单成交回调"""
        if order.side == "BUY":
            # 记录买入信息
            self.entry_prices[order.code] = order.filled_price
            self.entry_dates[order.code] = order.create_date
            self.highest_prices[order.code] = order.filled_price
            
            self.logger.info(f"[BUY] {order.code} @ {order.filled_price:.2f}")
        
        elif order.side == "SELL":
            # 清除持仓记录
            if order.code in self.entry_prices:
                entry_price = self.entry_prices[order.code]
                pnl = (order.filled_price - entry_price) / entry_price
                
                self.logger.info(f"[SELL] {order.code} @ {order.filled_price:.2f} "
                               f"PnL={pnl:.2%}")
                
                del self.entry_prices[order.code]
                del self.entry_dates[order.code]
                if order.code in self.highest_prices:
                    del self.highest_prices[order.code]
    
    def on_order_rejected(self, order, reason: str) -> None:
        """订单拒绝回调"""
        self.logger.warning(f"[REJECTED] {order.code} {order.side}: {reason}")


# ============================================================================
# 辅助函数: 快速创建策略
# ============================================================================
def create_momentum_reversal_strategy(
    momentum_window: int = 40,
    reversal_window: int = 5,
    top_n: int = 15,
    **kwargs
) -> MomentumReversalStrategy:
    """
    快速创建动量反转策略
    
    Args:
        momentum_window: 动量窗口
        reversal_window: 反转窗口
        top_n: 持仓数量
        **kwargs: 其他配置参数
    
    Returns:
        策略实例
    
    Example:
        >>> strategy = create_momentum_reversal_strategy(
        ...     momentum_window=40,
        ...     top_n=15
        ... )
    """
    config = MRConfig(
        momentum_window=momentum_window,
        reversal_window=reversal_window,
        top_n=top_n,
        **kwargs
    )
    
    return MomentumReversalStrategy(config)


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # 示例1: 默认配置
    strategy1 = MomentumReversalStrategy()
    
    # 示例2: 自定义配置
    custom_config = MRConfig(
        momentum_window=50,
        reversal_window=7,
        top_n=20,
        stop_loss=-0.10
    )
    strategy2 = MomentumReversalStrategy(custom_config)
    
    # 示例3: 快速创建
    strategy3 = create_momentum_reversal_strategy(
        momentum_window=40,
        top_n=15
    )
    
    print(f"策略已创建: {strategy3.name}")
    print(f"配置: {strategy3.config}")
