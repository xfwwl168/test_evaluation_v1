# ============================================================================
# 文件: factors/alpha_hunter_v2_factors.py
# ============================================================================
"""
Alpha-Hunter-V2 因子模块

核心改进:
1. 自适应 RSRS (市场状态感知)
2. 精细化开盘异动检测
3. 多层次压力位计算
4. 板块轮动强度因子
5. 筹码分布压力因子
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.signal import argrelextrema
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from factors.base import BaseFactor, FactorMeta, FactorRegistry
from config import settings


# ============================================================================
# 数据结构定义
# ============================================================================

class MarketState(Enum):
    """市场状态"""
    BULL_STRONG = "强势牛市"
    BULL_WEAK = "弱势牛市"
    BEAR_STRONG = "强势熊市"
    BEAR_WEAK = "弱势熊市"
    SHOCK = "震荡市"


@dataclass
class AlphaFactorResult:
    """Alpha 因子计算结果"""
    # 核心因子
    rsrs_adaptive: float  # 自适应 RSRS
    rsrs_r2: float  # R² 拟合度
    rsrs_momentum: float  # RSRS 动量 (加速度)

    # 价格动量
    opening_surge: float  # 开盘异动强度
    price_momentum: float  # 价格动量
    volume_surge: float  # 量能异动

    # 压力支撑
    pressure_distance: float  # 距压力位距离
    support_distance: float  # 距支撑位距离
    chip_pressure: float  # 筹码压力指数

    # 市场状态
    market_state: MarketState  # 市场状态
    sector_momentum: float  # 板块动量

    # 综合评分
    alpha_score: float  # 综合 Alpha 评分
    signal_quality: float  # 信号质量 (0-1)

    # 风险指标
    risk_score: float  # 风险评分
    volatility_regime: int  # 波动率状态


# ============================================================================
# 自适应 RSRS 因子
# ============================================================================

@FactorRegistry.register
class AdaptiveRSRSFactor(BaseFactor):
    """
    自适应 RSRS 因子

    改进点:
    1. 市场状态感知的偏度修正
    2. 波动率自适应窗口
    3. 异常值鲁棒处理
    4. 斜率加速度 (二阶导)
    """

    meta = FactorMeta(
        name="rsrs_adaptive_v2",
        category="technical",
        description="自适应 RSRS V2",
        lookback=700
    )

    def __init__(
            self,
            base_window: int = 18,
            std_window: int = 600,
            min_r2: float = 0.7,
            vol_adjust: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.base_window = base_window
        self.std_window = std_window
        self.min_r2 = min_r2
        self.vol_adjust = vol_adjust

        self.logger = logging.getLogger("AdaptiveRSRS")

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算自适应 RSRS"""
        result = self.compute_full(df)
        return result['rsrs_adaptive']

    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整计算"""
        n = len(df)
        if n < self.base_window + 100:
            return self._empty_result(df.index)

        high = df['high'].to_numpy(dtype=np.float64)
        low = df['low'].to_numpy(dtype=np.float64)
        close = df['close'].to_numpy(dtype=np.float64)
        volume = df['vol'].to_numpy(dtype=np.float64)

        # ===== 1. 动态窗口计算 (波动率自适应) =====
        if self.vol_adjust:
            window = self._calc_adaptive_window(close, volume)
        else:
            window = np.full(n, self.base_window)

        # ===== 2. 滚动 OLS 回归 =====
        slope, r2, residual_std = self._vectorized_ols_multi_window(high, low, window)

        # ===== 3. 鲁棒标准化 (MAD 方法) =====
        zscore = self._robust_zscore(slope, self.std_window)

        # ===== 4. R² 加权 =====
        score_r2 = zscore * r2

        # ===== 5. 市场状态自适应偏度修正 =====
        market_state = self._detect_market_state(close, volume)
        skew_penalty = self._adaptive_skew_penalty(slope, market_state, self.std_window)

        score_adjusted = np.where(
            score_r2 > 0,
            score_r2 * (1 - skew_penalty),
            score_r2
        )

        # ===== 6. 有效性过滤 =====
        valid = (r2 >= self.min_r2).astype(np.int8)
        final_score = np.where(valid, score_adjusted, score_adjusted * 0.3)

        # ===== 7. 动量 (斜率加速度) =====
        momentum = np.gradient(np.nan_to_num(slope, 0))
        acceleration = np.gradient(momentum)

        # ===== 8. 信号质量评估 =====
        quality = self._calc_signal_quality(r2, residual_std, valid)

        return pd.DataFrame({
            'rsrs_slope': slope,
            'rsrs_r2': r2,
            'rsrs_zscore': zscore,
            'rsrs_adaptive': final_score,
            'rsrs_momentum': momentum,
            'rsrs_acceleration': acceleration,
            'rsrs_valid': valid,
            'signal_quality': quality,
            'market_state': market_state
        }, index=df.index)

    def _calc_adaptive_window(
            self,
            close: np.ndarray,
            volume: np.ndarray
    ) -> np.ndarray:
        """
        计算自适应窗口 - 完全向量化实现

        高波动时缩短窗口 (更敏感)
        低波动时延长窗口 (更稳定)
        """
        n = len(close)
        window = np.full(n, self.base_window, dtype=np.float64)

        # 20 日波动率 - 向量化计算
        returns = np.diff(close, prepend=close[0]) / np.clip(close, 1e-10, None)

        # 使用pandas rolling计算20日波动率 (更高效的向量化实现)
        returns_series = pd.Series(returns)
        vol_20_series = returns_series.rolling(20).std() * np.sqrt(252)
        vol_20 = vol_20_series.to_numpy()

        # 向量化计算历史波动率分位数
        # 使用expanding window计算历史分位数
        vol_expanding = returns_series.rolling(20).std().expanding(min_periods=30).apply(
            lambda x: (x.iloc[-1] > x).mean() if len(x) > 0 else 0.5,
            raw=False
        ).to_numpy()

        # 向量化窗口调整
        vol_pct = np.where(np.arange(n) >= 252, vol_expanding, 0.5)

        # 向量化条件判断
        window = np.where(vol_pct > 0.8, max(12, self.base_window - 4), window)
        window = np.where((vol_pct > 0.6) & (vol_pct <= 0.8), max(14, self.base_window - 2), window)
        window = np.where(vol_pct < 0.2, min(24, self.base_window + 4), window)
        window = np.where((vol_pct < 0.4) & (vol_pct >= 0.2), min(22, self.base_window + 2), window)

        # 前30个使用默认窗口
        window[:30] = self.base_window

        return window.astype(int)

    def _vectorized_ols_multi_window(
            self,
            high: np.ndarray,
            low: np.ndarray,
            windows: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """多窗口 OLS (使用固定窗口近似)"""
        # 使用基础窗口计算，窗口变化时做插值调整
        window = self.base_window
        n = len(high)

        if n < window:
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

        low_win = sliding_window_view(low, window)
        high_win = sliding_window_view(high, window)

        x_mean = low_win.mean(axis=1, keepdims=True)
        y_mean = high_win.mean(axis=1, keepdims=True)

        x_dev = low_win - x_mean
        y_dev = high_win - y_mean

        cov_xy = (x_dev * y_dev).sum(axis=1)
        var_x = (x_dev ** 2).sum(axis=1)
        var_y = (y_dev ** 2).sum(axis=1)

        slope = np.divide(cov_xy, var_x, out=np.zeros_like(cov_xy), where=var_x > 1e-10)

        denom = var_x * var_y
        r2 = np.divide(cov_xy ** 2, denom, out=np.zeros_like(cov_xy), where=denom > 1e-10)

        # 残差标准差
        intercept = y_mean.flatten() - slope * x_mean.flatten()
        y_pred = slope.reshape(-1, 1) * low_win + intercept.reshape(-1, 1)
        residual_std = (high_win - y_pred).std(axis=1)

        # 填充
        pad = np.full(window - 1, np.nan)
        slope_full = np.concatenate([pad, slope])
        r2_full = np.concatenate([pad, r2])
        residual_full = np.concatenate([pad, residual_std])

        return slope_full, r2_full, residual_full

    def _robust_zscore(self, arr: np.ndarray, window: int) -> np.ndarray:
        """
        鲁棒 Z-Score (使用 MAD) - 向量化实现

        MAD = Median Absolute Deviation
        比标准差更抗异常值
        """
        # 使用pandas进行向量化rolling计算
        series = pd.Series(arr)

        # 向量化rolling median
        rolling_median = series.rolling(window=window, min_periods=60).median()

        # 向量化MAD计算
        def mad_calc(x):
            if len(x) < 60:
                return np.nan
            med = np.median(x)
            return np.median(np.abs(x - med)) * 1.4826

        rolling_mad = series.rolling(window=window, min_periods=60).apply(
            mad_calc, raw=True
        )

        # 避免除零
        rolling_mad = rolling_mad.replace(0, np.nan)

        # 计算z-score
        zscore = (series - rolling_median) / rolling_mad

        return zscore.to_numpy()

    def _detect_market_state(
            self,
            close: np.ndarray,
            volume: np.ndarray
    ) -> np.ndarray:
        """检测市场状态 - 向量化实现"""
        n = len(close)

        # 使用pandas向量化计算移动平均线
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)

        ma60 = close_series.rolling(60, min_periods=60).mean().to_numpy()
        ma20 = close_series.rolling(20, min_periods=20).mean().to_numpy()

        # 向量化计算价格相对均线的位置
        price_vs_ma60 = (close - ma60) / np.where(ma60 > 0, ma60, 1)
        price_vs_ma20 = (close - ma20) / np.where(ma20 > 0, ma20, 1)

        # 向量化计算量能比率
        vol_20 = volume_series.rolling(20, min_periods=20).mean().to_numpy()
        vol_5 = volume_series.rolling(5, min_periods=5).mean().to_numpy()
        vol_ratio = np.where(vol_20 > 0, vol_5 / vol_20, 1)

        # 向量化条件判断
        states = np.full(n, MarketState.SHOCK.value, dtype=object)

        # 强势牛市条件
        bull_strong = (price_vs_ma60 > 0.1) & (price_vs_ma20 > 0.03) & (vol_ratio > 1.2)
        # 弱势牛市条件
        bull_weak = (price_vs_ma60 > 0.1) & (price_vs_ma20 > 0.03) & (vol_ratio <= 1.2)
        # 强势熊市条件
        bear_strong = (price_vs_ma60 < -0.1) & (price_vs_ma20 < -0.03) & (vol_ratio > 1.2)
        # 弱势熊市条件
        bear_weak = (price_vs_ma60 < -0.1) & (price_vs_ma20 < -0.03) & (vol_ratio <= 1.2)

        states = np.where(bull_strong, MarketState.BULL_STRONG.value, states)
        states = np.where(bull_weak, MarketState.BULL_WEAK.value, states)
        states = np.where(bear_strong, MarketState.BEAR_STRONG.value, states)
        states = np.where(bear_weak, MarketState.BEAR_WEAK.value, states)

        # 前60个为震荡市
        states[:60] = MarketState.SHOCK.value

        return states

    def _adaptive_skew_penalty(
            self,
            slope: np.ndarray,
            market_states: np.ndarray,
            window: int
    ) -> np.ndarray:
        """自适应偏度惩罚 - 向量化实现"""
        n = len(slope)

        # 定义市场状态到惩罚系数的映射
        penalty_map = {
            MarketState.BULL_STRONG.value: 0.05,  # 牛市中右偏是正常的，惩罚较轻
            MarketState.BULL_WEAK.value: 0.10,
            MarketState.BEAR_STRONG.value: 0.20,  # 熊市中右偏可能是诱多，惩罚较重
            MarketState.BEAR_WEAK.value: 0.15,
            MarketState.SHOCK.value: 0.12
        }

        # 向量化基础惩罚系数
        base_penalty = np.array([penalty_map.get(state, 0.12) for state in market_states])

        # 使用pandas rolling计算偏度
        slope_series = pd.Series(slope)

        def calc_skew(x):
            if len(x) < 60:
                return 0
            valid = x[~np.isnan(x)]
            if len(valid) < 60:
                return 0
            return stats.skew(valid)

        # 向量化rolling skewness
        skewness = slope_series.rolling(window=window, min_periods=60).apply(
            calc_skew, raw=True
        ).to_numpy()

        # 向量化计算惩罚 (只惩罚右偏)
        skewness_clipped = np.where(skewness > 0, skewness, 0)
        penalty = np.clip(skewness_clipped * base_penalty, 0, 0.5)

        # 前window个置为0
        penalty[:window] = 0

        return penalty

    def _calc_signal_quality(
            self,
            r2: np.ndarray,
            residual_std: np.ndarray,
            valid: np.ndarray
    ) -> np.ndarray:
        """计算信号质量"""
        # 质量 = R² × (1 - 残差标准化) × 有效性
        residual_norm = np.clip(residual_std / np.nanmax(residual_std), 0, 1)
        quality = r2 * (1 - residual_norm * 0.5) * valid
        return np.nan_to_num(quality, 0)

    def _empty_result(self, index) -> pd.DataFrame:
        return pd.DataFrame({
            'rsrs_slope': np.nan,
            'rsrs_r2': np.nan,
            'rsrs_zscore': np.nan,
            'rsrs_adaptive': np.nan,
            'rsrs_momentum': np.nan,
            'rsrs_acceleration': np.nan,
            'rsrs_valid': 0,
            'signal_quality': 0,
            'market_state': MarketState.SHOCK.value
        }, index=index)


# ============================================================================
# 开盘异动因子
# ============================================================================

@FactorRegistry.register
class OpeningSurgeFactor(BaseFactor):
    """
    开盘异动因子 V2

    检测:
    1. 开盘跳空幅度
    2. 开盘量能异动 (模拟早盘15分钟)
    3. 集合竞价强度
    4. 开盘后价格走势
    """

    meta = FactorMeta(
        name="opening_surge_v2",
        category="technical",
        description="开盘异动因子 V2",
        lookback=20
    )

    def __init__(
            self,
            gap_threshold: float = 0.02,
            volume_threshold: float = 0.10,  # 早盘量占比阈值
            **kwargs
    ):
        super().__init__(**kwargs)
        self.gap_threshold = gap_threshold
        self.volume_threshold = volume_threshold

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算开盘异动评分"""
        result = self.compute_full(df)
        return result['surge_score']

    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整计算"""
        n = len(df)

        open_price = df['open'].to_numpy()
        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        volume = df['vol'].to_numpy().astype(np.float64)

        # 昨收
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        prev_volume = np.roll(volume, 1)
        prev_volume[0] = volume[0]

        # ===== 1. 跳空幅度 =====
        gap = (open_price - prev_close) / prev_close
        gap_score = np.clip(gap / self.gap_threshold, -2, 2)

        # ===== 2. 开盘位置 (在当日区间的位置) =====
        intraday_range = high - low
        open_position = np.where(
            intraday_range > 1e-10,
            (open_price - low) / intraday_range,
            0.5
        )
        # 开盘在高位更强势
        open_pos_score = (open_position - 0.5) * 2

        # ===== 3. 早盘量能估算 =====
        # 使用 (high-open)/(high-low) 估算早盘强度
        early_strength = np.where(
            intraday_range > 1e-10,
            (high - open_price) / intraday_range,
            0.5
        )

        # 量能集中度 (收盘接近最高价说明量能在尾盘，否则在早盘)
        close_to_high = np.where(
            intraday_range > 1e-10,
            (close - low) / intraday_range,
            0.5
        )

        # 早盘量能估计 = 全天量 × (1 - 收盘接近最高价的程度)
        early_volume_ratio = (1 - close_to_high) * 0.4 + 0.1  # 10%-50%
        early_volume_score = np.where(
            early_volume_ratio > self.volume_threshold,
            early_volume_ratio / self.volume_threshold,
            early_volume_ratio / self.volume_threshold * 0.5
        )

        # ===== 4. 量能异动 =====
        volume_ma5 = pd.Series(volume).rolling(5, min_periods=1).mean().to_numpy()
        volume_ratio = volume / np.clip(volume_ma5, 1, None)
        volume_surge = np.clip(volume_ratio - 1, -1, 3)

        # ===== 5. 集合竞价强度 (用跳空+开盘位置模拟) =====
        auction_strength = gap_score * 0.5 + open_pos_score * 0.5

        # ===== 6. 综合评分 =====
        surge_score = (
                0.30 * gap_score +
                0.20 * open_pos_score +
                0.25 * early_volume_score +
                0.15 * volume_surge +
                0.10 * auction_strength
        )

        # ===== 7. 信号标记 =====
        is_surge = (
                (gap > self.gap_threshold) |
                (volume_ratio > 2.0) |
                (surge_score > 1.0)
        ).astype(np.int8)

        return pd.DataFrame({
            'gap': gap,
            'gap_score': gap_score,
            'open_position': open_position,
            'early_volume_ratio': early_volume_ratio,
            'volume_surge': volume_surge,
            'auction_strength': auction_strength,
            'surge_score': surge_score,
            'is_surge': is_surge
        }, index=df.index)


# ============================================================================
# 多层次压力位因子
# ============================================================================

@FactorRegistry.register
class MultiLevelPressureFactor(BaseFactor):
    """
    多层次压力位因子

    计算:
    1. 技术压力 (20日/60日高点)
    2. 筹码压力 (成交密集区)
    3. 心理压力 (整数关口)
    4. 套牢盘压力 (历史高点回溯)
    """

    meta = FactorMeta(
        name="pressure_multi_v2",
        category="technical",
        description="多层次压力位 V2",
        lookback=250
    )

    def __init__(
            self,
            short_window: int = 20,
            mid_window: int = 60,
            pressure_lookback: int = 250,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.mid_window = mid_window
        self.pressure_lookback = pressure_lookback

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算综合压力距离"""
        result = self.compute_full(df)
        return result['combined_pressure_dist']

    def compute_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整计算"""
        n = len(df)

        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        volume = df['vol'].to_numpy().astype(np.float64)

        # ===== 1. 技术压力 =====
        tech_pressure_20 = self._calc_technical_pressure(high, self.short_window)
        tech_pressure_60 = self._calc_technical_pressure(high, self.mid_window)

        # ===== 2. 筹码压力 =====
        chip_pressure = self._calc_chip_pressure(close, high, low, volume)

        # ===== 3. 整数关口压力 =====
        round_pressure = self._calc_round_pressure(close)

        # ===== 4. 套牢盘压力 =====
        trapped_pressure = self._calc_trapped_pressure(close, high, volume)

        # ===== 5. 综合压力位 =====
        combined_pressure = np.minimum.reduce([
            np.where(tech_pressure_20 > close, tech_pressure_20, np.inf),
            np.where(tech_pressure_60 > close, tech_pressure_60, np.inf),
            np.where(chip_pressure > close, chip_pressure, np.inf),
            np.where(round_pressure > close, round_pressure, np.inf),
            np.where(trapped_pressure > close, trapped_pressure, np.inf)
        ])

        # 替换无穷大
        combined_pressure = np.where(
            np.isinf(combined_pressure),
            close * 1.1,
            combined_pressure
        )

        # ===== 6. 压力距离 =====
        pressure_dist = (combined_pressure - close) / close

        # ===== 7. 支撑位 =====
        support_20 = pd.Series(low).rolling(self.short_window, min_periods=5).min().to_numpy()
        support_dist = (close - support_20) / close

        # ===== 8. 安全评分 =====
        safety_score = np.clip(pressure_dist * 10, 0, 1) * 0.6 + np.clip(support_dist * 10, 0, 1) * 0.4

        return pd.DataFrame({
            'tech_pressure_20': tech_pressure_20,
            'tech_pressure_60': tech_pressure_60,
            'chip_pressure': chip_pressure,
            'round_pressure': round_pressure,
            'trapped_pressure': trapped_pressure,
            'combined_pressure': combined_pressure,
            'combined_pressure_dist': pressure_dist,
            'support_20': support_20,
            'support_dist': support_dist,
            'safety_score': safety_score
        }, index=df.index)

    def _calc_technical_pressure(self, high: np.ndarray, window: int) -> np.ndarray:
        """技术压力 (滚动最高价)"""
        return pd.Series(high).rolling(window, min_periods=5).max().to_numpy()

    def _calc_chip_pressure(
            self,
            close: np.ndarray,
            high: np.ndarray,
            low: np.ndarray,
            volume: np.ndarray
    ) -> np.ndarray:
        """筹码压力 (成交密集区) - 向量化优化实现"""
        n = len(close)
        lookback = min(self.pressure_lookback, n)

        # 使用滚动窗口计算价格分箱和成交量分布
        # 为了性能，我们使用简化的向量化方法

        # 计算滚动最高价和最低价
        high_series = pd.Series(high)
        low_series = pd.Series(low)

        rolling_high = high_series.rolling(lookback, min_periods=lookback//2).max().to_numpy()
        rolling_low = low_series.rolling(lookback, min_periods=lookback//2).min().to_numpy()

        # 使用价格位置作为筹码压力的代理 (向量化)
        # 价格接近滚动高点时压力大
        price_range = rolling_high - rolling_low
        price_position = np.where(
            price_range > 1e-10,
            (close - rolling_low) / price_range,
            0.5
        )

        # 计算滚动成交量加权平均价格 (VWAP) 作为压力位
        close_series = pd.Series(close)
        vol_series = pd.Series(volume)

        # 向量化计算VWAP
        vwap = (close_series * vol_series).rolling(lookback, min_periods=lookback//2).sum() / \
               vol_series.rolling(lookback, min_periods=lookback//2).sum()
        vwap = vwap.to_numpy()

        # 压力位于VWAP和当前价之间的最大值
        pressure = np.maximum(vwap, close * 1.02)

        # 当价格处于高位时，压力为滚动高点
        is_high = price_position > 0.8
        pressure = np.where(is_high, rolling_high * 1.02, pressure)

        # 填充NaN
        pressure = pd.Series(pressure).ffill().fillna(close.max() * 1.1).to_numpy()

        return pressure

    def _calc_round_pressure(self, close: np.ndarray) -> np.ndarray:
        """整数关口压力 - 向量化实现"""
        # 使用向量化条件选择step
        step = np.where(
            close < 10, 0.5,
            np.where(close < 50, 1.0,
                     np.where(close < 100, 5.0, 10.0))
        )

        # 向量化计算下一个整数关口
        next_round = np.ceil(close / step) * step

        # 如果正好在关口上，上移一个step
        is_exact = np.isclose(next_round, close, rtol=1e-10)
        next_round = np.where(is_exact, next_round + step, next_round)

        return next_round

    def _calc_trapped_pressure(
            self,
            close: np.ndarray,
            high: np.ndarray,
            volume: np.ndarray
    ) -> np.ndarray:
        """套牢盘压力 (历史高点带来的卖压) - 向量化优化实现"""
        n = len(close)
        lookback = min(self.pressure_lookback, n)

        # 使用滚动窗口向量化计算
        high_series = pd.Series(high)
        vol_series = pd.Series(volume)

        # 计算滚动最高点 (历史高点)
        rolling_max = high_series.rolling(lookback, min_periods=lookback//2).max().to_numpy()

        # 使用成交量加权的平均高点作为压力
        # 向量化计算滚动成交量加权高点
        vol_high = (high_series * vol_series).rolling(lookback, min_periods=lookback//2).sum() / \
                   vol_series.rolling(lookback, min_periods=lookback//2).sum()
        vol_high = vol_high.to_numpy()

        # 压力 = max(滚动最高点, 成交量加权高点) * 1.02
        pressure = np.maximum(rolling_max, vol_high) * 1.02

        # 当当前价格接近历史高点时，压力更大
        distance_to_max = (rolling_max - close) / np.where(close > 0, close, 1)
        is_near_max = distance_to_max < 0.05  # 距离高点5%以内

        # 向量化调整
        pressure = np.where(is_near_max & (rolling_max > close),
                            rolling_max * 1.05,  # 接近高点时增加压力
                            pressure)

        # 填充NaN
        pressure = pd.Series(pressure).ffill().fillna(close.max() * 1.1).to_numpy()

        return pressure


# ============================================================================
# 综合 Alpha 因子引擎
# ============================================================================

class AlphaFactorEngineV2:
    """
    Alpha-Hunter-V2 综合因子引擎

    整合所有因子，输出最终 Alpha 信号
    """

    def __init__(self):
        self.rsrs_factor = AdaptiveRSRSFactor()
        self.surge_factor = OpeningSurgeFactor()
        self.pressure_factor = MultiLevelPressureFactor()

        self.logger = logging.getLogger("AlphaEngineV2")

    def compute(
            self,
            df: pd.DataFrame,
            market_data: pd.DataFrame = None
    ) -> AlphaFactorResult:
        """计算综合 Alpha 因子"""
        if len(df) < 100:
            return self._empty_result()

        # 1. RSRS
        rsrs_data = self.rsrs_factor.compute_full(df)

        # 2. 开盘异动
        surge_data = self.surge_factor.compute_full(df)

        # 3. 压力位
        pressure_data = self.pressure_factor.compute_full(df)

        # 4. 市场情绪
        market_state = self._parse_market_state(rsrs_data['market_state'].iloc[-1])

        # 5. 综合评分
        alpha_score = self._compute_alpha_score(rsrs_data, surge_data, pressure_data, market_state)

        # 6. 风险评分
        risk_score = self._compute_risk_score(rsrs_data, pressure_data)

        return AlphaFactorResult(
            rsrs_adaptive=float(rsrs_data['rsrs_adaptive'].iloc[-1]),
            rsrs_r2=float(rsrs_data['rsrs_r2'].iloc[-1]),
            rsrs_momentum=float(rsrs_data['rsrs_momentum'].iloc[-1]),
            opening_surge=float(surge_data['surge_score'].iloc[-1]),
            price_momentum=float(rsrs_data['rsrs_momentum'].iloc[-1]),
            volume_surge=float(surge_data['volume_surge'].iloc[-1]),
            pressure_distance=float(pressure_data['combined_pressure_dist'].iloc[-1]),
            support_distance=float(pressure_data['support_dist'].iloc[-1]),
            chip_pressure=float(pressure_data['chip_pressure'].iloc[-1]),
            market_state=market_state,
            sector_momentum=0.0,  # 需要板块数据
            alpha_score=alpha_score,
            signal_quality=float(rsrs_data['signal_quality'].iloc[-1]),
            risk_score=risk_score,
            volatility_regime=0
        )

    def _parse_market_state(self, state_str) -> MarketState:
        """解析市场状态"""
        for state in MarketState:
            if state.value == state_str:
                return state
        return MarketState.SHOCK

    def _compute_alpha_score(
            self,
            rsrs_data: pd.DataFrame,
            surge_data: pd.DataFrame,
            pressure_data: pd.DataFrame,
            market_state: MarketState
    ) -> float:
        """计算综合 Alpha 评分"""
        rsrs = rsrs_data['rsrs_adaptive'].iloc[-1]
        rsrs_qual = rsrs_data['signal_quality'].iloc[-1]
        surge = surge_data['surge_score'].iloc[-1]
        pressure_dist = pressure_data['combined_pressure_dist'].iloc[-1]
        safety = pressure_data['safety_score'].iloc[-1]

        # 处理 NaN
        rsrs = rsrs if not np.isnan(rsrs) else 0
        surge = surge if not np.isnan(surge) else 0
        pressure_dist = pressure_dist if not np.isnan(pressure_dist) else 0.05
        safety = safety if not np.isnan(safety) else 0.5

        # 基础分
        base_score = (
                0.40 * np.clip(rsrs, -2, 2) +
                0.20 * np.clip(surge, -2, 2) +
                0.25 * np.clip(pressure_dist * 10, -2, 2) +
                0.15 * np.clip(safety, 0, 1) * 2
        )

        # 市场状态调整
        state_mult = {
            MarketState.BULL_STRONG: 1.2,
            MarketState.BULL_WEAK: 1.0,
            MarketState.SHOCK: 0.8,
            MarketState.BEAR_WEAK: 0.6,
            MarketState.BEAR_STRONG: 0.4
        }

        adjusted_score = base_score * state_mult.get(market_state, 0.8)

        # 信号质量加权
        final_score = adjusted_score * (0.5 + 0.5 * rsrs_qual)

        return round(float(final_score), 4)

    def _compute_risk_score(
            self,
            rsrs_data: pd.DataFrame,
            pressure_data: pd.DataFrame
    ) -> float:
        """计算风险评分 (0-1, 越高越危险)"""
        r2 = rsrs_data['rsrs_r2'].iloc[-1]
        pressure_dist = pressure_data['combined_pressure_dist'].iloc[-1]

        r2 = r2 if not np.isnan(r2) else 0.5
        pressure_dist = pressure_dist if not np.isnan(pressure_dist) else 0.05

        # 距离压力位近 + R² 低 = 高风险
        risk = (1 - np.clip(pressure_dist * 10, 0, 1)) * 0.5 + (1 - r2) * 0.5

        return round(float(np.clip(risk, 0, 1)), 4)

    def _empty_result(self) -> AlphaFactorResult:
        return AlphaFactorResult(
            rsrs_adaptive=0, rsrs_r2=0, rsrs_momentum=0,
            opening_surge=0, price_momentum=0, volume_surge=0,
            pressure_distance=0, support_distance=0, chip_pressure=0,
            market_state=MarketState.SHOCK, sector_momentum=0,
            alpha_score=0, signal_quality=0, risk_score=1, volatility_regime=0
        )