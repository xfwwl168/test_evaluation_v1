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
        计算自适应窗口

        高波动时缩短窗口 (更敏感)
        低波动时延长窗口 (更稳定)
        """
        n = len(close)
        window = np.full(n, self.base_window)

        # 20 日波动率
        returns = np.diff(close, prepend=close[0]) / np.clip(close, 1e-10, None)

        for i in range(30, n):
            vol_20 = returns[i - 20:i].std() * np.sqrt(252)

            # 波动率分位
            if i >= 252:
                vol_history = np.array([
                    returns[j - 20:j].std() * np.sqrt(252)
                    for j in range(30, i)
                ])
                vol_pct = (vol_20 > vol_history).mean()
            else:
                vol_pct = 0.5

            # 自适应窗口: 高波动→短窗口, 低波动→长窗口
            if vol_pct > 0.8:
                window[i] = max(12, self.base_window - 4)
            elif vol_pct > 0.6:
                window[i] = max(14, self.base_window - 2)
            elif vol_pct < 0.2:
                window[i] = min(24, self.base_window + 4)
            elif vol_pct < 0.4:
                window[i] = min(22, self.base_window + 2)

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
        鲁棒 Z-Score (使用 MAD)

        MAD = Median Absolute Deviation
        比标准差更抗异常值
        """
        n = len(arr)
        zscore = np.full(n, np.nan)

        for i in range(window, n):
            data = arr[i - window:i]
            valid_data = data[~np.isnan(data)]

            if len(valid_data) < 60:
                continue

            median = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median))

            # MAD 转换为标准差等效
            mad_std = mad * 1.4826

            if mad_std > 1e-10:
                zscore[i] = (arr[i] - median) / mad_std

        return zscore

    def _detect_market_state(
            self,
            close: np.ndarray,
            volume: np.ndarray
    ) -> np.ndarray:
        """检测市场状态"""
        n = len(close)
        states = np.full(n, MarketState.SHOCK.value, dtype=object)

        for i in range(60, n):
            # 60 日趋势
            ma60 = close[i - 60:i].mean()
            ma20 = close[i - 20:i].mean()

            price_vs_ma60 = (close[i] - ma60) / ma60
            price_vs_ma20 = (close[i] - ma20) / ma20

            # 量能
            vol_20 = volume[i - 20:i].mean()
            vol_5 = volume[i - 5:i].mean()
            vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1

            if price_vs_ma60 > 0.1 and price_vs_ma20 > 0.03:
                if vol_ratio > 1.2:
                    states[i] = MarketState.BULL_STRONG.value
                else:
                    states[i] = MarketState.BULL_WEAK.value
            elif price_vs_ma60 < -0.1 and price_vs_ma20 < -0.03:
                if vol_ratio > 1.2:
                    states[i] = MarketState.BEAR_STRONG.value
                else:
                    states[i] = MarketState.BEAR_WEAK.value
            else:
                states[i] = MarketState.SHOCK.value

        return states

    def _adaptive_skew_penalty(
            self,
            slope: np.ndarray,
            market_states: np.ndarray,
            window: int
    ) -> np.ndarray:
        """自适应偏度惩罚"""
        n = len(slope)
        penalty = np.zeros(n)

        for i in range(window, n):
            data = slope[i - window:i]
            valid_data = data[~np.isnan(data)]

            if len(valid_data) < 60:
                continue

            skewness = stats.skew(valid_data)
            state = market_states[i]

            # 根据市场状态调整惩罚系数
            if state == MarketState.BULL_STRONG.value:
                # 牛市中右偏是正常的，惩罚较轻
                base_penalty = 0.05
            elif state == MarketState.BULL_WEAK.value:
                base_penalty = 0.10
            elif state == MarketState.BEAR_STRONG.value:
                # 熊市中右偏可能是诱多，惩罚较重
                base_penalty = 0.20
            elif state == MarketState.BEAR_WEAK.value:
                base_penalty = 0.15
            else:
                base_penalty = 0.12

            # 右偏惩罚
            if skewness > 0:
                penalty[i] = min(skewness * base_penalty, 0.5)

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
            lookback: int = 250,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.mid_window = mid_window
        self.lookback = lookback

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
        """筹码压力 (成交密集区)"""
        n = len(close)
        pressure = np.full(n, np.nan)

        lookback = min(self.lookback, n)

        for i in range(lookback, n):
            window_close = close[i - lookback:i]
            window_vol = volume[i - lookback:i]
            window_high = high[i - lookback:i]

            current = close[i]

            # 价格分箱
            price_bins = np.linspace(
                window_close.min() * 0.9,
                window_high.max() * 1.1,
                50
            )

            vol_profile = np.zeros(len(price_bins) - 1)

            for j in range(len(window_close)):
                idx = np.searchsorted(price_bins, window_close[j]) - 1
                idx = max(0, min(idx, len(vol_profile) - 1))
                vol_profile[idx] += window_vol[j]

            # 找上方密集区
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            above_mask = bin_centers > current

            if above_mask.any():
                above_vol = vol_profile[above_mask]
                above_prices = bin_centers[above_mask]

                if len(above_vol) > 0 and above_vol.sum() > 0:
                    # 成交量加权价格
                    weights = above_vol / above_vol.sum()
                    pressure[i] = (above_prices * weights).sum()

        # 填充
        pressure = pd.Series(pressure).fillna(method='ffill').fillna(close.max() * 1.1).to_numpy()

        return pressure

    def _calc_round_pressure(self, close: np.ndarray) -> np.ndarray:
        """整数关口压力"""
        result = []

        for price in close:
            if price < 10:
                step = 0.5
            elif price < 50:
                step = 1.0
            elif price < 100:
                step = 5.0
            else:
                step = 10.0

            next_round = np.ceil(price / step) * step
            if next_round == price:
                next_round += step

            result.append(next_round)

        return np.array(result)

    def _calc_trapped_pressure(
            self,
            close: np.ndarray,
            high: np.ndarray,
            volume: np.ndarray
    ) -> np.ndarray:
        """套牢盘压力 (历史高点带来的卖压)"""
        n = len(close)
        pressure = np.full(n, np.nan)

        lookback = min(self.lookback, n)

        for i in range(lookback, n):
            current = close[i]

            # 找历史高点
            window_high = high[i - lookback:i]
            window_vol = volume[i - lookback:i]

            # 在当前价上方的历史高点
            above_mask = window_high > current

            if above_mask.any():
                above_highs = window_high[above_mask]
                above_vols = window_vol[above_mask]

                # 成交量加权的套牢位
                if above_vols.sum() > 0:
                    weights = above_vols / above_vols.sum()
                    pressure[i] = (above_highs * weights).sum()

        # 填充
        pressure = pd.Series(pressure).fillna(method='ffill').fillna(close.max() * 1.1).to_numpy()

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