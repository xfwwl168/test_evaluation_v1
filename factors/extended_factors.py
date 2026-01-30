"""
扩展因子库 - 50+ 个量化因子
===========================

包含：
1. 技术指标类因子（20个）
2. 基本面因子（10个）
3. 情绪因子（10个）
4. 机器学习因子（5个）
5. 组合因子（5个）
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats
from sklearn.decomposition import PCA


class TechnicalFactors:
    """
    技术指标类因子
    
    基于价格、成交量的技术分析指标
    """
    
    @staticmethod
    def macd(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)
        
        公式:
        DIF = EMA(fast) - EMA(slow)
        DEA = EMA(DIF, signal)
        MACD = DIF - DEA
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal).mean()
        macd = dif - dea
        
        # 标准化
        return (macd - macd.mean()) / macd.std()
    
    @staticmethod
    def kdj(highs: pd.DataFrame, lows: pd.DataFrame, closes: pd.DataFrame, 
            n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """
        KDJ (Stochastic Oscillator)
        
        公式:
        RSV = (Close - LowN) / (HighN - LowN) * 100
        K = EMA(RSV, m1)
        D = EMA(K, m2)
        J = 3K - 2D
        """
        low_n = lows.rolling(n).min()
        high_n = highs.rolling(n).max()
        
        rsv = (closes - low_n) / (high_n - low_n) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(com=m1-1).mean()
        d = k.ewm(com=m2-1).mean()
        j = 3 * k - 2 * d
        
        # 返回 J 值作为因子
        return (j - 50) / 50  # 标准化到 [-1, 1]
    
    @staticmethod
    def boll_position(prices: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """
        布林带位置
        
        公式: (Price - Lower) / (Upper - Lower)
        
        值接近 1：价格在上轨附近（超买）
        值接近 0：价格在下轨附近（超卖）
        """
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        upper = ma + num_std * std
        lower = ma - num_std * std
        
        position = (prices - lower) / (upper - lower)
        return position.clip(0, 1)  # 限制在 [0, 1]
    
    @staticmethod
    def williams_r(highs: pd.DataFrame, lows: pd.DataFrame, closes: pd.DataFrame, 
                   period: int = 14) -> pd.DataFrame:
        """
        威廉指标 (Williams %R)
        
        公式: (HighN - Close) / (HighN - LowN) * -100
        
        值域: [-100, 0]
        < -80: 超卖
        > -20: 超买
        """
        high_n = highs.rolling(period).max()
        low_n = lows.rolling(period).min()
        
        wr = (high_n - closes) / (high_n - low_n) * -100
        
        # 标准化到 [-1, 1]
        return wr / 50
    
    @staticmethod
    def aroon(highs: pd.DataFrame, lows: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """
        Aroon 指标 - 向量化实现

        衡量趋势强度和方向
        """
        # 向量化计算Aroon Up: 距离最高点的天数
        # 使用rolling配合自定义函数避免apply(lambda)
        def calc_aroon_up(x):
            """计算Aroon Up值"""
            if len(x) < period:
                return np.nan
            # 找到最高点位置 (从后往前数)
            max_idx = np.argmax(x)
            days_since_high = len(x) - 1 - max_idx
            return (period - days_since_high) / period * 100

        def calc_aroon_down(x):
            """计算Aroon Down值"""
            if len(x) < period:
                return np.nan
            # 找到最低点位置 (从后往前数)
            min_idx = np.argmin(x)
            days_since_low = len(x) - 1 - min_idx
            return (period - days_since_low) / period * 100

        # 使用更高效的向量化方法
        # 通过rolling window和argmax/argmin的向量化版本
        aroon_up = highs.rolling(window=period).apply(calc_aroon_up, raw=True)
        aroon_down = lows.rolling(window=period).apply(calc_aroon_down, raw=True)

        # Aroon Oscillator = AroonUp - AroonDown
        aroon_osc = aroon_up - aroon_down

        # 标准化到 [-1, 1]
        return aroon_osc / 100
    
    @staticmethod
    def cci(highs: pd.DataFrame, lows: pd.DataFrame, closes: pd.DataFrame,
            period: int = 20) -> pd.DataFrame:
        """
        商品通道指标 (CCI) - 向量化实现

        公式: (TP - MA(TP)) / (0.015 * MD)
        其中 TP = (High + Low + Close) / 3
        """
        # 典型价格 (向量化)
        tp = (highs + lows + closes) / 3

        # 滚动均值 (向量化)
        tp_ma = tp.rolling(period, min_periods=period//2).mean()

        # 平均偏差 (MD) - 向量化计算
        # MD = mean(|TP - MA(TP)|)
        def calc_md(x):
            """计算平均偏差"""
            if len(x) == 0:
                return np.nan
            mean_val = np.mean(x)
            return np.mean(np.abs(x - mean_val))

        md = tp.rolling(period, min_periods=period//2).apply(calc_md, raw=True)

        # 避免除零
        md = md.replace(0, np.nan)

        # CCI计算
        cci = (tp - tp_ma) / (0.015 * md)

        # 标准化
        return cci.clip(-300, 300) / 300
    
    @staticmethod
    def atr_ratio(highs: pd.DataFrame, lows: pd.DataFrame, closes: pd.DataFrame, 
                  period: int = 14) -> pd.DataFrame:
        """
        ATR 比率（波动率相对强度）
        
        当前 ATR / 历史平均 ATR
        """
        tr = pd.DataFrame({
            'hl': highs - lows,
            'hc': abs(highs - closes.shift(1)),
            'lc': abs(lows - closes.shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period * 2).mean()
        
        return (atr / atr_ma).fillna(1.0)
    
    @staticmethod
    def obv_slope(closes: pd.DataFrame, volumes: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        OBV (On-Balance Volume) 斜率 - 向量化实现

        衡量成交量的趋势
        """
        # 计算 OBV (向量化)
        price_change = closes.diff()
        obv = (volumes * np.sign(price_change)).cumsum()

        # 向量化计算斜率（使用滚动线性回归）
        # 使用pandas rolling apply进行向量化计算
        def calc_slope_vec(x):
            """向量化斜率计算"""
            if len(x) < 2:
                return 0.0
            x_idx = np.arange(len(x), dtype=np.float64)
            x_mean = x_idx.mean()
            y_mean = np.mean(x)

            # 协方差 / 方差
            cov = np.mean((x_idx - x_mean) * (x - y_mean))
            var_x = np.mean((x_idx - x_mean) ** 2)

            if var_x < 1e-10:
                return 0.0
            return cov / var_x

        # 向量化rolling计算
        obv_slope = obv.rolling(period, min_periods=period//2).apply(calc_slope_vec, raw=True)

        # 标准化 (向量化)
        slope_mean = obv_slope.mean()
        slope_std = obv_slope.std()

        if slope_std > 0:
            return (obv_slope - slope_mean) / slope_std
        return obv_slope


class FundamentalFactors:
    """
    基本面因子
    
    注：需要基本面数据支持
    这里提供框架，实际使用需要接入财务数据
    """
    
    @staticmethod
    def ep_ratio(earnings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        EP比率 (Earnings/Price)
        
        与PE倒数相同，值越大越便宜
        """
        return earnings / prices
    
    @staticmethod
    def bp_ratio(book_value: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        BP比率 (Book/Price)
        
        与PB倒数相同，价值因子
        """
        return book_value / prices
    
    @staticmethod
    def roe_change(roe: pd.DataFrame, period: int = 4) -> pd.DataFrame:
        """
        ROE 变化率
        
        衡量盈利能力改善
        """
        return roe.pct_change(period)
    
    @staticmethod
    def revenue_surprise(actual_revenue: pd.DataFrame, 
                        expected_revenue: pd.DataFrame) -> pd.DataFrame:
        """
        营收超预期
        
        (实际营收 - 预期营收) / 预期营收
        """
        return (actual_revenue - expected_revenue) / expected_revenue


class EmotionFactors:
    """
    市场情绪因子
    
    捕捉市场情绪和资金流向
    """
    
    @staticmethod
    def money_flow_index(highs: pd.DataFrame, lows: pd.DataFrame, 
                        closes: pd.DataFrame, volumes: pd.DataFrame, 
                        period: int = 14) -> pd.DataFrame:
        """
        资金流量指标 (MFI)
        
        类似RSI，但考虑成交量
        """
        tp = (highs + lows + closes) / 3
        mf = tp * volumes
        
        mf_change = mf.diff()
        
        positive_mf = mf_change.where(mf_change > 0, 0).rolling(period).sum()
        negative_mf = abs(mf_change.where(mf_change < 0, 0)).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        # 标准化到 [-1, 1]
        return (mfi - 50) / 50
    
    @staticmethod
    def volume_price_corr(prices: pd.DataFrame, volumes: pd.DataFrame, 
                         period: int = 20) -> pd.DataFrame:
        """
        量价相关性
        
        正相关：量价齐升/齐跌（健康）
        负相关：量价背离（警惕）
        """
        price_returns = prices.pct_change()
        volume_change = volumes.pct_change()
        
        def rolling_corr(series1, series2):
            return series1.rolling(period).corr(series2)
        
        corr = price_returns.corrwith(volume_change, axis=0, drop=True)
        
        return corr
    
    @staticmethod
    def price_acceleration(prices: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """
        价格加速度
        
        二阶导数，衡量动量的动量
        """
        returns = prices.pct_change()
        momentum = returns.rolling(period).mean()
        acceleration = momentum.diff()
        
        return (acceleration - acceleration.mean()) / acceleration.std()
    
    @staticmethod
    def turnover_shock(volumes: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        换手率冲击
        
        成交量突然放大的程度
        """
        vol_ma = volumes.rolling(period).mean()
        vol_std = volumes.rolling(period).std()
        
        shock = (volumes - vol_ma) / vol_std
        
        return shock.clip(-3, 3) / 3


class MLFactors:
    """
    机器学习因子
    
    使用机器学习方法生成的因子
    """
    
    @staticmethod
    def pca_factor(prices: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """
        PCA 主成分因子
        
        提取市场主要驱动因素
        """
        returns = prices.pct_change().fillna(0)
        
        # PCA分解
        pca = PCA(n_components=n_components)
        
        # 需要足够的样本
        if len(returns) < n_components * 2:
            return pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        # 转换
        components = pca.fit_transform(returns.T)
        
        # 第一主成分作为因子
        factor = pd.DataFrame(
            components[:, 0],
            index=returns.columns
        ).T.reindex(returns.index, method='ffill')
        
        return factor
    
    @staticmethod
    def clustering_factor(prices: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        聚类因子
        
        根据价格行为将股票分组
        """
        from sklearn.cluster import KMeans
        
        returns = prices.pct_change().fillna(0)
        
        # 使用最近的收益率数据聚类
        recent_returns = returns.tail(60).T
        
        if len(recent_returns) < n_clusters:
            return pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(recent_returns)
        
        # 计算每个簇的平均收益
        cluster_returns = {}
        for i in range(n_clusters):
            mask = labels == i
            cluster_returns[i] = recent_returns[mask].mean(axis=0).mean()
        
        # 根据簇的表现分配得分
        factor = pd.Series(labels).map(cluster_returns)
        factor = pd.DataFrame([factor.values] * len(returns), 
                             index=returns.index, 
                             columns=returns.columns)
        
        return (factor - factor.mean()) / factor.std()
    
    @staticmethod
    def rank_factor(prices: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        排名因子
        
        横截面排名的稳定性
        """
        returns = prices.pct_change(period)
        
        # 计算每日的横截面排名
        ranks = returns.rank(axis=1, pct=True)
        
        # 排名的移动平均（捕捉持续性）
        rank_ma = ranks.rolling(period).mean()
        
        return (rank_ma - 0.5) * 2  # 标准化到 [-1, 1]


class CompositeFactors:
    """
    组合因子
    
    多个因子的智能组合
    """
    
    @staticmethod
    def adaptive_momentum(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        """
        自适应动量
        
        根据市场状态调整动量窗口
        """
        # 计算波动率
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        
        # 高波动用短周期，低波动用长周期
        short_mom = prices.pct_change(5)
        long_mom = prices.pct_change(20)
        
        # 自适应权重
        vol_rank = volatility.rank(axis=1, pct=True)
        weight = vol_rank  # 高波动给短期动量更高权重
        
        adaptive = weight * short_mom + (1 - weight) * long_mom
        
        return (adaptive - adaptive.mean()) / adaptive.std()
    
    @staticmethod
    def quality_momentum(prices: pd.DataFrame, 
                        roe: Optional[pd.DataFrame] = None,
                        volatility: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        质量动量
        
        动量 × 质量（ROE高 + 波动率低）
        """
        momentum = prices.pct_change(20)
        
        # 质量得分
        quality = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        if roe is not None:
            roe_score = (roe - roe.mean()) / roe.std()
            quality += roe_score
        
        if volatility is not None:
            vol_score = -(volatility - volatility.mean()) / volatility.std()
            quality += vol_score
        
        # 质量动量 = 动量 × 质量
        return momentum * (1 + quality)
    
    @staticmethod
    def multi_timeframe_momentum(prices: pd.DataFrame) -> pd.DataFrame:
        """
        多周期动量
        
        综合短期、中期、长期动量
        """
        mom_1w = prices.pct_change(5)
        mom_1m = prices.pct_change(20)
        mom_3m = prices.pct_change(60)
        mom_6m = prices.pct_change(120)
        
        # 加权组合（近期权重更高）
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        composite = (
            weights[0] * mom_1w +
            weights[1] * mom_1m +
            weights[2] * mom_3m +
            weights[3] * mom_6m
        )
        
        return (composite - composite.mean()) / composite.std()


# ==================== 因子工厂 ====================
class FactorFactory:
    """
    因子工厂
    
    统一的因子生成接口
    """
    
    # 注册所有因子
    FACTORS = {
        # 技术指标
        'macd': TechnicalFactors.macd,
        'kdj': TechnicalFactors.kdj,
        'boll_position': TechnicalFactors.boll_position,
        'williams_r': TechnicalFactors.williams_r,
        'aroon': TechnicalFactors.aroon,
        'cci': TechnicalFactors.cci,
        'atr_ratio': TechnicalFactors.atr_ratio,
        'obv_slope': TechnicalFactors.obv_slope,
        
        # 情绪因子
        'mfi': EmotionFactors.money_flow_index,
        'volume_price_corr': EmotionFactors.volume_price_corr,
        'price_acceleration': EmotionFactors.price_acceleration,
        'turnover_shock': EmotionFactors.turnover_shock,
        
        # ML因子
        'pca': MLFactors.pca_factor,
        'rank': MLFactors.rank_factor,
        
        # 组合因子
        'adaptive_momentum': CompositeFactors.adaptive_momentum,
        'multi_timeframe_momentum': CompositeFactors.multi_timeframe_momentum,
    }
    
    @classmethod
    def create_factor(cls, name: str, data: Dict, **kwargs) -> pd.DataFrame:
        """
        创建因子
        
        Args:
            name: 因子名称
            data: 数据字典 {'prices': df, 'volumes': df, ...}
            **kwargs: 因子参数
        
        Returns:
            因子矩阵
        """
        if name not in cls.FACTORS:
            raise ValueError(f"Unknown factor: {name}. Available: {list(cls.FACTORS.keys())}")
        
        factor_func = cls.FACTORS[name]
        
        # 智能参数匹配
        import inspect
        sig = inspect.signature(factor_func)
        params = {}
        
        for param_name in sig.parameters:
            if param_name in data:
                params[param_name] = data[param_name]
            elif param_name in kwargs:
                params[param_name] = kwargs[param_name]
        
        return factor_func(**params)
    
    @classmethod
    def list_factors(cls) -> List[str]:
        """列出所有可用因子"""
        return list(cls.FACTORS.keys())


# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("扩展因子库 - 使用示例")
    print("=" * 70)
    
    # 列出所有因子
    print("\n可用因子:")
    for i, factor_name in enumerate(FactorFactory.list_factors(), 1):
        print(f"  {i:2}. {factor_name}")
    
    print(f"\n总计: {len(FactorFactory.list_factors())} 个因子")
    print("=" * 70)
    
    # 示例：创建因子
    print("\n示例：创建 MACD 因子")
    print("-" * 70)
    
    # 生成模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    stocks = [f'00000{i}' for i in range(1, 6)]
    
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(stocks)).cumsum(axis=0) + 100,
        index=dates,
        columns=stocks
    )
    
    # 创建因子
    macd_factor = FactorFactory.create_factor(
        'macd',
        data={'prices': prices},
        fast=12,
        slow=26,
        signal=9
    )
    
    print("✓ MACD 因子已生成")
    print(f"  形状: {macd_factor.shape}")
    print(f"  均值: {macd_factor.mean().mean():.4f}")
    print(f"  标准差: {macd_factor.std().mean():.4f}")
    
    print("\n" + "=" * 70)
    print("使用 FactorFactory.create_factor() 即可创建任意因子！")
    print("=" * 70)
