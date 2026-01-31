
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional

def calc_rsi_correct(close: np.ndarray, period: int = 14) -> np.ndarray:
    """标准RSI - 使用SMMA而非SMA"""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    result = np.full(len(close), np.nan)
    
    # 初始SMA（第period个）
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # SMMA公式：(prev * (period-1) + current) / period
    for i in range(period, len(close)):
        if i == period:
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
            else:
                result[i] = 100
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            if avg_loss > 1e-10:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
            else:
                result[i] = 100
    
    return result

def calc_atr_correct(high: np.ndarray, low: np.ndarray, 
                     close: np.ndarray, period: int = 14) -> np.ndarray:
    """标准ATR - 使用SMMA"""
    if len(close) < period:
        return np.full(len(close), np.nan)
        
    # 计算True Range
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    # Note: abs(high - prev_close) calculation needs alignment
    
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0] # First TR is just high - low usually
    
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, max(tr2, tr3))
    
    atr = np.full(len(close), np.nan)
    
    # Initial ATR is mean of TR
    atr[period-1] = np.mean(tr[:period])
    
    # SMMA平滑
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def calc_bb_correct(close: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """布林带正确实现"""
    s = pd.Series(close)
    sma = s.rolling(period).mean().values
    std = s.rolling(period).std().values
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower

def calc_macd_correct(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD正确实现"""
    s = pd.Series(close)
    ema_fast = s.ewm(span=fast, adjust=False).mean().values
    ema_slow = s.ewm(span=slow, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
    macd = (dif - dea) * 2
    return dif, dea, macd
