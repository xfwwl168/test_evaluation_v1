
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# 保持原有的类定义
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SignalQuality(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class Signal:
    code: str
    signal_type: SignalType
    strength: float
    timestamp: pd.Timestamp
    price: float
    volume: float
    factor_values: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = pd.to_datetime(self.timestamp)

class SignalDeduplicator:
    """改进的信号去重处理 (基于DataFrame向量化)"""
    
    def __init__(self, similarity_threshold: float = 0.05, top_n: int = 50):
        self.similarity_threshold = similarity_threshold
        self.top_n = top_n
        self.logger = logging.getLogger("SignalDeduplicator")
    
    def deduplicate(self, 
                   signals_df: pd.DataFrame,
                   held_codes: Optional[set] = None) -> pd.DataFrame:
        """
        综合去重处理
        1. 排除已持仓代码
        2. 去除重复代码的多个信号
        3. 去除相似信号
        4. 限制信号数量
        """
        if len(signals_df) == 0:
            return signals_df
        
        df = signals_df.copy()
        
        # 步骤1：排除已持仓
        if held_codes is not None:
            df = df[~df['code'].isin(held_codes)]
        
        if len(df) == 0:
            return df
        
        # 步骤2：排序并去重（每个代码只保留评分最高的）
        # 假设df中有 'score' 列，或者使用 'strength'
        sort_col = 'score' if 'score' in df.columns else 'strength'
        if sort_col not in df.columns:
            # 如果没有分数，暂不排序
            pass
        else:
            df = df.sort_values(sort_col, ascending=False)
            
        df = df.drop_duplicates(['code'], keep='first')
        
        # 步骤3：去除相似信号
        df = self._remove_similar_signals(df)
        
        # 步骤4：限制信号数量
        df = df.head(self.top_n)
        
        return df.reset_index(drop=True)
    
    def _remove_similar_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """去除评分相似的信号"""
        sort_col = 'score' if 'score' in df.columns else 'strength'
        if len(df) < 2 or sort_col not in df.columns:
            return df
        
        try:
            # 按评分分簇（10个分簇）
            # qcut可能会因为重复值报错，使用 drop=True
            df['score_cluster'] = pd.qcut(
                df[sort_col],
                q=min(10, len(df)),
                labels=False,
                duplicates='drop'
            )
            
            # 每个簇内只保留评分最高的
            # groupby automatically sorts by group key, we want to pick max score in each cluster
            # Since we already sorted by score descending, first in cluster is max? 
            # Not necessarily if qcut reorders.
            # Safe way:
            idx = df.groupby('score_cluster')[sort_col].idxmax()
            df = df.loc[idx]
            
            df = df.drop('score_cluster', axis=1)
            
            # Re-sort
            df = df.sort_values(sort_col, ascending=False)
            
        except Exception as e:
            self.logger.warning(f"Error in similarity removal: {e}")
            pass
        
        return df

# 兼容性接口
def deduplicate_signals(signals: List[Signal], held_codes: Optional[set] = None) -> List[Signal]:
    """
    对 Signal 对象列表进行去重
    """
    if not signals:
        return []
        
    # 转换为 DataFrame
    data = []
    for s in signals:
        row = {
            'code': s.code,
            'strength': s.strength,
            'confidence': s.confidence,
            'quality_score': s.quality_score,
            'score': s.strength + s.confidence, # 合成一个score
            'obj': s
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    deduplicator = SignalDeduplicator()
    result_df = deduplicator.deduplicate(df, held_codes)
    
    return result_df['obj'].tolist()

def create_signal(*args, **kwargs) -> Signal:
    """创建信号对象"""
    return Signal(*args, **kwargs)

def get_signal_metrics(signals: List[Signal]) -> Dict[str, Any]:
    """获取信号指标"""
    return {
        'count': len(signals),
        'avg_strength': sum(s.strength for s in signals) / len(signals) if signals else 0
    }
