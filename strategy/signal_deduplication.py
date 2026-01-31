# ============================================================================
# 文件: strategy/signal_deduplication.py
# 说明: 信号去重系统 - 减少50%虚假信号，避免过度交易
# ============================================================================
"""
信号去重和优化系统

核心功能:
- 综合信号去重算法
- 相似信号检测和合并
- 持仓状态管理
- 信号质量评分
- 交易成本优化

目标:
- 减少50%虚假信号
- 避免过度交易
- 提高信号质量
- 优化交易成本
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import math

from utils.numerical_stability import numerical_stability


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalQuality(Enum):
    """信号质量等级"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Signal:
    """交易信号"""
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
        """后处理初始化"""
        if isinstance(self.timestamp, str):
            self.timestamp = pd.to_datetime(self.timestamp)


@dataclass
class Position:
    """持仓信息"""
    code: str
    quantity: float
    avg_price: float
    entry_time: pd.Timestamp
    last_update: pd.Timestamp
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class SignalDeduplicationConfig:
    """信号去重配置"""
    # 时间窗口
    time_window_minutes: int = 30  # 30分钟内的相似信号合并
    min_time_gap_seconds: int = 300  # 最小时间间隔（5分钟）
    
    # 价格相似度阈值
    price_similarity_threshold: float = 0.02  # 2%价格差异内认为相似
    min_strength_difference: float = 0.1     # 强度差异阈值
    
    # 信号质量过滤
    min_quality_score: float = 0.6
    min_confidence: float = 0.5
    
    # 持仓约束
    max_positions: int = 20
    position_size_limit: float = 0.1  # 单个股票最大仓位10%
    
    # 交易成本优化
    min_trade_amount: float = 1000.0  # 最小交易金额
    transaction_cost_rate: float = 0.001  # 交易成本率
    
    # 去重算法参数
    clustering_similarity: float = 0.7  # 聚类相似度
    diversity_factor: float = 0.3      # 多样性因子


class SignalDeduplicator:
    """信号去重器"""
    
    def __init__(self, config: SignalDeduplicationConfig):
        """
        初始化信号去重器
        
        Args:
            config: 信号去重配置
        """
        self.config = config
        self.logger = logging.getLogger("SignalDeduplicator")
        
        # 信号存储
        self.signals_history: List[Signal] = []
        self.active_positions: Dict[str, Position] = {}
        
        # 统计信息
        self.stats = {
            'total_signals': 0,
            'deduplicated_signals': 0,
            'high_quality_signals': 0,
            'trades_executed': 0,
            'false_signals_reduced': 0
        }
    
    def deduplicate(self, signals: List[Signal]) -> List[Signal]:
        """
        核心去重方法
        
        Args:
            signals: 原始信号列表
            
        Returns:
            去重后的信号列表
        """
        if not signals:
            return []
        
        start_time = time.time()
        self.stats['total_signals'] += len(signals)
        
        # 1. 基础过滤
        filtered_signals = self._basic_filter(signals)
        
        # 2. 信号质量评分
        quality_signals = self._score_signal_quality(filtered_signals)
        
        # 3. 去除相似信号
        unique_signals = self._remove_similar_signals(quality_signals)
        
        # 4. 持仓约束检查
        final_signals = self._apply_position_constraints(unique_signals)
        
        # 5. 更新统计
        deduplicated_count = len(signals) - len(final_signals)
        self.stats['deduplicated_signals'] += deduplicated_count
        self.stats['high_quality_signals'] += len(final_signals)
        
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Deduplication: {len(signals)} -> {len(final_signals)} "
            f"(removed {deduplicated_count}, {elapsed_time:.3f}s)"
        )
        
        # 保存到历史
        self.signals_history.extend(final_signals)
        
        return final_signals
    
    def update_positions(self, signals: List[Signal]) -> Dict[str, Position]:
        """
        更新持仓状态
        
        Args:
            signals: 执行后的信号列表
            
        Returns:
            更新后的持仓字典
        """
        for signal in signals:
            if signal.signal_type == SignalType.BUY:
                self._add_position(signal)
            elif signal.signal_type == SignalType.SELL:
                self._close_position(signal)
        
        return self.active_positions.copy()
    
    def get_signal_metrics(self) -> Dict[str, Any]:
        """获取信号指标"""
        total = self.stats['total_signals']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'deduplication_rate': self.stats['deduplicated_signals'] / total,
            'quality_rate': self.stats['high_quality_signals'] / max(1, total - self.stats['deduplicated_signals']),
            'active_positions': len(self.active_positions),
            'signal_efficiency': self.stats['high_quality_signals'] / max(1, total)
        }
    
    def clear_history(self):
        """清空历史信号"""
        self.signals_history.clear()
        self.logger.info("Signal history cleared")
    
    # ==================== 私有方法 ====================
    
    def _basic_filter(self, signals: List[Signal]) -> List[Signal]:
        """基础信号过滤"""
        filtered = []
        
        for signal in signals:
            # 基础有效性检查
            if not self._is_signal_valid(signal):
                continue
            
            # 时间窗口过滤
            if self._is_duplicate_time(signal):
                continue
            
            # 强度阈值过滤
            if signal.strength < self.config.min_strength_difference:
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def _score_signal_quality(self, signals: List[Signal]) -> List[Signal]:
        """信号质量评分"""
        for signal in signals:
            # 计算质量评分
            quality_score = self._calculate_quality_score(signal)
            signal.quality_score = quality_score
            
            # 计算置信度
            confidence = self._calculate_confidence(signal)
            signal.confidence = confidence
        
        # 过滤低质量信号
        quality_threshold = max(self.config.min_quality_score, 0.3)
        return [s for s in signals if s.quality_score >= quality_threshold]
    
    def _remove_similar_signals(self, signals: List[Signal]) -> List[Signal]:
        """去除相似信号"""
        if len(signals) <= 1:
            return signals
        
        # 按强度排序（保留强度更高的信号）
        signals_sorted = sorted(signals, key=lambda s: s.strength, reverse=True)
        
        unique_signals = []
        
        for signal in signals_sorted:
            # 检查是否与已保留的信号相似
            is_similar = False
            for unique_signal in unique_signals:
                if self._are_signals_similar(signal, unique_signal):
                    is_similar = True
                    break
            
            if not is_similar:
                unique_signals.append(signal)
        
        return unique_signals
    
    def _apply_position_constraints(self, signals: List[Signal]) -> List[Signal]:
        """应用持仓约束"""
        # 检查仓位限制
        valid_signals = []
        
        for signal in signals:
            # 检查是否超出最大仓位
            if signal.code in self.active_positions:
                current_position = self.active_positions[signal.code]
                if signal.signal_type == SignalType.BUY:
                    # 计算新仓位
                    new_quantity = current_position.quantity + signal.volume
                    # 这里需要知道总资产来计算仓位比例
                    # 暂时跳过此检查
                    pass
            
            # 检查交易金额限制
            trade_amount = signal.price * signal.volume
            if trade_amount < self.config.min_trade_amount:
                continue
            
            valid_signals.append(signal)
        
        # 限制信号数量
        if len(valid_signals) > self.config.max_positions:
            # 选择质量评分最高的信号
            valid_signals.sort(key=lambda s: s.quality_score, reverse=True)
            valid_signals = valid_signals[:self.config.max_positions]
        
        return valid_signals
    
    def _is_signal_valid(self, signal: Signal) -> bool:
        """检查信号有效性"""
        try:
            # 基本字段检查
            if not signal.code or not signal.signal_type:
                return False
            
            if signal.price <= 0 or signal.volume <= 0:
                return False
            
            if pd.isna(signal.timestamp):
                return False
            
            # 强度检查
            if not -10 <= signal.strength <= 10:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_duplicate_time(self, signal: Signal) -> bool:
        """检查时间重复"""
        current_time = signal.timestamp
        
        # 检查最近的时间窗口内是否有相似信号
        time_threshold = current_time - pd.Timedelta(minutes=self.config.time_window_minutes)
        
        for history_signal in self.signals_history[-50:]:  # 只检查最近的50个信号
            if (history_signal.code == signal.code and 
                abs((current_time - history_signal.timestamp).total_seconds()) < self.config.min_time_gap_seconds):
                return True
        
        return False
    
    def _calculate_quality_score(self, signal: Signal) -> float:
        """计算信号质量评分"""
        score = 0.0
        
        # 1. 信号强度权重 (30%)
        strength_score = min(abs(signal.strength) / 5.0, 1.0) * 0.3
        
        # 2. 置信度权重 (25%)
        confidence_score = signal.confidence * 0.25
        
        # 3. 价格合理性权重 (20%)
        price_score = self._calculate_price_reasonableness(signal) * 0.2
        
        # 4. 因子一致性权重 (15%)
        factor_score = self._calculate_factor_consistency(signal) * 0.15
        
        # 5. 时机合理性权重 (10%)
        timing_score = self._calculate_timing_score(signal) * 0.1
        
        score = strength_score + confidence_score + price_score + factor_score + timing_score
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_confidence(self, signal: Signal) -> float:
        """计算信号置信度"""
        confidence = 0.5  # 基础置信度
        
        # 1. 信号强度贡献
        if signal.strength > 2:
            confidence += 0.3
        elif signal.strength > 1:
            confidence += 0.2
        elif signal.strength > 0:
            confidence += 0.1
        
        # 2. 因子一致性贡献
        if signal.factor_values:
            consistency = self._calculate_factor_consistency(signal)
            confidence += consistency * 0.3
        
        # 3. 价格趋势贡献
        if signal.metadata.get('price_trend', 0) > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_price_reasonableness(self, signal: Signal) -> float:
        """计算价格合理性"""
        # 基于成交量和价格关系判断合理性
        volume = signal.volume
        price = signal.price
        
        # 简化的合理性评分
        if volume > 0 and price > 0:
            # 价格和成交量的合理性关系
            score = min(volume / (price * 1000), 1.0)  # 归一化
            return score
        
        return 0.5
    
    def _calculate_factor_consistency(self, signal: Signal) -> float:
        """计算因子一致性"""
        if not signal.factor_values:
            return 0.5
        
        # 检查多个因子的一致性
        factors = signal.factor_values
        
        # 简化的一致性检查
        consistency_score = 0.5
        
        # 如果有多个因子，检查它们的方向一致性
        positive_factors = sum(1 for v in factors.values() if v > 0)
        total_factors = len(factors)
        
        if total_factors > 0:
            consistency_ratio = positive_factors / total_factors
            if consistency_ratio > 0.7 or consistency_ratio < 0.3:
                consistency_score = 0.8  # 高一致性
            else:
                consistency_score = 0.6  # 中等一致性
        
        return consistency_score
    
    def _calculate_timing_score(self, signal: Signal) -> float:
        """计算时机评分"""
        # 基于当前时间的市场活跃度评分
        current_time = signal.timestamp
        
        # 简化的时机评分（实际应该基于市场数据）
        hour = current_time.hour
        minute = current_time.minute
        
        # 交易时段评分
        if 9 <= hour <= 11 or 13 <= hour <= 15:
            base_score = 0.8  # 正常交易时段
        elif hour == 9 and minute < 30:
            base_score = 0.6  # 开盘时段
        else:
            base_score = 0.3  # 非交易时段
        
        return base_score
    
    def _are_signals_similar(self, signal1: Signal, signal2: Signal) -> bool:
        """判断两个信号是否相似"""
        # 1. 时间相似度
        time_diff = abs((signal1.timestamp - signal2.timestamp).total_seconds())
        time_similar = time_diff < self.config.min_time_gap_seconds
        
        # 2. 价格相似度
        price_diff = abs(signal1.price - signal2.price) / max(signal1.price, signal2.price)
        price_similar = price_diff < self.config.price_similarity_threshold
        
        # 3. 强度相似度
        strength_diff = abs(signal1.strength - signal2.strength)
        strength_similar = strength_diff < self.config.min_strength_difference
        
        # 4. 信号类型相同
        type_similar = signal1.signal_type == signal2.signal_type
        
        # 5. 相同股票
        code_similar = signal1.code == signal2.code
        
        # 综合判断
        return (time_similar and price_similar and strength_similar and type_similar and code_similar)
    
    def _add_position(self, signal: Signal):
        """添加持仓"""
        if signal.code in self.active_positions:
            # 增加持仓
            position = self.active_positions[signal.code]
            total_quantity = position.quantity + signal.volume
            total_cost = position.avg_price * position.quantity + signal.price * signal.volume
            
            position.quantity = total_quantity
            position.avg_price = total_cost / total_quantity
            position.last_update = signal.timestamp
        else:
            # 新建持仓
            position = Position(
                code=signal.code,
                quantity=signal.volume,
                avg_price=signal.price,
                entry_time=signal.timestamp,
                last_update=signal.timestamp
            )
            self.active_positions[signal.code] = position
    
    def _close_position(self, signal: Signal):
        """平仓"""
        if signal.code not in self.active_positions:
            return
        
        position = self.active_positions[signal.code]
        
        # 计算盈亏
        pnl = (signal.price - position.avg_price) * min(signal.volume, position.quantity)
        
        position.realized_pnl += pnl
        position.quantity -= signal.volume
        position.last_update = signal.timestamp
        
        # 如果全部平仓，移除持仓
        if position.quantity <= 0:
            del self.active_positions[signal.code]


class PortfolioManager:
    """组合管理器"""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        初始化组合管理器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.daily_returns: List[float] = []
        self.logger = logging.getLogger("PortfolioManager")
    
    def get_portfolio_value(self) -> float:
        """获取组合价值"""
        total_value = self.current_capital
        
        for position in self.positions.values():
            # 这里应该根据市场价格计算当前价值
            # 简化处理，使用持仓成本
            position_value = position.avg_price * position.quantity
            total_value += position_value
        
        return total_value
    
    def get_position_weights(self) -> Dict[str, float]:
        """获取持仓权重"""
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return {}
        
        weights = {}
        for code, position in self.positions.items():
            position_value = position.avg_price * position.quantity
            weights[code] = position_value / total_value
        
        return weights
    
    def check_risk_constraints(self) -> Dict[str, bool]:
        """检查风险约束"""
        constraints = {
            'max_single_position': True,
            'max_total_exposure': True,
            'max_concentration': True
        }
        
        weights = self.get_position_weights()
        
        # 检查单个仓位限制
        for weight in weights.values():
            if weight > 0.2:  # 20%单股限制
                constraints['max_single_position'] = False
                break
        
        # 检查总持仓
        total_exposure = sum(weights.values())
        if total_exposure > 0.8:  # 80%持仓限制
            constraints['max_total_exposure'] = False
        
        # 检查集中度
        if len(weights) > 0:
            max_weight = max(weights.values())
            if max_weight > 0.3:  # 30%集中度限制
                constraints['max_concentration'] = False
        
        return constraints


# 全局信号去重器实例
signal_deduplicator = SignalDeduplicator(SignalDeduplicationConfig())
portfolio_manager = PortfolioManager()

# 便捷函数
def deduplicate_signals(signals: List[Signal]) -> List[Signal]:
    """去重信号"""
    return signal_deduplicator.deduplicate(signals)

def update_portfolio_positions(signals: List[Signal]) -> Dict[str, Position]:
    """更新组合持仓"""
    return signal_deduplicator.update_positions(signals)

def get_signal_metrics() -> Dict[str, Any]:
    """获取信号指标"""
    return signal_deduplicator.get_signal_metrics()

def create_signal(
    code: str,
    signal_type: SignalType,
    strength: float,
    timestamp: pd.Timestamp,
    price: float,
    volume: float,
    **kwargs
) -> Signal:
    """创建信号"""
    return Signal(
        code=code,
        signal_type=signal_type,
        strength=strength,
        timestamp=timestamp,
        price=price,
        volume=volume,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试信号
    test_signals = [
        create_signal(
            code="000001",
            signal_type=SignalType.BUY,
            strength=2.5,
            timestamp=pd.Timestamp.now(),
            price=10.0,
            volume=1000,
            factor_values={"rsi": 70, "momentum": 0.8}
        ),
        create_signal(
            code="000001",
            signal_type=SignalType.BUY,
            strength=2.3,
            timestamp=pd.Timestamp.now() + pd.Timedelta(minutes=10),
            price=10.05,
            volume=1000,
            factor_values={"rsi": 72, "momentum": 0.7}
        )
    ]
    
    print("Testing Signal Deduplication...")
    print(f"Original signals: {len(test_signals)}")
    
    # 去重测试
    deduplicated = deduplicate_signals(test_signals)
    print(f"Deduplicated signals: {len(deduplicated)}")
    
    # 获取指标
    metrics = get_signal_metrics()
    print(f"Signal metrics: {metrics}")
    
    # 创建多个不同信号进行测试
    diverse_signals = [
        create_signal("000001", SignalType.BUY, 3.0, pd.Timestamp.now(), 10.0, 1000),
        create_signal("000002", SignalType.BUY, 2.5, pd.Timestamp.now(), 15.0, 800),
        create_signal("000003", SignalType.SELL, -2.0, pd.Timestamp.now(), 20.0, 500),
        create_signal("000001", SignalType.BUY, 2.8, pd.Timestamp.now() + pd.Timedelta(minutes=15), 10.02, 1000)
    ]
    
    print(f"\nTesting diverse signals: {len(diverse_signals)}")
    final_signals = deduplicate_signals(diverse_signals)
    print(f"Final signals after deduplication: {len(final_signals)}")
    
    # 更新持仓
    positions = update_portfolio_positions(final_signals)
    print(f"Active positions: {len(positions)}")