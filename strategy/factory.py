# ============================================================================
# 文件: strategy/factory.py
# ============================================================================
"""
策略工厂 - 统一策略创建和管理

提供统一的策略创建接口，支持:
- 策略自动注册和发现
- 策略参数配置管理
- 动态策略组合
- 策略生命周期管理
"""
from typing import Dict, List, Optional, Type, Any
import logging
from dataclasses import dataclass, field
import json
from pathlib import Path
import pandas as pd

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """
    策略配置
    
    存储策略的参数和元数据
    """
    name: str                          # 策略名称
    params: Dict[str, Any] = field(default_factory=dict)  # 策略参数
    enabled: bool = True               # 是否启用
    priority: int = 0                  # 优先级
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'params': self.params,
            'enabled': self.enabled,
            'priority': self.priority,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyConfig':
        """从字典创建"""
        return cls(
            name=data['name'],
            params=data.get('params', {}),
            enabled=data.get('enabled', True),
            priority=data.get('priority', 0),
            metadata=data.get('metadata', {})
        )


class StrategyFactory:
    """
    策略工厂
    
    功能:
    - 统一策略创建接口
    - 策略配置管理
    - 策略组合管理
    - 策略元数据查询
    
    使用示例:
    ```python
    # 创建单个策略
    factory = StrategyFactory()
    strategy = factory.create('momentum', params={'top_n': 20})
    
    # 创建策略组合
    combo = factory.create_combo(['momentum', 'short_term_rsrs'])
    
    # 从配置文件加载
    factory.load_config('strategies.json')
    strategies = factory.create_all_enabled()
    ```
    """
    
    def __init__(self):
        """初始化策略工厂"""
        self._configs: Dict[str, StrategyConfig] = {}
        self._strategies: Dict[str, BaseStrategy] = {}
        self._load_default_configs()
    
    def _load_default_configs(self):
        """加载默认策略配置"""
        default_configs = {
            'momentum': StrategyConfig(
                name='momentum',
                params={
                    'lookback': 20,
                    'top_n': 20,
                    'min_momentum': 0.05,
                    'max_volatility': 0.5,
                    'use_volatility_weight': True,
                },
                priority=1,
                metadata={
                    'category': 'momentum',
                    'timeframe': 'medium',
                    'description': '动量策略 - 基于价格动量和趋势跟随'
                }
            ),
            'short_term_rsrs': StrategyConfig(
                name='short_term_rsrs',
                params={
                    'rsrs_entry_threshold': 0.7,
                    'r2_threshold': 0.8,
                    'volume_multiplier': 1.5,
                    'fixed_stop_loss': 0.03,
                    'trailing_atr_mult': 2.0,
                    'max_holding_days': 5,
                },
                priority=2,
                metadata={
                    'category': 'short_term',
                    'timeframe': 'short',
                    'description': '短线RSRS策略 - 高胜率短线交易'
                }
            ),
            'rsrs': StrategyConfig(
                name='rsrs',
                params={
                    'rsrs_threshold': 1.2,
                    'r2_threshold': 0.7,
                    'lookback': 18,
                },
                priority=1,
                metadata={
                    'category': 'trend',
                    'timeframe': 'medium',
                    'description': 'RSRS策略 - 相对强弱回归斜率'
                }
            ),
            'alpha_hunter': StrategyConfig(
                name='alpha_hunter',
                params={
                    'lookback': 20,
                    'min_alpha': 0.5,
                    'max_beta': 1.5,
                },
                priority=2,
                metadata={
                    'category': 'alpha',
                    'timeframe': 'medium',
                    'description': 'Alpha Hunter - Alpha因子选股'
                }
            ),
            'alpha_hunter_v2': StrategyConfig(
                name='alpha_hunter_v2',
                params={
                    'lookback': 20,
                    'min_rsrs': 0.8,
                    'min_r2': 0.8,
                    'top_n': 10,
                },
                priority=3,
                metadata={
                    'category': 'alpha',
                    'timeframe': 'medium',
                    'description': 'Alpha Hunter V2 - 增强版Alpha因子'
                }
            ),
        }
        
        self._configs.update(default_configs)
        logger.info(f"Loaded {len(default_configs)} default strategy configs")
    
    def register_config(self, config: StrategyConfig):
        """注册策略配置"""
        self._configs[config.name] = config
        logger.info(f"Registered config for strategy: {config.name}")
    
    def get_config(self, name: str) -> Optional[StrategyConfig]:
        """获取策略配置"""
        return self._configs.get(name)
    
    def update_config(self, name: str, params: Dict[str, Any]):
        """更新策略配置"""
        if name not in self._configs:
            self._configs[name] = StrategyConfig(name=name)
        self._configs[name].params.update(params)
        logger.info(f"Updated config for strategy: {name}")
    
    def create(self, name: str, params: Optional[Dict] = None) -> BaseStrategy:
        """
        创建策略实例
        
        Args:
            name: 策略名称
            params: 策略参数 (可选，会与配置中的参数合并)
        
        Returns:
            策略实例
        """
        # 获取策略类
        strategy_cls = StrategyRegistry.get(name)
        
        # 合并参数
        merged_params = {}
        if name in self._configs:
            merged_params.update(self._configs[name].params)
        if params:
            merged_params.update(params)
        
        # 创建策略实例
        strategy = strategy_cls(params=merged_params)
        self._strategies[name] = strategy
        
        logger.info(f"Created strategy instance: {name} with params: {merged_params}")
        return strategy
    
    def create_combo(self, strategy_names: List[str], 
                     weights: Optional[List[float]] = None) -> 'ComboStrategy':
        """
        创建策略组合
        
        Args:
            strategy_names: 策略名称列表
            weights: 权重列表 (可选，默认为均等权重)
        
        Returns:
            组合策略实例
        """
        if weights is None:
            weights = [1.0 / len(strategy_names)] * len(strategy_names)
        
        if len(weights) != len(strategy_names):
            raise ValueError("Number of weights must match number of strategies")
        
        strategies = []
        for name in strategy_names:
            strategy = self.create(name)
            strategies.append(strategy)
        
        return ComboStrategy(strategies=strategies, weights=weights)
    
    def create_all_enabled(self) -> Dict[str, BaseStrategy]:
        """创建所有启用的策略"""
        strategies = {}
        enabled_configs = [cfg for cfg in self._configs.values() if cfg.enabled]
        
        # 按优先级排序
        enabled_configs.sort(key=lambda x: x.priority, reverse=True)
        
        for config in enabled_configs:
            try:
                strategy = self.create(config.name)
                strategies[config.name] = strategy
            except Exception as e:
                logger.error(f"Failed to create strategy {config.name}: {e}")
        
        logger.info(f"Created {len(strategies)} enabled strategies")
        return strategies
    
    def list_available(self) -> List[str]:
        """列出所有可用策略"""
        return StrategyRegistry.list_all()
    
    def list_configured(self) -> List[str]:
        """列出所有已配置的策略"""
        return list(self._configs.keys())
    
    def get_info(self, name: str) -> Dict:
        """获取策略信息"""
        # 从注册表获取基本信息
        registry_info = StrategyRegistry.get_info(name)
        
        # 添加配置信息
        config = self._configs.get(name)
        if config:
            registry_info['config'] = config.to_dict()
        
        return registry_info
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        data = {
            name: config.to_dict()
            for name, config in self._configs.items()
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved strategy configs to: {filepath}")
    
    def load_config(self, filepath: str):
        """从文件加载配置"""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for name, config_data in data.items():
            self._configs[name] = StrategyConfig.from_dict(config_data)
        
        logger.info(f"Loaded strategy configs from: {filepath}")
    
    def clear_cache(self):
        """清除缓存的策略实例"""
        self._strategies.clear()
        logger.info("Cleared strategy cache")


class ComboStrategy(BaseStrategy):
    """
    组合策略
    
    将多个策略组合在一起，按权重汇总信号
    
    使用示例:
    ```python
    combo = ComboStrategy(
        strategies=[momentum, rsrs],
        weights=[0.6, 0.4]
    )
    signals = combo.generate_signals(context)
    ```
    """
    
    name = "combo"
    version = "1.0.0"
    
    def __init__(self, strategies: List[BaseStrategy], weights: List[float]):
        """
        初始化组合策略
        
        Args:
            strategies: 策略列表
            weights: 权重列表
        """
        super().__init__()
        
        if len(strategies) != len(weights):
            raise ValueError("Number of strategies must match number of weights")
        
        # 归一化权重
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.strategies = strategies
        
        self.logger.info(f"ComboStrategy initialized with {len(strategies)} sub-strategies")
    
    def initialize(self) -> None:
        """初始化所有子策略"""
        for strategy in self.strategies:
            strategy.initialize()
        super().initialize()
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算所有子策略的因子"""
        all_factors = {}
        
        for i, strategy in enumerate(self.strategies):
            factors = strategy.compute_factors(history)
            # 添加前缀避免冲突
            prefix = f"{strategy.name}_"
            for key, value in factors.items():
                all_factors[f"{prefix}{key}"] = value
        
        return all_factors
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """
        生成组合信号
        
        汇总所有子策略的信号，按权重计算最终目标仓位
        """
        from collections import defaultdict
        
        # 收集所有策略的信号
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(context)
            all_signals.append(signals)
        
        # 按股票汇总权重
        code_weights = defaultdict(float)
        code_reasons = defaultdict(list)
        
        for signals, weight in zip(all_signals, self.weights):
            for signal in signals:
                if signal.side == OrderSide.BUY:
                    code_weights[signal.code] += signal.weight * weight
                    code_reasons[signal.code].append(signal.reason)
                elif signal.side == OrderSide.SELL:
                    code_weights[signal.code] = 0.0
                    code_reasons[signal.code].append(signal.reason)
        
        # 生成最终信号
        final_signals = []
        for code, weight in code_weights.items():
            if weight > 0.001:  # 忽略极小仓位
                reasons = "; ".join(code_reasons[code])
                final_signals.append(Signal(
                    code=code,
                    side=OrderSide.BUY,
                    weight=weight,
                    reason=f"Combo: {reasons}"
                ))
            elif weight == 0.0 and code in context.positions:
                final_signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason=f"Combo: {code_reasons[code][0] if code_reasons[code] else 'Weight zero'}"
                ))
        
        return final_signals
    
    def on_order_filled(self, order: 'Order') -> None:
        """委托子策略处理订单成交"""
        for strategy in self.strategies:
            strategy.on_order_filled(order)
    
    def on_order_rejected(self, order: 'Order', reason: str) -> None:
        """委托子策略处理订单拒绝"""
        for strategy in self.strategies:
            strategy.on_order_rejected(order, reason)
    
    def on_day_end(self, context: StrategyContext) -> None:
        """委托子策略处理日终"""
        for strategy in self.strategies:
            strategy.on_day_end(context)


# 全局工厂实例
_global_factory: Optional[StrategyFactory] = None


def get_factory() -> StrategyFactory:
    """获取全局策略工厂实例"""
    global _global_factory
    if _global_factory is None:
        _global_factory = StrategyFactory()
    return _global_factory


def create_strategy(name: str, params: Optional[Dict] = None) -> BaseStrategy:
    """
    快捷创建策略
    
    Args:
        name: 策略名称
        params: 策略参数
    
    Returns:
        策略实例
    """
    return get_factory().create(name, params)
