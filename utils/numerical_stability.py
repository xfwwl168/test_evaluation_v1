# ============================================================================
# 文件: utils/numerical_stability.py
# 说明: 数值稳定性增强 - 防止极端值、NaN处理、数值溢出
# ============================================================================
"""
数值稳定性工具包

核心功能:
- 安全除法操作
- 极端值裁剪
- NaN值处理
- 数值验证和过滤
- 溢出防护

目标:
- 防止计算中的数值错误
- 确保因子计算的稳定性
- 提高系统鲁棒性
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import logging
import warnings


class NaNStrategy(Enum):
    """NaN处理策略"""
    DROP = "drop"           # 删除NaN值
    FILL_FORWARD = "ffill"  # 前向填充
    FILL_BACKWARD = "bfill" # 后向填充
    FILL_MEAN = "mean"      # 均值填充
    FILL_MEDIAN = "median" # 中位数填充
    FILL_ZERO = "zero"      # 零值填充
    INTERPOLATE = "interpolate"  # 插值填充
    KEEP = "keep"           # 保持NaN


class OutlierStrategy(Enum):
    """极端值处理策略"""
    CLIP = "clip"           # 裁剪到指定范围
    DROP = "drop"           # 删除极端值
    REPLACE_MEDIAN = "median"  # 替换为中位数
    REPLACE_MEAN = "mean"   # 替换为均值
    WINSORIZE = "winsorize" # Winsorization处理
    ZSCORE_FILTER = "zscore" # Z-Score过滤


@dataclass
class NumericalConfig:
    """数值稳定性配置"""
    # 除法安全配置
    division_epsilon: float = 1e-10
    division_max_value: float = 1e10
    division_min_value: float = -1e10
    
    # 极端值配置
    outlier_percentiles: Tuple[float, float] = (1.0, 99.0)  # (1%, 99%)
    outlier_std_threshold: float = 5.0  # Z-Score阈值
    clip_range: Optional[Tuple[float, float]] = None  # 裁剪范围
    
    # NaN处理配置
    nan_strategy: NaNStrategy = NaNStrategy.INTERPOLATE
    nan_threshold: float = 0.3  # NaN比例超过此阈值则删除
    
    # 数值验证配置
    enable_range_check: bool = True
    enable_nan_check: bool = True
    enable_inf_check: bool = True
    enable_overflow_check: bool = True
    
    # 日志配置
    log_warnings: bool = True
    log_errors: bool = True


class NumericalStability:
    """数值稳定性工具类"""
    
    # 暴露枚举供外部使用
    OutlierStrategy = OutlierStrategy
    NaNStrategy = NaNStrategy
    
    def __init__(self, config: Optional[NumericalConfig] = None):
        """
        初始化数值稳定性工具
        
        Args:
            config: 数值稳定性配置
        """
        self.config = config or NumericalConfig()
        self.logger = logging.getLogger("NumericalStability")
        
        # 统计信息
        self.stats = {
            'safe_divisions': 0,
            'clipped_values': 0,
            'nan_filled': 0,
            'outliers_removed': 0,
            'overflows_handled': 0
        }
    
    def safe_divide(
        self,
        numerator: Union[np.ndarray, pd.Series, float],
        denominator: Union[np.ndarray, pd.Series, float],
        fill_value: float = 0.0,
        epsilon: Optional[float] = None,
        max_abs_value: Optional[float] = None
    ) -> Union[np.ndarray, pd.Series, float]:
        """
        安全除法运算 (增强版：支持相对阈值和极大值检查)
        
        Args:
            numerator: 分子
            denominator: 分母
            fill_value: 分母为0或过小时的填充值
            epsilon: 最小分母阈值 (绝对)
            max_abs_value: 结果最大绝对值阈值
            
        Returns:
            除法结果
        """
        if epsilon is None:
            epsilon = self.config.division_epsilon
        if max_abs_value is None:
            max_abs_value = self.config.division_max_value
        
        try:
            # 转换为numpy数组
            if isinstance(numerator, pd.Series):
                numerator = numerator.values
            if isinstance(denominator, pd.Series):
                denominator = denominator.values
            
            numerator = np.asarray(numerator, dtype=np.float64)
            denominator = np.asarray(denominator, dtype=np.float64)
            
            # 处理标量情况
            if numerator.ndim == 0 and denominator.ndim == 0:
                abs_num = abs(numerator)
                abs_den = abs(denominator)
                relative_threshold = max(epsilon, abs_num * 1e-8)
                
                if abs_den < relative_threshold or np.isnan(denominator) or np.isinf(denominator):
                    return fill_value
                
                result = numerator / denominator
                
                # 检查溢出
                if np.isnan(result) or np.isinf(result) or abs(result) > max_abs_value:
                    return fill_value
                
                self.stats['safe_divisions'] += 1
                return result
            
            # 数组情况 - 处理广播
            if numerator.shape != denominator.shape:
                numerator, denominator = np.broadcast_arrays(numerator, denominator)
                # broadcast_arrays returns read-only views often, but we only read from them.
                # result needs to be allocated.
            
            result = np.full_like(numerator, fill_value, dtype=np.float64)
            
            abs_num = np.abs(numerator)
            abs_den = np.abs(denominator)
            
            # 相对阈值判断
            relative_threshold = np.maximum(epsilon, abs_num * 1e-8)
            
            # 有效除法位置
            valid_mask = (
                (abs_den > relative_threshold) &
                (~np.isnan(denominator)) &
                (~np.isinf(denominator)) &
                (~np.isnan(numerator)) &
                (~np.isinf(numerator))
            )
            
            if np.any(valid_mask):
                # 使用 np.errstate 忽略除法警告，因为我们已经过滤了
                with np.errstate(divide='ignore', invalid='ignore'):
                    temp_result = numerator[valid_mask] / denominator[valid_mask]
                
                # 检查结果有效性 (finite check)
                finite_mask = np.isfinite(temp_result)
                
                # 检查极大值
                extreme_mask = np.abs(temp_result) > max_abs_value
                
                # 最终有效掩码 (在valid_mask子集内)
                final_valid_in_subset = finite_mask & (~extreme_mask)
                
                # 将子集结果映射回全集
                # 注意：valid_mask是全集掩码
                # temp_result 对应 valid_mask=True 的位置
                
                # 我们需要更新 result 在 valid_mask 为 True 且 final_valid_in_subset 为 True 的位置
                
                # 可以这样做：
                # 1. 先把 temp_result 赋值给 result[valid_mask]
                # 2. 然后把不满足 final_valid_in_subset 的位置设为 fill_value
                
                current_values = result[valid_mask]
                current_values[:] = temp_result # 赋值
                
                # 处理无效值
                invalid_in_subset = ~final_valid_in_subset
                if np.any(invalid_in_subset):
                    current_values[invalid_in_subset] = fill_value
                    self.stats['overflows_handled'] += np.sum(invalid_in_subset)
                
                result[valid_mask] = current_values
                self.stats['safe_divisions'] += np.sum(final_valid_in_subset)
            
            # 保持原始类型
            if isinstance(numerator, pd.Series):
                result = pd.Series(result, index=numerator.index, name=numerator.name)
            
            return result
            
        except Exception as e:
            if self.config.log_errors:
                self.logger.error(f"Safe divide error: {str(e)}")
            return fill_value
    
    def clip_extremes(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        method: OutlierStrategy = OutlierStrategy.CLIP,
        **kwargs
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        极端值处理
        
        Args:
            data: 输入数据
            method: 处理方法
            **kwargs: 方法特定参数
            
        Returns:
            处理后的数据
        """
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame逐列处理
                result = data.copy()
                for col in data.columns:
                    result[col] = self.clip_extremes(data[col], method, **kwargs)
                return result
            
            if isinstance(data, pd.Series):
                result = data.copy()
                index = data.index
            else:
                result = np.array(data)
                index = None
            
            # 记录原始NaN数量
            original_nan_count = np.sum(np.isnan(result)) if result.ndim <= 1 else 0
            
            if method == OutlierStrategy.CLIP:
                result = self._clip_values(result, **kwargs)
                
            elif method == OutlierStrategy.WINSORIZE:
                result = self._winsorize_values(result, **kwargs)
                
            elif method == OutlierStrategy.ZSCORE_FILTER:
                result = self._zscore_filter(result, **kwargs)
                
            elif method == OutlierStrategy.REPLACE_MEDIAN:
                result = self._replace_with_median(result, **kwargs)
                
            elif method == OutlierStrategy.REPLACE_MEAN:
                result = self._replace_with_mean(result, **kwargs)
            
            # 统计裁剪数量
            if result.shape == np.array(data).shape:
                clipped_count = np.sum(np.isnan(result)) - original_nan_count
                self.stats['clipped_values'] += clipped_count
            
            if isinstance(data, pd.Series) and index is not None:
                result = pd.Series(result, index=index, name=data.name)
            
            return result
            
        except Exception as e:
            if self.config.log_errors:
                self.logger.error(f"Clip extremes error: {str(e)}")
            return data
    
    def handle_nan(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        strategy: Optional[NaNStrategy] = None,
        **kwargs
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        NaN值处理
        
        Args:
            data: 输入数据
            strategy: 处理策略
            **kwargs: 策略特定参数
            
        Returns:
            处理后的数据
        """
        if strategy is None:
            strategy = self.config.nan_strategy
        
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame逐列处理
                result = data.copy()
                for col in data.columns:
                    result[col] = self.handle_nan(data[col], strategy, **kwargs)
                return result
            
            if isinstance(data, pd.Series):
                result = data.copy()
                original_nan_count = result.isna().sum()
            else:
                result = np.array(data)
                original_nan_count = np.sum(np.isnan(result))
            
            if original_nan_count == 0:
                return result
            
            if strategy == NaNStrategy.DROP:
                result = self._drop_nan_values(result)
                
            elif strategy == NaNStrategy.FILL_FORWARD:
                result = self._fill_forward(result)
                
            elif strategy == NaNStrategy.FILL_BACKWARD:
                result = self._fill_backward(result)
                
            elif strategy == NaNStrategy.FILL_MEAN:
                result = self._fill_with_mean(result)
                
            elif strategy == NaNStrategy.FILL_MEDIAN:
                result = self._fill_with_median(result)
                
            elif strategy == NaNStrategy.FILL_ZERO:
                result = self._fill_with_zero(result)
                
            elif strategy == NaNStrategy.INTERPOLATE:
                result = self._interpolate_values(result)
            
            # 统计NaN处理数量
            if isinstance(data, pd.Series):
                new_nan_count = result.isna().sum()
            else:
                new_nan_count = np.sum(np.isnan(result))
            
            filled_count = original_nan_count - new_nan_count
            self.stats['nan_filled'] += filled_count
            
            return result
            
        except Exception as e:
            if self.config.log_errors:
                self.logger.error(f"Handle NaN error: {str(e)}")
            return data
    
    def validate_factor_values(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        factor_name: str,
        check_range: bool = True,
        check_nan: bool = True,
        check_inf: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        因子值有效性验证
        
        Args:
            data: 因子数据
            factor_name: 因子名称
            check_range: 是否检查数值范围
            check_nan: 是否检查NaN值
            check_inf: 是否检查无穷值
            
        Returns:
            验证结果字典
        """
        result = {
            'factor_name': factor_name,
            'is_valid': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            if isinstance(data, pd.Series):
                values = data.values
                index = data.index
            else:
                values = np.asarray(data)
                index = None
            
            # 基本统计
            result['stats'] = {
                'total_count': len(values),
                'nan_count': np.sum(np.isnan(values)),
                'inf_count': np.sum(np.isinf(values)),
                'min_value': np.nanmin(values),
                'max_value': np.nanmax(values),
                'mean_value': np.nanmean(values),
                'std_value': np.nanstd(values)
            }
            
            # NaN检查
            if check_nan and result['stats']['nan_count'] > 0:
                nan_ratio = result['stats']['nan_count'] / result['stats']['total_count']
                if nan_ratio > self.config.nan_threshold:
                    result['issues'].append(f"NaN比例过高: {nan_ratio:.2%}")
                    result['is_valid'] = False
                else:
                    result['issues'].append(f"存在NaN值: {result['stats']['nan_count']}个")
            
            # 无穷值检查
            if check_inf and result['stats']['inf_count'] > 0:
                result['issues'].append(f"存在无穷值: {result['stats']['inf_count']}个")
                result['is_valid'] = False
            
            # 数值范围检查
            if check_range:
                range_issues = self._check_value_range(values, factor_name)
                if range_issues:
                    result['issues'].extend(range_issues)
                    result['is_valid'] = False
            
            # 因子特定验证
            factor_issues = self._validate_factor_specific(data, factor_name)
            if factor_issues:
                result['issues'].extend(factor_issues)
                result['is_valid'] = False
            
        except Exception as e:
            result['is_valid'] = False
            result['issues'].append(f"验证过程出错: {str(e)}")
            if self.config.log_errors:
                self.logger.error(f"Factor validation error for {factor_name}: {str(e)}")
        
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0
    
    # ==================== 私有方法 ====================
    
    def _clip_values(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """裁剪极端值"""
        if data.ndim == 0:
            return data
        
        result = np.array(data)
        min_val = kwargs.get('min_val', np.nanpercentile(result, self.config.outlier_percentiles[0]))
        max_val = kwargs.get('max_val', np.nanpercentile(result, self.config.outlier_percentiles[1]))
        
        if not (np.isnan(min_val) or np.isnan(max_val)):
            result = np.clip(result, min_val, max_val)
        
        return result
    
    def _winsorize_values(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Winsorization处理"""
        if data.ndim == 0:
            return data
        
        result = np.array(data)
        lower_pct = self.config.outlier_percentiles[0]
        upper_pct = self.config.outlier_percentiles[1]
        
        try:
            lower_bound = np.nanpercentile(result, lower_pct)
            upper_bound = np.nanpercentile(result, upper_pct)
            result = np.clip(result, lower_bound, upper_bound)
        except Exception:
            pass  # 如果失败则不处理
        
        return result
    
    def _zscore_filter(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Z-Score过滤"""
        if data.ndim == 0:
            return data
        
        result = np.array(data)
        mean_val = np.nanmean(result)
        std_val = np.nanstd(result)
        
        if std_val > 0:
            z_scores = (result - mean_val) / std_val
            threshold = kwargs.get('threshold', self.config.outlier_std_threshold)
            result[np.abs(z_scores) > threshold] = np.nan
        
        return result
    
    def _replace_with_median(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """替换为中位数"""
        result = np.array(data)
        median_val = np.nanmedian(result)
        result[np.isnan(result) | np.isinf(result)] = median_val
        return result
    
    def _replace_with_mean(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """替换为均值"""
        result = np.array(data)
        mean_val = np.nanmean(result)
        result[np.isnan(result) | np.isinf(result)] = mean_val
        return result
    
    def _drop_nan_values(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """删除NaN值"""
        if isinstance(data, pd.Series):
            return data.dropna()
        else:
            return data[~np.isnan(data)]
    
    def _fill_forward(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """前向填充"""
        if isinstance(data, pd.Series):
            return data.fillna(method='ffill')
        else:
            result = np.array(data)
            mask = ~np.isnan(result)
            indices = np.arange(len(result))
            result[~mask] = np.interp(indices[~mask], indices[mask], result[mask])
            return result
    
    def _fill_backward(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """后向填充"""
        if isinstance(data, pd.Series):
            return data.fillna(method='bfill')
        else:
            result = np.array(data)
            mask = ~np.isnan(result)
            indices = np.arange(len(result))
            result[~mask] = np.interp(indices[~mask], indices[mask], result[mask])
            return result
    
    def _fill_with_mean(self, data: np.ndarray) -> np.ndarray:
        """均值填充"""
        result = np.array(data)
        mean_val = np.nanmean(result)
        result[np.isnan(result) | np.isinf(result)] = mean_val
        return result
    
    def _fill_with_median(self, data: np.ndarray) -> np.ndarray:
        """中位数填充"""
        result = np.array(data)
        median_val = np.nanmedian(result)
        result[np.isnan(result) | np.isinf(result)] = median_val
        return result
    
    def _fill_with_zero(self, data: np.ndarray) -> np.ndarray:
        """零值填充"""
        result = np.array(data)
        result[np.isnan(result) | np.isinf(result)] = 0.0
        return result
    
    def _interpolate_values(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """插值填充"""
        if isinstance(data, pd.Series):
            return data.interpolate(method='linear', limit_direction='both')
        else:
            result = np.array(data)
            mask = ~np.isnan(result)
            if np.any(mask):
                indices = np.arange(len(result))
                result[~mask] = np.interp(indices[~mask], indices[mask], result[mask])
            return result
    
    def _check_value_range(self, values: np.ndarray, factor_name: str) -> List[str]:
        """检查数值范围"""
        issues = []
        
        try:
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            
            # 因子特定的合理范围检查
            if 'rsi' in factor_name.lower():
                if min_val < 0 or max_val > 100:
                    issues.append(f"RSI值超出合理范围 [0, 100]: [{min_val:.2f}, {max_val:.2f}]")
            
            elif 'price' in factor_name.lower() or 'close' in factor_name.lower():
                if min_val <= 0:
                    issues.append(f"价格数据存在非正值: {min_val:.2f}")
                if max_val > 10000:
                    issues.append(f"价格数据可能异常: {max_val:.2f}")
            
            elif 'volume' in factor_name.lower() or 'vol' in factor_name.lower():
                if min_val < 0:
                    issues.append(f"成交量存在负值: {min_val:.0f}")
                if max_val > 1e12:
                    issues.append(f"成交量数据可能异常: {max_val:.0f}")
            
            elif 'ratio' in factor_name.lower():
                if min_val < -10 or max_val > 10:
                    issues.append(f"比率因子值超出范围 [-10, 10]: [{min_val:.2f}, {max_val:.2f}]")
            
            # 通用异常值检查
            if not np.isfinite(min_val) or not np.isfinite(max_val):
                issues.append("存在非有限数值")
            
        except Exception as e:
            issues.append(f"范围检查出错: {str(e)}")
        
        return issues
    
    def _validate_factor_specific(self, data: Union[np.ndarray, pd.Series], factor_name: str) -> List[str]:
        """因子特定验证"""
        issues = []
        
        try:
            if isinstance(data, pd.Series):
                values = data.values
            else:
                values = np.asarray(data)
            
            # RSRS特定验证
            if 'rsrs' in factor_name.lower():
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    if np.min(valid_values) < -10 or np.max(valid_values) > 10:
                        issues.append(f"RSRS值超出预期范围 [-10, 10]")
            
            # 动量因子验证
            if 'momentum' in factor_name.lower():
                if np.any(np.abs(values) > 5):
                    issues.append("动量因子值可能异常（绝对值>5）")
            
            # 波动率因子验证
            if 'volatility' in factor_name.lower():
                if np.any(values < 0):
                    issues.append("波动率因子存在负值")
                if np.any(values > 5):
                    issues.append("波动率因子值可能异常（>5）")
        
        except Exception as e:
            issues.append(f"因子特定验证出错: {str(e)}")
        
        return issues


# 全局数值稳定性实例
numerical_stability = NumericalStability()

# 便捷函数
def safe_divide(
    numerator: Union[np.ndarray, pd.Series, float],
    denominator: Union[np.ndarray, pd.Series, float],
    fill_value: float = 0.0
) -> Union[np.ndarray, pd.Series, float]:
    """安全除法"""
    return numerical_stability.safe_divide(numerator, denominator, fill_value)

def clip_extremes(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    method: OutlierStrategy = OutlierStrategy.CLIP
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """裁剪极端值"""
    return numerical_stability.clip_extremes(data, method)

def handle_nan(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    strategy: NaNStrategy = NaNStrategy.INTERPOLATE
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """处理NaN值"""
    return numerical_stability.handle_nan(data, strategy)

def validate_factor_values(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    factor_name: str
) -> Dict[str, Any]:
    """验证因子值"""
    return numerical_stability.validate_factor_values(data, factor_name)


if __name__ == "__main__":
    # 测试代码
    import warnings
    warnings.filterwarnings('ignore')
    
    # 创建测试数据
    test_data = pd.Series([1.0, 2.0, np.nan, 4.0, np.inf, 6.0, 1000.0, 8.0])
    print(f"Original data: {test_data.tolist()}")
    
    # 测试安全除法
    numerator = pd.Series([1, 2, 3, 4])
    denominator = pd.Series([1, 0, 0.5, np.nan])
    result = safe_divide(numerator, denominator)
    print(f"Safe divide result: {result.tolist()}")
    
    # 测试极端值处理
    clipped = clip_extremes(test_data)
    print(f"Clipped extremes: {clipped.tolist()}")
    
    # 测试NaN处理
    filled = handle_nan(test_data)
    print(f"NaN handled: {filled.tolist()}")
    
    # 测试因子验证
    validation = validate_factor_values(test_data, "test_factor")
    print(f"Validation result: {validation}")
    
    # 测试统计信息
    stats = numerical_stability.get_stats()
    print(f"Numerical stability stats: {stats}")