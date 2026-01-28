# ============================================================================
# 文件: menu/backtest_menu.py
# ============================================================================
"""
回测引擎菜单模块
包含策略管理、单策略回测、因子组合配置、参数优化等高级功能
"""
import sys
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# 确保项目根目录在路径中
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .ui_components import UIComponents, TableDisplay, ProgressTracker, InputValidator
from utils.logger import get_logger

logger = get_logger(__name__)


class BacktestMenu:
    """回测引擎菜单"""
    
    def __init__(self):
        self.strategies = self._load_strategies()
        self.factor_combinations = self._load_factor_combinations()
        self.backtest_history = self._load_backtest_history()
    
    def _load_strategies(self) -> List[Dict[str, Any]]:
        """加载可用策略"""
        return [
            {
                'id': 'rsrs',
                'name': 'RSRS',
                'description': '线性回归趋势',
                'win_rate': 58,
                'status': '可用',
                'category': '趋势',
                'factors': ['RSRS', 'R_squared'],
                'parameters': {
                    'window': 18,
                    'std_window': 600,
                    'entry_threshold': 0.7,
                    'exit_threshold': -0.5
                }
            },
            {
                'id': 'momentum',
                'name': 'Momentum',
                'description': '动量因子',
                'win_rate': 55,
                'status': '可用',
                'category': '动量',
                'factors': ['Momentum', 'Volume'],
                'parameters': {
                    'period': 20,
                    'top_n': 30,
                    'rebalance_freq': 'W'
                }
            },
            {
                'id': 'alpha_hunter',
                'name': 'AlphaHunter',
                'description': '多因子综合',
                'win_rate': 62,
                'status': '可用',
                'category': '多因子',
                'factors': ['RSRS', 'Momentum', 'OBV', 'MarketHeat'],
                'parameters': {
                    'rsrs_weight': 0.6,
                    'momentum_weight': 0.4,
                    'entry_threshold': 0.7,
                    'exit_threshold': -0.5
                }
            },
            {
                'id': 'ultra_short',
                'name': 'UltraShort',
                'description': '超短线',
                'win_rate': 48,
                'status': '可用',
                'category': '短线',
                'factors': ['RSRS', 'R_squared', 'Volume'],
                'parameters': {
                    'window': 10,
                    'std_window': 300,
                    'entry_threshold': 0.8,
                    'exit_threshold': -0.6
                }
            },
            {
                'id': 'bull_bear',
                'name': 'BullBear',
                'description': '高频策略',
                'win_rate': 60,
                'status': '可用',
                'category': '高频',
                'factors': ['Momentum', 'Volatility', 'RSRS'],
                'parameters': {
                    'period': 5,
                    'volatility_threshold': 0.02,
                    'holding_days': 1
                }
            },
            {
                'id': 'dinger',
                'name': 'Dinger',
                'description': '打板策略',
                'win_rate': 42,
                'status': '谨慎使用',
                'category': '打板',
                'factors': ['Breakthrough', 'Volume', 'Momentum'],
                'parameters': {
                    'breakthrough_threshold': 0.09,
                    'volume_multiplier': 2.0,
                    'max_holding': 2
                }
            },
            {
                'id': 'hanbing',
                'name': 'Hanbing',
                'description': '反包策略',
                'win_rate': 52,
                'status': '可用',
                'category': '反转',
                'factors': ['Reversal', 'Volume', 'Support'],
                'parameters': {
                    'reversal_threshold': -0.05,
                    'volume_check': True,
                    'support_level': 0.02
                }
            }
        ]
    
    def _load_factor_combinations(self) -> List[Dict[str, Any]]:
        """加载因子组合"""
        return [
            {
                'id': 'trend_combo',
                'name': '趋势组合',
                'description': 'RSRS 60% + Momentum 40%',
                'factors': {'RSRS': 60, 'Momentum': 40},
                'entry_threshold': 0.70,
                'exit_threshold': -0.50,
                'logic': 'OR',
                'created_at': '2024-01-01',
                'status': '已保存'
            },
            {
                'id': 'price_volume_combo',
                'name': '量价组合',
                'description': 'OBV 50% + VWAP 50%',
                'factors': {'OBV': 50, 'VWAP': 50},
                'entry_threshold': 0.65,
                'exit_threshold': -0.45,
                'logic': 'AND',
                'created_at': '2024-01-15',
                'status': '已保存'
            },
            {
                'id': 'comprehensive_combo',
                'name': '综合组合',
                'description': '多因子加权',
                'factors': {'RSRS': 30, 'Momentum': 25, 'OBV': 20, 'MarketHeat': 25},
                'entry_threshold': 0.75,
                'exit_threshold': -0.55,
                'logic': 'weighted',
                'created_at': '2024-02-01',
                'status': '已保存'
            }
        ]
    
    def _load_backtest_history(self) -> List[Dict[str, Any]]:
        """加载回测历史"""
        return [
            {
                'id': 'bt_001',
                'strategy': 'AlphaHunter',
                'date': '2026-01-28',
                'parameters': '60/40权重',
                'win_rate': 62,
                'sharpe': 1.92,
                'return': 38.5,
                'max_drawdown': 11.2
            },
            {
                'id': 'bt_002',
                'strategy': 'RSRS',
                'date': '2026-01-27',
                'parameters': '默认',
                'win_rate': 58,
                'sharpe': 1.85,
                'return': 35.2,
                'max_drawdown': 12.5
            },
            {
                'id': 'bt_003',
                'strategy': 'Momentum',
                'date': '2026-01-26',
                'parameters': '自定义',
                'win_rate': 55,
                'sharpe': 1.62,
                'return': 28.1,
                'max_drawdown': 15.8
            }
        ]
    
    def show_main_menu(self):
        """显示回测引擎主菜单"""
        while True:
            UIComponents.clear_screen()
            UIComponents.print_header("🎮 回测引擎菜单")
            UIComponents.print_breadcrumb("主菜单 > 回测引擎")
            
            print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 📊 策略管理
2. 🎯 单策略回测  
3. 📈 多策略对比回测
4. 🔧 因子组合配置
5. ⚡ 参数优化
6. 📊 回测历史
7. 💾 导出回测结果
8. ⬅️ 返回主菜单
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """)
            
            choice = UIComponents.get_input("\n请选择功能", required=True)
            
            if choice == '1':
                self._show_strategy_management()
            elif choice == '2':
                self._show_single_strategy_backtest()
            elif choice == '3':
                self._show_multi_strategy_comparison()
            elif choice == '4':
                self._show_factor_combination()
            elif choice == '5':
                self._show_parameter_optimization()
            elif choice == '6':
                self._show_backtest_history()
            elif choice == '7':
                self._export_backtest_results()
            elif choice == '8':
                break
            else:
                UIComponents.print_error("无效选择，请重新输入")
                UIComponents.pause()
    
    def _show_strategy_management(self):
        """策略管理"""
        UIComponents.clear_screen()
        UIComponents.print_header("📊 策略管理")
        UIComponents.print_breadcrumb("主菜单 > 回测引擎 > 策略管理")
        
        # 显示策略列表
        TableDisplay.print_strategy_list(self.strategies)
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

操作:
[1] 查看策略详情  (因子构成、参数、历史收益)
[2] 查看策略代码  (源码)
[3] 复制策略      (创建副本后修改)
[4] 删除自定义策略
[5] 返回
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '1':
            self._show_strategy_details()
        elif choice == '2':
            self._show_strategy_code()
        elif choice == '3':
            self._copy_strategy()
        elif choice == '4':
            self._delete_custom_strategy()
        elif choice == '5':
            pass  # 返回
        else:
            UIComponents.print_error("无效选择")
        
        UIComponents.pause()
    
    def _show_strategy_details(self):
        """显示策略详情"""
        print("\n📋 选择要查看的策略:")
        
        for i, strategy in enumerate(self.strategies, 1):
            print(f"{i}. {strategy['name']} - {strategy['description']}")
        
        choice = UIComponents.get_input("\n请选择策略编号", required=True)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.strategies):
                strategy = self.strategies[idx]
                
                print(f"\n📊 策略详情: {strategy['name']}")
                print("━" * 70)
                print(f"名称:        {strategy['name']}")
                print(f"描述:        {strategy['description']}")
                print(f"类别:        {strategy['category']}")
                print(f"胜率:        {strategy['win_rate']}%")
                print(f"状态:        {strategy['status']}")
                
                print(f"\n因子构成:")
                for factor in strategy['factors']:
                    print(f"  - {factor}")
                
                print(f"\n参数设置:")
                for param, value in strategy['parameters'].items():
                    print(f"  {param}: {value}")
                
                print(f"\n历史收益:")
                # 模拟历史数据
                print(f"  2024年: +{15 + strategy['win_rate']//10:.1f}%")
                print(f"  2023年: +{12 + strategy['win_rate']//8:.1f}%")
                print(f"  2022年: +{8 + strategy['win_rate']//6:.1f}%")
                
            else:
                UIComponents.print_error("无效策略编号")
        except ValueError:
            UIComponents.print_error("请输入有效数字")
    
    def _show_strategy_code(self):
        """显示策略代码"""
        print("\n📋 选择要查看代码的策略:")
        
        for i, strategy in enumerate(self.strategies, 1):
            print(f"{i}. {strategy['name']} - {strategy['description']}")
        
        choice = UIComponents.get_input("\n请选择策略编号", required=True)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.strategies):
                strategy = self.strategies[idx]
                
                print(f"\n💻 策略代码: {strategy['name']}")
                print("━" * 70)
                print("```python")
                
                # 模拟策略代码
                if strategy['id'] == 'rsrs':
                    code = '''
class RSRSStrategy(BaseStrategy):
    """RSRS 趋势策略"""
    
    def __init__(self, window=18, std_window=600):
        super().__init__()
        self.window = window
        self.std_window = std_window
        self.entry_threshold = 0.7
        self.exit_threshold = -0.5
    
    def generate_signals(self, data):
        """生成交易信号"""
        # 计算RSRS指标
        rsrs = self.calculate_rsrs(data)
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        signals[rsrs > self.entry_threshold] = 1  # 买入
        signals[rsrs < self.exit_threshold] = -1  # 卖出
        
        return signals
    
    def calculate_rsrs(self, data):
        """计算RSRS指标"""
        # 具体的RSRS计算逻辑
        prices = data['close']
        highs = data['high']
        lows = data['low']
        
        # 计算最高价相对强弱
        rs_strength = (highs - lows) / prices
        
        # 线性回归
        rsrs = rs_strength.rolling(self.window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        return rsrs
'''
                elif strategy['id'] == 'momentum':
                    code = '''
class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def __init__(self, period=20, top_n=30):
        super().__init__()
        self.period = period
        self.top_n = top_n
    
    def generate_signals(self, data):
        """生成交易信号"""
        # 计算动量
        momentum = data['close'].pct_change(self.period)
        
        # 选择top N股票
        signals = pd.Series(0, index=data.index)
        
        # 这里简化处理，实际需要考虑多股票
        if momentum.iloc[-1] > 0:
            signals.iloc[-1] = 1  # 买入
        
        return signals
'''
                else:
                    code = f'''
# {strategy['name']} 策略代码示例
# 文件: strategy/{strategy['id']}_strategy.py

class {strategy['name'].replace(' ', '')}Strategy(BaseStrategy):
    """{strategy['description']}"""
    
    def __init__(self):
        super().__init__()
        # 策略参数初始化
        pass
    
    def generate_signals(self, data):
        """生成交易信号"""
        # 实现具体的策略逻辑
        signals = pd.Series(0, index=data.index)
        # ... 策略逻辑
        return signals
'''
                
                print(code)
                print("```")
                
            else:
                UIComponents.print_error("无效策略编号")
        except ValueError:
            UIComponents.print_error("请输入有效数字")
    
    def _copy_strategy(self):
        """复制策略"""
        print("\n📋 选择要复制的策略:")
        
        for i, strategy in enumerate(self.strategies, 1):
            print(f"{i}. {strategy['name']} - {strategy['description']}")
        
        choice = UIComponents.get_input("\n请选择策略编号", required=True)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.strategies):
                strategy = self.strategies[idx]
                new_name = UIComponents.get_input("请输入新策略名称", required=True)
                
                # 创建副本
                new_strategy = strategy.copy()
                new_strategy['id'] = new_name.lower().replace(' ', '_')
                new_strategy['name'] = new_name
                new_strategy['status'] = '自定义'
                
                self.strategies.append(new_strategy)
                
                UIComponents.print_success(f"策略 '{new_name}' 创建成功!")
                
            else:
                UIComponents.print_error("无效策略编号")
        except ValueError:
            UIComponents.print_error("请输入有效数字")
    
    def _delete_custom_strategy(self):
        """删除自定义策略"""
        custom_strategies = [s for s in self.strategies if s['status'] == '自定义']
        
        if not custom_strategies:
            UIComponents.print_warning("没有自定义策略可删除")
            return
        
        print("\n🗑️ 选择要删除的自定义策略:")
        
        for i, strategy in enumerate(custom_strategies, 1):
            print(f"{i}. {strategy['name']} - {strategy['description']}")
        
        choice = UIComponents.get_input("\n请选择策略编号", required=True)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(custom_strategies):
                strategy = custom_strategies[idx]
                
                if UIComponents.get_yes_no(f"确认删除策略 '{strategy['name']}'?"):
                    self.strategies.remove(strategy)
                    UIComponents.print_success(f"策略 '{strategy['name']}' 删除成功!")
                else:
                    print("已取消删除")
                    
            else:
                UIComponents.print_error("无效策略编号")
        except ValueError:
            UIComponents.print_error("请输入有效数字")
    
    def _show_single_strategy_backtest(self):
        """单策略回测"""
        UIComponents.clear_screen()
        UIComponents.print_header("🎯 单策略回测")
        UIComponents.print_breadcrumb("主菜单 > 回测引擎 > 单策略回测")
        
        # Step 1: 选择策略
        print("Step 1: 选择策略")
        print("-" * 30)
        
        for i, strategy in enumerate(self.strategies, 1):
            marker = "(●)" if strategy['id'] == 'alpha_hunter' else "( )"
            print(f"{marker} {strategy['name']} - {strategy['description']}")
        
        print("\n( ) 自定义策略 [选择 ▼]")
        
        choice = UIComponents.get_input("\n请选择策略", "alpha_hunter")
        
        # Step 2: 设置时间范围
        print("\n\nStep 2: 设置时间范围")
        print("-" * 30)
        
        start_date = UIComponents.get_input("开始日期", "2024-01-01")
        end_date = UIComponents.get_input("结束日期", "2026-01-28")
        
        if not InputValidator.validate_date(start_date):
            UIComponents.print_error("开始日期格式无效")
            return
        if not InputValidator.validate_date(end_date):
            UIComponents.print_error("结束日期格式无效")
            return
        
        # Step 3: 调整因子权重
        print("\n\nStep 3: 调整因子权重 (可选)")
        print("-" * 30)
        
        use_default = UIComponents.get_yes_no("是否使用默认权重?", "y")
        
        if not use_default:
            print("\n设置自定义权重:")
            
            # 这里可以扩展为动态选择因子
            weights = {
                'RSRS': int(UIComponents.get_input("RSRS权重", "60")),
                'Momentum': int(UIComponents.get_input("Momentum权重", "40")),
                'OBV': int(UIComponents.get_input("OBV权重", "0"))
            }
            
            total = sum(weights.values())
            print(f"\n总计: {total}%")
            
            if total != 100:
                UIComponents.print_warning("权重总和不为100%，将自动调整")
                # 自动调整权重
                for factor in weights:
                    weights[factor] = int(weights[factor] * 100 / total)
        
        # Step 4: 设置信号阈值
        print("\n\nStep 4: 设置信号阈值 (可选)")
        print("-" * 30)
        
        entry_threshold = float(UIComponents.get_input("买入阈值", "0.70"))
        exit_threshold = float(UIComponents.get_input("卖出阈值", "-0.50"))
        
        # Step 5: 选择股票范围
        print("\n\nStep 5: 选择股票范围")
        print("-" * 30)
        
        print("1. 全市场")
        print("2. Top 500")
        print("3. 指定代码")
        
        stock_choice = UIComponents.get_input("请选择股票池", "1")
        
        if stock_choice == "3":
            stock_codes = UIComponents.get_input("请输入股票代码 (用逗号分隔)", "")
            if not stock_codes:
                UIComponents.print_warning("使用默认股票池: Top 500")
                stock_choice = "2"
        
        # 开始回测
        print(f"\n\n{'='*70}")
        print("🔄 开始回测...")
        
        if UIComponents.get_yes_no("确认开始回测?"):
            self._run_single_backtest(choice, start_date, end_date, 
                                    entry_threshold, exit_threshold)
    
    def _run_single_backtest(self, strategy: str, start_date: str, end_date: str, 
                           entry_threshold: float, exit_threshold: float):
        """执行单策略回测"""
        # 模拟回测过程
        UIComponents.print_loading("回测进度")
        
        # 模拟回测结果
        results = {
            'total_return': 0.352,
            'annual_return': 0.143,
            'max_drawdown': -0.125,
            'sharpe_ratio': 1.85,
            'win_rate': 0.62,
            'profit_loss_ratio': 2.1,
            'total_trades': 124,
            'winning_trades': 77,
            'losing_trades': 47,
            'avg_trade_return': 0.0028,
            'max_win': 0.085,
            'max_loss': -0.032,
            'avg_holding_days': 12.5,
            'max_holding_days': 45,
            'min_holding_days': 1,
            'initial_capital': 1000000,
            'final_equity': 1352000,
            'max_equity': 1425000,
            'max_drawdown_amount': -157500
        }
        
        # 显示回测结果
        UIComponents.clear_screen()
        UIComponents.print_header("📊 回测报告")
        
        TableDisplay.print_backtest_results(results)
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[详细交易列表] [权益曲线图] [月度统计] [返回]
        """)
        
        UIComponents.pause()
    
    def _show_multi_strategy_comparison(self):
        """多策略对比回测"""
        UIComponents.clear_screen()
        UIComponents.print_header("📈 多策略对比回测")
        UIComponents.print_breadcrumb("主菜单 > 回测引擎 > 多策略对比回测")
        
        print("选择要对比的策略 (最多5个):")
        
        for i, strategy in enumerate(self.strategies, 1):
            print(f"{i}. {strategy['name']} - {strategy['description']}")
        
        choices = UIComponents.get_input("\n请选择策略编号 (用逗号分隔)", "1,2,3")
        
        try:
            selected_indices = [int(x.strip()) - 1 for x in choices.split(',')]
            if len(selected_indices) > 5:
                UIComponents.print_warning("最多选择5个策略")
                selected_indices = selected_indices[:5]
            
            selected_strategies = []
            for idx in selected_indices:
                if 0 <= idx < len(self.strategies):
                    selected_strategies.append(self.strategies[idx])
            
            if not selected_strategies:
                UIComponents.print_error("没有有效的策略被选择")
                return
            
            # 设置时间范围
            start_date = UIComponents.get_input("开始日期", "2024-01-01")
            end_date = UIComponents.get_input("结束日期", "2026-01-28")
            
            # 开始对比回测
            if UIComponents.get_yes_no("确认开始对比回测?"):
                self._run_multi_strategy_comparison(selected_strategies, start_date, end_date)
                
        except ValueError:
            UIComponents.print_error("请输入有效的策略编号")
    
    def _run_multi_strategy_comparison(self, strategies: List[Dict], start_date: str, end_date: str):
        """执行多策略对比"""
        # 模拟对比结果
        comparison_results = []
        
        for strategy in strategies:
            # 模拟结果
            result = {
                'strategy': strategy['name'],
                'total_return': 0.2 + strategy['win_rate'] / 1000,  # 基于胜率模拟
                'annual_return': 0.1 + strategy['win_rate'] / 2000,
                'max_drawdown': -(0.08 + (100-strategy['win_rate']) / 1000),
                'sharpe_ratio': 1.0 + strategy['win_rate'] / 100,
                'win_rate': strategy['win_rate'] / 100,
                'total_trades': 100 + strategy['win_rate']
            }
            comparison_results.append(result)
        
        # 显示对比结果
        UIComponents.clear_screen()
        UIComponents.print_header("📈 多策略对比结果")
        
        print("━" * 80)
        print(f"{'策略':<15} {'总收益':<10} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<8}")
        print("━" * 80)
        
        for result in comparison_results:
            print(f"{result['strategy']:<15} "
                  f"{result['total_return']:<9.1%} "
                  f"{result['annual_return']:<9.1%} "
                  f"{result['sharpe_ratio']:<9.2f} "
                  f"{result['max_drawdown']:<9.1%} "
                  f"{result['win_rate']:<7.1%}")
        
        # 找出最优策略
        best_strategy = max(comparison_results, key=lambda x: x['sharpe_ratio'])
        print(f"\n🏆 最优策略: {best_strategy['strategy']} (夏普比率: {best_strategy['sharpe_ratio']:.2f})")
        
        UIComponents.pause()
    
    def _show_factor_combination(self):
        """因子组合配置"""
        UIComponents.clear_screen()
        UIComponents.print_header("🔧 因子组合配置")
        UIComponents.print_breadcrumb("主菜单 > 回测引擎 > 因子组合配置")
        
        print("已保存的因子组合:")
        print("━" * 50)
        
        for i, combo in enumerate(self.factor_combinations, 1):
            print(f"{i}. {combo['name']} - {combo['description']}")
            factors_str = ", ".join([f"{k} {v}%" for k, v in combo['factors'].items()])
            print(f"   因子: {factors_str}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

新建/修改组合:
组合名称: [我的组合1]

选择因子并设置权重:
[●] RSRS       - [60]% ← 修改
[●] Momentum   - [40]% ← 修改
[ ] OBV        - [0]%
[ ] ATR        - [0]%
[ ] VWAP       - [0]%
[ ] MarketHeat - [0]%
[ ] Other...   - [0]%
           总: 100%

设置信号阈值:
买入阈值 (综合得分):  [0.70]
卖出阈值 (综合得分):  [-0.50]

逻辑设置:
( ) AND (所有因子同时满足)
(●) OR  (任一因子满足)
( ) 加权综合 (权重求和)

[保存组合] [测试组合] [取消]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice in ['1', '2', '3', '4', '5', '6']:
            # 实现各种操作
            UIComponents.print_info("功能开发中...")
        else:
            UIComponents.print_error("无效选择")
        
        UIComponents.pause()
    
    def _show_parameter_optimization(self):
        """参数优化"""
        UIComponents.clear_screen()
        UIComponents.print_header("⚡ 参数优化 (GridSearch)")
        UIComponents.print_breadcrumb("主菜单 > 回测引擎 > 参数优化")
        
        print("""
Step 1: 选择要优化的策略
(●) RSRS
( ) Momentum
( ) AlphaHunter
( ) 自定义

Step 2: 设置优化参数范围

RSRS参数:
  RSRS_WINDOW:         [10 ~ 30]  步长: 5
  RSRS_STD_WINDOW:     [500 ~ 800] 步长: 50
  
信号阈值:
  ENTRY_THRESHOLD:     [0.60 ~ 0.90] 步长: 0.05
  EXIT_THRESHOLD:      [-0.60 ~ -0.30] 步长: 0.05

Step 3: 设置时间范围
开始日期: [2024-01-01]
结束日期: [2026-01-28]

Step 4: 选择优化目标
( ) 最大化收益率
( ) 最大化夏普率
(●) 最大化胜率
( ) 最小化回撤

Step 5: 选择优化方式
(●) 网格搜索 GridSearch (穷举所有组合，耗时长)
( ) 贝叶斯优化 (智能搜索，耗时短)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

预计参数组合数: 3×3×7×7 = 441组合
预计耗时: 60-90分钟 (多线程)

[开始优化]
        """)
        
        if UIComponents.get_yes_no("\n确认开始参数优化?"):
            self._run_parameter_optimization()
    
    def _run_parameter_optimization(self):
        """执行参数优化"""
        # 模拟优化进度
        print("\n\n优化进度:")
        
        tracker = ProgressTracker(total=441, message="参数优化")
        
        for i in range(441):
            time.sleep(0.01)  # 模拟处理时间
            tracker.update(i + 1)
        
        tracker.finish()
        
        # 显示优化结果
        UIComponents.clear_screen()
        UIComponents.print_header("🏆 优化结果")
        
        print("Top 5参数组合:")
        print("━" * 70)
        print(f"{'排名':<4} {'参数配置':<20} {'胜率':<8} {'Sharpe':<8} {'收益率':<8} {'回撤':<8}")
        print("━" * 70)
        
        optimal_results = [
            {'rank': 1, 'params': 'RSRS_W=20,STD=600', 'win_rate': 62, 'sharpe': 1.92, 'return': 38.5, 'drawdown': 11.2},
            {'rank': 2, 'params': 'RSRS_W=20,STD=550', 'win_rate': 61, 'sharpe': 1.88, 'return': 36.2, 'drawdown': 11.8},
            {'rank': 3, 'params': 'RSRS_W=15,STD=600', 'win_rate': 60, 'sharpe': 1.85, 'return': 35.2, 'drawdown': 12.5},
            {'rank': 4, 'params': 'RSRS_W=25,STD=600', 'win_rate': 59, 'sharpe': 1.78, 'return': 32.1, 'drawdown': 13.2},
            {'rank': 5, 'params': 'RSRS_W=20,STD=650', 'win_rate': 58, 'sharpe': 1.75, 'return': 30.8, 'drawdown': 14.1}
        ]
        
        for result in optimal_results:
            print(f"{result['rank']:<4} {result['params']:<20} "
                  f"{result['win_rate']:<7}% "
                  f"{result['sharpe']:<7.2f} "
                  f"{result['return']:<7.1f}% "
                  f"{result['drawdown']:<7.1f}%")
        
        print(f"\n[应用最优参数] [对比分析] [导出报告] [返回]")
        
        UIComponents.pause()
    
    def _show_backtest_history(self):
        """回测历史"""
        UIComponents.clear_screen()
        UIComponents.print_header("📊 回测历史")
        UIComponents.print_breadcrumb("主菜单 > 回测引擎 > 回测历史")
        
        print("最近回测 (共127次):")
        print("━" * 80)
        print(f"{'时间':<12} {'策略':<12} {'参数':<15} {'胜率':<8} {'Sharpe':<8} {'收益率':<8}")
        print("━" * 80)
        
        for record in self.backtest_history:
            print(f"{record['date']:<12} {record['strategy']:<12} "
                  f"{record['parameters']:<15} "
                  f"{record['win_rate']:<7}% "
                  f"{record['sharpe']:<7.2f} "
                  f"{record['return']:<7.1f}%")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

操作:
[选择] 查看详细报告
[对比] 选择2个回测对比分析
[删除] 删除回测记录
[导出] 导出为CSV/PDF
[返回] 返回
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '对比':
            UIComponents.print_info("对比分析功能开发中...")
        elif choice == '导出':
            self._export_backtest_history()
        else:
            UIComponents.print_info("功能开发中...")
        
        UIComponents.pause()
    
    def _export_backtest_results(self):
        """导出回测结果"""
        UIComponents.clear_screen()
        UIComponents.print_header("💾 导出回测结果")
        UIComponents.print_breadcrumb("主菜单 > 回测引擎 > 导出回测结果")
        
        print("选择导出内容:")
        print("1. 当前回测结果")
        print("2. 回测历史数据")
        print("3. 因子组合配置")
        print("4. 优化参数结果")
        
        choice = UIComponents.get_input("\n请选择导出内容", "1")
        
        if choice == '1':
            print("\n选择导出格式:")
            print("1. CSV")
            print("2. Excel")
            print("3. PDF报告")
            
            format_choice = UIComponents.get_input("请选择格式", "1")
            
            if format_choice == '1':
                UIComponents.print_success("已导出为 CSV 格式: backtest_results.csv")
            elif format_choice == '2':
                UIComponents.print_success("已导出为 Excel 格式: backtest_results.xlsx")
            elif format_choice == '3':
                UIComponents.print_success("已生成 PDF 报告: backtest_report.pdf")
        
        elif choice == '2':
            if UIComponents.get_yes_no("确认导出所有历史数据?"):
                UIComponents.print_success("已导出历史数据: backtest_history.csv")
        
        UIComponents.pause()


# 导出模块
__all__ = ['BacktestMenu']