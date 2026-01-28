# ============================================================================
# 文件: menu/ui_components.py
# ============================================================================
"""
UI 通用组件库
提供统一的用户界面组件和交互逻辑
"""
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pandas as pd

# 确保项目根目录在路径中
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


class UIComponents:
    """UI组件库"""
    
    @staticmethod
    def clear_screen():
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_header(title: str, width: int = 70):
        """打印头部"""
        print("=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
    
    @staticmethod
    def print_breadcrumb(location: str, width: int = 70):
        """打印面包屑导航"""
        print(f"📍 当前位置: {location}")
        print("-" * width)
    
    @staticmethod
    def print_subheader(title: str, width: int = 70):
        """打印子标题"""
        print(f"\n{title}")
        print("━" * width)
    
    @staticmethod
    def print_success(message: str):
        """打印成功信息"""
        print(f"✅ {message}")
    
    @staticmethod
    def print_warning(message: str):
        """打印警告信息"""
        print(f"⚠️  {message}")
    
    @staticmethod
    def print_error(message: str):
        """打印错误信息"""
        print(f"❌ {message}")
    
    @staticmethod
    def print_info(message: str):
        """打印信息"""
        print(f"ℹ️  {message}")
    
    @staticmethod
    def print_loading(message: str = "处理中", width: int = 70):
        """打印加载动画"""
        chars = "|/-\\"
        for i in range(20):
            print(f"\r{message} {chars[i % len(chars)]}", end="", flush=True)
            time.sleep(0.1)
        print("\n")
    
    @staticmethod
    def print_progress(current: int, total: int, message: str = "进度", width: int = 50):
        """打印进度条"""
        if total == 0:
            percentage = 0
        else:
            percentage = current / total
        
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        
        print(f"\r{message}: |{bar}| {percentage:.1%} ({current}/{total})", end="", flush=True)
        
        if current == total:
            print()  # 换行
    
    @staticmethod
    def pause():
        """暂停等待"""
        input("\n按 Enter 继续...")
    
    @staticmethod
    def get_input(prompt: str, default: str = None, required: bool = False) -> str:
        """获取用户输入"""
        if default:
            user_input = input(f"{prompt} (默认: {default}): ").strip()
            return user_input if user_input else default
        else:
            while True:
                user_input = input(f"{prompt}: ").strip()
                if user_input or not required:
                    return user_input
                print("❌ 输入不能为空，请重新输入")
    
    @staticmethod
    def get_choice(prompt: str, choices: List[str], allow_back: bool = True) -> int:
        """获取选择"""
        print(f"\n{prompt}")
        
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")
        
        if allow_back:
            print(f"  0. 返回")
        
        while True:
            try:
                choice = int(input("\n请选择 (数字): "))
                if 0 <= choice <= len(choices):
                    return choice
                else:
                    print(f"❌ 请输入 0-{len(choices)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效数字")
    
    @staticmethod
    def get_yes_no(prompt: str, default: str = None) -> bool:
        """获取是否选择"""
        if default:
            user_input = input(f"{prompt} (y/n, 默认: {default}): ").strip().lower()
            return user_input == 'y'
        else:
            while True:
                user_input = input(f"{prompt} (y/n): ").strip().lower()
                if user_input in ['y', 'n']:
                    return user_input == 'y'
                print("❌ 请输入 y 或 n")


class MenuDisplay:
    """菜单显示组件"""
    
    @staticmethod
    def print_main_menu():
        """打印主菜单"""
        print("""
🎯 量化交易引擎 v3.0 (Option A 完整版)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 回测引擎                    📡 实盘监控                   📈 市场分析
  1. 策略管理                   8. 策略配置                 15. 因子有效性分析
  2. 单策略回测                 9. 实时全市场扫描            16. 行业对比分析
  3. 多策略对比回测            10. 跟踪单只股票              17. 单只股票深度分析
  4. 因子组合配置              11. 买入信号热力图           18. 因子排名 (Top 100)
  5. 参数优化                  12. 卖出信号列表              19. 行业板块分析
  6. 回测历史                 13. 持仓管理                  20. 市场总体统计

📊 数据管理                    🔧 系统管理
  7. 数据更新管理             21. 系统设置
  7. 数据库管理               22. 日志查看
                             23. 系统诊断
                             
🎮 其他
  0. 退出系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)


class TableDisplay:
    """表格显示组件"""
    
    @staticmethod
    def print_strategy_list(strategies: List[Dict[str, Any]]):
        """打印策略列表"""
        if not strategies:
            print("📋 当前可用策略: 无")
            return
        
        print("📋 当前可用策略:")
        print("-" * 80)
        print(f"{'编号':<4} {'策略名称':<15} {'描述':<25} {'胜率':<8} {'状态':<8}")
        print("-" * 80)
        
        for i, strategy in enumerate(strategies, 1):
            name = strategy.get('name', f'策略{i}')
            desc = strategy.get('description', '无描述')
            win_rate = strategy.get('win_rate', 'N/A')
            status = strategy.get('status', '可用')
            
            # 根据胜率设置颜色
            if isinstance(win_rate, (int, float)):
                if win_rate >= 60:
                    win_rate_str = f"{win_rate}% 🟢"
                elif win_rate >= 50:
                    win_rate_str = f"{win_rate}% 🟡"
                else:
                    win_rate_str = f"{win_rate}% 🔴"
            else:
                win_rate_str = str(win_rate)
            
            print(f"{i:<4} {name:<15} {desc:<25} {win_rate_str:<8} {status:<8}")
    
    @staticmethod
    def print_backtest_results(results: Dict[str, Any]):
        """打印回测结果"""
        if not results:
            print("❌ 无回测结果")
            return
        
        print("\n📊 回测报告")
        print("━" * 70)
        
        # 基本指标
        print("基本指标:")
        print(f"  总收益率:        {results.get('total_return', 0):+.1%}   📈")
        print(f"  年化收益率:      {results.get('annual_return', 0):+.1%}   📈")
        print(f"  最大回撤:        {results.get('max_drawdown', 0):.1%}   📉")
        print(f"  夏普比率:        {results.get('sharpe_ratio', 0):.2f}     ✅" if results.get('sharpe_ratio', 0) > 1 else f"  夏普比率:        {results.get('sharpe_ratio', 0):.2f}     ⚠️")
        print(f"  胜率:            {results.get('win_rate', 0):.0%}      ✅" if results.get('win_rate', 0) > 0.5 else f"  胜率:            {results.get('win_rate', 0):.0%}      ⚠️")
        print(f"  盈亏比:          {results.get('profit_loss_ratio', 0):.1f}:1    ✅" if results.get('profit_loss_ratio', 0) > 1.5 else f"  盈亏比:          {results.get('profit_loss_ratio', 0):.1f}:1    ⚠️")
        
        # 交易统计
        print("\n交易统计:")
        print(f"  交易次数:        {results.get('total_trades', 0)}      📊")
        print(f"  赚钱次数:        {results.get('winning_trades', 0)}       🟢")
        print(f"  亏钱次数:        {results.get('losing_trades', 0)}       🔴")
        print(f"  平均单笔收益:    {results.get('avg_trade_return', 0):+.2%}   📊")
        print(f"  最大单笔收益:    {results.get('max_win', 0):+.1%}    🎯")
        print(f"  最大单笔亏损:    {results.get('max_loss', 0):+.1f}    ⚠️")
        
        # 时间统计
        print("\n时间统计:")
        print(f"  平均持仓天数:    {results.get('avg_holding_days', 0):.1f}天   📅")
        print(f"  最长持仓:        {results.get('max_holding_days', 0)}天     📅")
        print(f"  最短持仓:        {results.get('min_holding_days', 0)}天     📅")
        
        # 资金统计
        print("\n资金统计:")
        initial_capital = results.get('initial_capital', 1000000)
        final_equity = results.get('final_equity', initial_capital)
        max_equity = results.get('max_equity', final_equity)
        max_drawdown_amount = results.get('max_drawdown_amount', 0)
        
        print(f"  初始资金:        {initial_capital:,.0f} 💰")
        print(f"  最终权益:        {final_equity:,.0f} 💰")
        print(f"  最大权益:        {max_equity:,.0f} 💰")
        print(f"  最大回撤额:      {max_drawdown_amount:+,.0f} 💰")
    
    @staticmethod
    def print_factor_analysis(factor_results: List[Dict[str, Any]]):
        """打印因子分析结果"""
        if not factor_results:
            print("❌ 无因子分析结果")
            return
        
        print("\n⚡ 因子有效性分析")
        print("━" * 70)
        
        print("各因子单独使用胜率 (前五):")
        print("-" * 70)
        print(f"{'排名':<4} {'因子名称':<12} {'胜率':<8} {'状态':<8} {'信号数':<8} {'准确度':<8}")
        print("-" * 70)
        
        for i, factor in enumerate(factor_results[:5], 1):
            name = factor.get('name', f'因子{i}')
            win_rate = factor.get('win_rate', 0)
            accuracy = factor.get('accuracy', 0)
            signals = factor.get('signals_count', 0)
            
            # 状态判断
            if win_rate >= 65:
                status = "✅ 有效"
            elif win_rate >= 55:
                status = "⚠️ 下降"
            elif win_rate >= 50:
                status = "⚠️ 不稳定"
            else:
                status = "❌ 失效"
            
            win_rate_str = f"{win_rate:.0f}%"
            print(f"{i:<4} {name:<12} {win_rate_str:<8} {status:<8} {signals:<8} {accuracy:.0f}%")
        
        # 最优组合
        print("\n最有效的因子组合:")
        combinations = factor_results[:3] if len(factor_results) >= 3 else factor_results
        for i, combo in enumerate(combinations, 1):
            combo_name = combo.get('name', f'组合{i}')
            combo_win_rate = combo.get('win_rate', 0)
            print(f"{i}. {combo_name}    胜率: {combo_win_rate:.0f}%")
    
    @staticmethod
    def print_industry_analysis(industry_results: List[Dict[str, Any]]):
        """打印行业分析结果"""
        if not industry_results:
            print("❌ 无行业分析结果")
            return
        
        print("\n📈 行业对比分析")
        print("━" * 70)
        
        print("按涨幅排名:")
        print("-" * 80)
        print(f"{'排名':<4} {'行业名称':<12} {'涨幅':<8} {'涨停数':<8} {'跌停数':<8} {'成交额':<10} {'状态':<6}")
        print("-" * 80)
        
        for i, industry in enumerate(industry_results, 1):
            name = industry.get('name', f'行业{i}')
            change = industry.get('change', 0)
            up_limit = industry.get('up_limit_count', 0)
            down_limit = industry.get('down_limit_count', 0)
            volume = industry.get('volume', 0)
            
            # 状态判断
            if change >= 5:
                status = "🏆"
            elif change >= 3:
                status = "✅"
            elif change >= 0:
                status = "⚠️"
            else:
                status = "❌"
            
            change_str = f"{change:+.1f}%"
            volume_str = f"{volume/1e8:.0f}M" if volume >= 1e8 else f"{volume/1e6:.0f}K"
            
            print(f"{i:<4} {name:<12} {change_str:<8} {up_limit:<8} {down_limit:<8} {volume_str:<10} {status:<6}")


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, message: str = "进度"):
        self.total = total
        self.current = 0
        self.message = message
        self.start_time = time.time()
        self.running = False
    
    def update(self, current: int):
        """更新进度"""
        self.current = current
        UIComponents.print_progress(current, self.total, self.message)
    
    def finish(self):
        """完成"""
        elapsed = time.time() - self.start_time
        print(f"\n✅ {self.message}完成! 耗时: {elapsed:.1f}秒")
    
    def start_async(self):
        """异步开始"""
        self.running = True
        self.update(0)
    
    def stop_async(self):
        """异步停止"""
        self.running = False
        self.finish()


class InputValidator:
    """输入验证器"""
    
    @staticmethod
    def validate_date(date_str: str) -> bool:
        """验证日期格式"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_stock_code(code: str) -> bool:
        """验证股票代码"""
        return len(code) == 6 and code.isdigit()
    
    @staticmethod
    def validate_percentage(value: str) -> bool:
        """验证百分比"""
        try:
            return 0 <= float(value) <= 100
        except ValueError:
            return False
    
    @staticmethod
    def validate_numeric_range(value: str, min_val: float, max_val: float) -> bool:
        """验证数值范围"""
        try:
            num = float(value)
            return min_val <= num <= max_val
        except ValueError:
            return False


class AsyncTaskRunner:
    """异步任务执行器"""
    
    def __init__(self):
        self.tasks = {}
        self.results = {}
    
    def run_task(self, name: str, func: Callable, *args, **kwargs):
        """运行任务"""
        def task_wrapper():
            try:
                result = func(*args, **kwargs)
                self.results[name] = result
            except Exception as e:
                self.results[name] = f"Error: {str(e)}"
        
        thread = threading.Thread(target=task_wrapper)
        thread.start()
        self.tasks[name] = thread
        return thread
    
    def wait_for_task(self, name: str, timeout: int = None):
        """等待任务完成"""
        if name in self.tasks:
            self.tasks[name].join(timeout)
            return self.results.get(name)
        return None
    
    def wait_for_all(self, timeout: int = None):
        """等待所有任务完成"""
        for name, thread in self.tasks.items():
            thread.join(timeout)
    
    def get_result(self, name: str):
        """获取结果"""
        return self.results.get(name)


# 导出所有组件
__all__ = [
    'UIComponents',
    'MenuDisplay', 
    'TableDisplay',
    'ProgressTracker',
    'InputValidator',
    'AsyncTaskRunner'
]