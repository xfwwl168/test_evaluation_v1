# ============================================================================
# 文件: menu/live_monitor.py
# ============================================================================
"""
实盘监控菜单模块
包含策略配置、实时扫描、股票跟踪、信号热力图等功能
"""
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# 确保项目根目录在路径中
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .ui_components import UIComponents, TableDisplay, ProgressTracker, InputValidator
from utils.logger import get_logger

logger = get_logger(__name__)


class LiveMonitorMenu:
    """实盘监控菜单"""
    
    def __init__(self):
        self.current_strategy = 'alpha_hunter'
        self.monitored_stocks = []
        self.buy_signals = []
        self.sell_signals = []
        self.portfolios = []
        
        # 加载模拟数据
        self._load_mock_data()
    
    def _load_mock_data(self):
        """加载模拟数据"""
        # 模拟买入信号
        self.buy_signals = [
            {'code': '000001', 'name': '平安银行', 'rsrs': 0.82, 'momentum': 0.75, 'heat': 0.68, 'strength': 0.79, 'limit_up': 2.3, 'volume': '258M'},
            {'code': '000002', 'name': '万科A', 'rsrs': 0.81, 'momentum': 0.73, 'heat': 0.65, 'strength': 0.77, 'limit_up': 1.5, 'volume': '185M'},
            {'code': '000333', 'name': '美的集团', 'rsrs': 0.80, 'momentum': 0.72, 'heat': 0.70, 'strength': 0.76, 'limit_up': 1.2, 'volume': '325M'},
            {'code': '600000', 'name': '浦发银行', 'rsrs': 0.79, 'momentum': 0.71, 'heat': 0.62, 'strength': 0.74, 'limit_up': 0.8, 'volume': '195M'},
        ]
        
        # 模拟卖出信号
        self.sell_signals = [
            {'code': '300001', 'name': '特联发展', 'rsrs': -0.52, 'momentum': -0.35, 'strength': -0.43, 'change': -3.2},
            {'code': '300002', 'name': '洛阳钼业', 'rsrs': -0.55, 'momentum': -0.38, 'strength': -0.46, 'change': -2.8},
        ]
        
        # 模拟跟踪股票
        self.monitored_stocks = [
            {
                'code': '000001',
                'name': '平安银行',
                'price': 18.45,
                'change': 2.3,
                'volume': '14.0M',
                'amount': '258.5M',
                'rsrs': 0.8234,
                'momentum': 0.7512,
                'obv': '12.5M',
                'market_heat': 0.68,
                'vol_rank': 0.85,
                'signal': '买入',
                'strength': 0.79
            }
        ]
    
    def show_main_menu(self):
        """显示实盘监控主菜单"""
        while True:
            UIComponents.clear_screen()
            UIComponents.print_header("📡 实盘监控菜单")
            UIComponents.print_breadcrumb("主菜单 > 实盘监控")
            
            print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 🎯 策略配置              ✨ 子菜单
2. 🔍 实时全市场扫描
3. 📍 跟踪单只股票
4. 📋 查看当前跟踪列表
5. 📊 买入信号热力图
6. 🔴 卖出信号列表
7. 💾 持仓管理
8. ⬅️  返回主菜单
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """)
            
            choice = UIComponents.get_input("\n请选择功能", required=True)
            
            if choice == '1':
                self._show_strategy_configuration()
            elif choice == '2':
                self._show_realtime_market_scan()
            elif choice == '3':
                self._show_stock_tracking()
            elif choice == '4':
                self._show_tracking_list()
            elif choice == '5':
                self._show_buy_signals_heatmap()
            elif choice == '6':
                self._show_sell_signals()
            elif choice == '7':
                self._show_portfolio_management()
            elif choice == '8':
                break
            else:
                UIComponents.print_error("无效选择，请重新输入")
                UIComponents.pause()
    
    def _show_strategy_configuration(self):
        """策略配置子菜单"""
        while True:
            UIComponents.clear_screen()
            UIComponents.print_header("🎯 策略配置")
            UIComponents.print_breadcrumb("主菜单 > 实盘监控 > 策略配置")
            
            print(f"""
实盘监控当前策略: {self.current_strategy.upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] 切换预设策略
    ○ RSRS       (趋势追踪)
    ○ Momentum   (动量策略)
    ● AlphaHunter (多因子综合) ← 当前
    ○ UltraShort (超短线)
    
[2] 创建临时策略 (本次监听使用，不保存)
    ├─ 选择因子组合
    │  [趋势组合 ▼]
    │  (RSRS 60% + Momentum 40%)
    │
    ├─ 调整权重 (可选)
    │  RSRS:     [60]%
    │  Momentum: [40]%
    │  其他:     [0]%
    │
    └─ 设置阈值
       买入: [0.70]
       卖出: [-0.50]

[3] 保存当前为模板
    输入模板名: [我的策略_v2]
    [保存]
    
[4] 导入自定义策略
    从文件: [选择文件...]
    [导入并应用]

[5] 策略对比
    显示各策略近30天的信号准确度
    (RSRS 胜率58% vs Momentum 胜率55%)

[6] 返回

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前策略详情:
因子:    AlphaHunter (多因子权重)
买入阈值: 0.70
卖出阈值: -0.50
刷新频率: 实时
            """)
            
            choice = UIComponents.get_input("\n请选择操作", required=True)
            
            if choice == '1':
                self._switch_preset_strategy()
            elif choice == '2':
                self._create_temporary_strategy()
            elif choice == '3':
                self._save_strategy_template()
            elif choice == '4':
                self._import_custom_strategy()
            elif choice == '5':
                self._compare_strategies()
            elif choice == '6':
                break
            else:
                UIComponents.print_error("无效选择")
                UIComponents.pause()
    
    def _switch_preset_strategy(self):
        """切换预设策略"""
        strategies = [
            ('RSRS', '趋势追踪'),
            ('Momentum', '动量策略'),
            ('AlphaHunter', '多因子综合'),
            ('UltraShort', '超短线')
        ]
        
        print("\n选择策略:")
        for i, (name, desc) in enumerate(strategies, 1):
            marker = "●" if name.lower() == self.current_strategy else "○"
            print(f"{i}. {marker} {name} ({desc})")
        
        choice = UIComponents.get_input("\n请选择策略", required=True)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(strategies):
                selected_strategy = strategies[idx][0].lower()
                self.current_strategy = selected_strategy
                UIComponents.print_success(f"已切换到策略: {strategies[idx][0]}")
            else:
                UIComponents.print_error("无效策略编号")
        except ValueError:
            UIComponents.print_error("请输入有效数字")
        
        UIComponents.pause()
    
    def _create_temporary_strategy(self):
        """创建临时策略"""
        print("\n🛠️ 创建临时策略")
        print("-" * 30)
        
        print("选择因子组合:")
        combos = [
            "趋势组合 (RSRS 60% + Momentum 40%)",
            "量价组合 (OBV 50% + VWAP 50%)",
            "综合组合 (多因子加权)",
            "自定义组合"
        ]
        
        for i, combo in enumerate(combos, 1):
            print(f"{i}. {combo}")
        
        combo_choice = UIComponents.get_input("\n请选择组合", "1")
        
        print("\n调整权重 (可选):")
        rsrs_weight = UIComponents.get_input("RSRS权重", "60")
        momentum_weight = UIComponents.get_input("Momentum权重", "40")
        
        print("\n设置阈值:")
        entry_threshold = UIComponents.get_input("买入阈值", "0.70")
        exit_threshold = UIComponents.get_input("卖出阈值", "-0.50")
        
        # 创建临时策略（这里可以实际创建策略对象）
        temp_strategy = {
            'name': '临时策略',
            'rsrs_weight': float(rsrs_weight),
            'momentum_weight': float(momentum_weight),
            'entry_threshold': float(entry_threshold),
            'exit_threshold': float(exit_threshold)
        }
        
        if UIComponents.get_yes_no("确认创建临时策略并应用到当前监听?"):
            UIComponents.print_success("临时策略创建成功，已应用到监听!")
            # 这里可以实际应用到监听系统
            UIComponents.pause()
    
    def _save_strategy_template(self):
        """保存策略模板"""
        template_name = UIComponents.get_input("请输入模板名称", "我的策略_v2")
        
        template = {
            'name': template_name,
            'strategy': self.current_strategy,
            'entry_threshold': 0.70,
            'exit_threshold': -0.50,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if UIComponents.get_yes_no(f"确认保存策略模板 '{template_name}'?"):
            UIComponents.print_success(f"策略模板 '{template_name}' 保存成功!")
            # 这里可以实际保存到文件或数据库
            UIComponents.pause()
    
    def _import_custom_strategy(self):
        """导入自定义策略"""
        file_path = UIComponents.get_input("请输入策略文件路径", "")
        
        if not file_path:
            UIComponents.print_warning("文件路径不能为空")
            return
        
        if UIComponents.get_yes_no(f"确认从 '{file_path}' 导入策略?"):
            UIComponents.print_success("策略导入成功!")
            # 这里可以实际导入策略文件
            UIComponents.pause()
    
    def _compare_strategies(self):
        """策略对比"""
        UIComponents.clear_screen()
        UIComponents.print_header("📊 策略对比分析")
        
        print("各策略近30天信号准确度:")
        print("━" * 60)
        print(f"{'策略':<15} {'胜率':<10} {'信号数':<10} {'平均收益':<12} {'评级'}")
        print("━" * 60)
        
        comparisons = [
            {'strategy': 'RSRS', 'win_rate': 58, 'signals': 45, 'avg_return': 2.1, 'rating': '良好'},
            {'strategy': 'Momentum', 'win_rate': 55, 'signals': 38, 'avg_return': 1.8, 'rating': '一般'},
            {'strategy': 'AlphaHunter', 'win_rate': 62, 'signals': 52, 'avg_return': 2.5, 'rating': '优秀'},
            {'strategy': 'UltraShort', 'win_rate': 48, 'signals': 67, 'avg_return': 1.2, 'rating': '较差'}
        ]
        
        for comp in comparisons:
            print(f"{comp['strategy']:<15} "
                  f"{comp['win_rate']:<9}% "
                  f"{comp['signals']:<9} "
                  f"{comp['avg_return']:<11.1f}% "
                  f"{comp['rating']}")
        
        print(f"\n💡 建议: 当前使用 AlphaHunter 策略表现最佳")
        
        UIComponents.pause()
    
    def _show_realtime_market_scan(self):
        """实时全市场扫描"""
        UIComponents.clear_screen()
        UIComponents.print_header("🔍 实时全市场扫描")
        UIComponents.print_breadcrumb("主菜单 > 实盘监控 > 实时全市场扫描")
        
        print(f"""
当前使用策略: {self.current_strategy.upper()}

选项:
( ) 使用当前策略
(●) 临时切换策略:
    [AlphaHunter ▼]

[开始扫描]
        """)
        
        if UIComponents.get_yes_no("确认开始扫描?"):
            # 模拟扫描进度
            print("\n扫描进度: 🔍 扫描中...")
            
            tracker = ProgressTracker(total=3000, message="全市场扫描")
            
            for i in range(3000):
                time.sleep(0.0001)  # 模拟快速扫描
                tracker.update(i + 1)
            
            tracker.finish()
            
            # 显示扫描结果
            UIComponents.clear_screen()
            UIComponents.print_header("🟢 买入信号")
            
            if self.buy_signals:
                print("买入信号 (23个):")
                print("━" * 80)
                print(f"{'#':<4} {'代码':<8} {'名称':<10} {'RSRS':<8} {'Mom':<8} {'热度':<8} {'强度':<8} {'涨停%':<8} {'成交额':<10}")
                print("━" * 80)
                
                for i, signal in enumerate(self.buy_signals, 1):
                    print(f"{i:<4} {signal['code']:<8} {signal['name']:<10} "
                          f"{signal['rsrs']:<7.2f} {signal['momentum']:<7.2f} "
                          f"{signal['heat']:<7.2f} {signal['strength']:<7.2f} "
                          f"{signal['limit_up']:<7.1f}% {signal['volume']:<10}")
            else:
                print("暂无买入信号")
            
            print(f"\n🔴 卖出信号 (8个):")
            print("━" * 70)
            print(f"{'#':<4} {'代码':<8} {'名称':<10} {'RSRS':<8} {'Mom':<8} {'强度':<8} {'涨跌%':<8}")
            print("━" * 70)
            
            for i, signal in enumerate(self.sell_signals, 1):
                print(f"{i:<4} {signal['code']:<8} {signal['name']:<10} "
                      f"{signal['rsrs']:<7.2f} {signal['momentum']:<7.2f} "
                      f"{signal['strength']:<7.2f} {signal['change']:<7.1f}%")
            
            print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

操作:
[跟踪] 选中列表中的股票进行追踪
[详情] 查看单只股票详细因子
[导出] 导出为CSV/Excel
[返回]
            """)
            
            choice = UIComponents.get_input("\n请选择操作", required=True)
            
            if choice == '跟踪':
                stock_code = UIComponents.get_input("请输入要跟踪的股票代码", "000001")
                UIComponents.print_success(f"已添加 {stock_code} 到跟踪列表")
            elif choice == '详情':
                stock_code = UIComponents.get_input("请输入股票代码", "000001")
                self._show_stock_details(stock_code)
            elif choice == '导出':
                UIComponents.print_success("已导出扫描结果")
            else:
                pass
            
            UIComponents.pause()
    
    def _show_stock_details(self, code: str):
        """显示股票详细因子"""
        print(f"\n📊 {code} 详细因子分析:")
        print("-" * 50)
        
        # 模拟详细信息
        details = {
            'RSRS': {'value': 0.8234, 'status': '强势', 'signal': '买入'},
            'Momentum': {'value': 0.7512, 'status': '看多', 'signal': '买入'},
            'OBV': {'value': '12.5M', 'status': '增量', 'signal': '中性'},
            'MarketHeat': {'value': 0.68, 'status': '偏热', 'signal': '中性'},
            'VolRank': {'value': 0.85, 'status': '强', 'signal': '买入'}
        }
        
        for factor, info in details.items():
            status_emoji = "✅" if info['status'] in ['强势', '看多', '强'] else "⚠️"
            signal_emoji = "🟢" if info['signal'] == '买入' else "🟡"
            
            print(f"{factor:<12}: {info['value']:<10} {status_emoji} {info['status']:<8} {signal_emoji} {info['signal']}")
    
    def _show_stock_tracking(self):
        """跟踪单只股票"""
        UIComponents.clear_screen()
        UIComponents.print_header("📍 跟踪单只股票")
        UIComponents.print_breadcrumb("主菜单 > 实盘监控 > 跟踪单只股票")
        
        print("""
输入股票代码: 000001 ↵

选择监听策略:
( ) RSRS
( ) Momentum
(●) AlphaHunter (当前)
( ) 自定义

如需自定义，选择因子:
□ RSRS       权重: [60]%
□ Momentum   权重: [40]%
□ OBV        权重: [0]%
更多...

信号阈值:
买入:  [0.70]
卖出:  [-0.50]

[开始监听]
        """)
        
        stock_code = UIComponents.get_input("\n请输入股票代码", "000001")
        stock_name = f"平安银行"  # 模拟名称识别
        
        if UIComponents.get_yes_no("确认开始监听?"):
            self._start_stock_monitoring(stock_code, stock_name)
    
    def _start_stock_monitoring(self, code: str, name: str):
        """开始股票监听"""
        UIComponents.clear_screen()
        UIComponents.print_header(f"📍 实时监听 {code}")
        
        print(f"策略: {self.current_strategy.upper()}")
        print(f"\n基本信息:")
        print(f"├─ 最新价:     18.45 ¥")
        print(f"├─ 涨跌幅:     +2.3%  🟢")
        print(f"├─ 成交额:     258.5M 💰")
        print(f"└─ 成交量:     14.0M  📊")
        
        print(f"\n实时因子值:")
        print(f"├─ RSRS:        0.8234 ✅ (强势)")
        print(f"├─ Momentum:    0.7512 ✅ (看多)")
        print(f"├─ OBV:         12.5M  📈 (增量)")
        print(f"├─ MarketHeat:  0.68   🔥 (偏热)")
        print(f"└─ VolRank:     0.85   🎯 (强)")
        
        print(f"\n综合评分: 0.79/1.0")
        print(f"\n📊 信号: 🟢 强烈买入")
        print(f"原因: RSRS 强势 + Momentum 看多 + 热度高")
        
        print(f"\n历史信号 (过去7天):")
        print("日期      信号   强度  操作")
        print("━━━━━━━━━━━━━━━━━━━━━")
        print("01-28   买入   0.79  +2.3%")
        print("01-27   买入   0.76  +1.8%")
        print("01-26   持仓   0.65  -0.5%")
        print("01-25   卖出  -0.52  -1.2%")
        
        print(f"\n监听中...")
        print("(信号变化时实时更新，按 Q 停止监听)")
        
        # 模拟监听
        while True:
            try:
                user_input = input("\n输入命令 (Q退出, 其他查看详情): ").strip().upper()
                if user_input == 'Q':
                    break
                else:
                    UIComponents.print_info("显示更多详细信息...")
            except KeyboardInterrupt:
                break
        
        UIComponents.print_success("监听已停止")
        UIComponents.pause()
    
    def _show_tracking_list(self):
        """查看跟踪列表"""
        UIComponents.clear_screen()
        UIComponents.print_header("📋 当前跟踪列表")
        
        if not self.monitored_stocks:
            print("📋 当前跟踪列表: 空")
            UIComponents.pause()
            return
        
        print(f"📋 当前跟踪列表 ({len(self.monitored_stocks)}只股票):")
        print("━" * 100)
        print(f"{'代码':<8} {'名称':<10} {'最新价':<8} {'涨跌幅':<8} {'信号':<8} {'强度':<8} {'最后更新'}")
        print("━" * 100)
        
        for stock in self.monitored_stocks:
            change_emoji = "🟢" if stock['change'] > 0 else "🔴" if stock['change'] < 0 else "⚪"
            signal_emoji = "🟢" if stock['signal'] == '买入' else "🔴" if stock['signal'] == '卖出' else "🟡"
            
            print(f"{stock['code']:<8} {stock['name']:<10} {stock['price']:<7.2f} "
                  f"{stock['change']:<+7.1f}% {signal_emoji}{stock['signal']:<6} "
                  f"{stock['strength']:<7.2f} 刚刚")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

操作:
[详情] 查看选中股票详细信息
[移除] 从跟踪列表中移除股票
[导出] 导出跟踪列表
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '详情':
            code = UIComponents.get_input("请输入股票代码", "000001")
            self._show_stock_details(code)
        elif choice == '移除':
            code = UIComponents.get_input("请输入要移除的股票代码", "000001")
            UIComponents.print_success(f"已从跟踪列表移除 {code}")
        elif choice == '导出':
            UIComponents.print_success("已导出跟踪列表")
        
        UIComponents.pause()
    
    def _show_buy_signals_heatmap(self):
        """买入信号热力图"""
        UIComponents.clear_screen()
        UIComponents.print_header("📊 买入信号热力图")
        
        print("📊 买入信号热力图 (按信号强度排序)")
        print("━" * 90)
        print(f"{'排名':<4} {'代码':<8} {'名称':<10} {'RSRS':<8} {'动量':<8} {'热度':<8} {'综合':<8} {'状态'}")
        print("━" * 90)
        
        # 模拟更多数据
        heatmap_data = [
            {'rank': 1, 'code': '000001', 'name': '平安银行', 'rsrs': 0.82, 'momentum': 0.75, 'heat': 0.68, 'score': 0.79, 'status': '🔥'},
            {'rank': 2, 'code': '000002', 'name': '万科A', 'rsrs': 0.81, 'momentum': 0.73, 'heat': 0.65, 'score': 0.77, 'status': '🔥'},
            {'rank': 3, 'code': '000333', 'name': '美的集团', 'rsrs': 0.80, 'momentum': 0.72, 'heat': 0.70, 'score': 0.76, 'status': '🟢'},
            {'rank': 4, 'code': '600000', 'name': '浦发银行', 'rsrs': 0.79, 'momentum': 0.71, 'heat': 0.62, 'score': 0.74, 'status': '🟢'},
            {'rank': 5, 'code': '600036', 'name': '招商银行', 'rsrs': 0.78, 'momentum': 0.70, 'heat': 0.60, 'score': 0.73, 'status': '🟢'},
        ]
        
        for item in heatmap_data:
            print(f"{item['rank']:<4} {item['code']:<8} {item['name']:<10} "
                  f"{item['rsrs']:<7.2f} {item['momentum']:<7.2f} "
                  f"{item['heat']:<7.2f} {item['score']:<7.2f} {item['status']}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

信号强度说明:
🔥 强烈买入 (综合评分 > 0.75)
🟢 买入 (综合评分 0.70-0.75)
🟡 观望 (综合评分 0.65-0.70)
⚪ 无信号 (综合评分 < 0.65)

操作:
[添加到跟踪] 选择股票添加到跟踪列表
[导出] 导出热力图数据
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '添加到跟踪':
            code = UIComponents.get_input("请输入要添加的股票代码", "000001")
            UIComponents.print_success(f"已添加 {code} 到跟踪列表")
        elif choice == '导出':
            UIComponents.print_success("已导出热力图数据")
        
        UIComponents.pause()
    
    def _show_sell_signals(self):
        """卖出信号列表"""
        UIComponents.clear_screen()
        UIComponents.print_header("🔴 卖出信号列表")
        
        if not self.sell_signals:
            print("🔴 当前无卖出信号")
            UIComponents.pause()
            return
        
        print(f"🔴 卖出信号列表 ({len(self.sell_signals)}个信号):")
        print("━" * 80)
        print(f"{'#':<4} {'代码':<8} {'名称':<10} {'RSRS':<8} {'动量':<8} {'强度':<8} {'跌幅':<8} {'建议'}")
        print("━" * 80)
        
        for i, signal in enumerate(self.sell_signals, 1):
            advice = "建议卖出" if signal['strength'] < -0.4 else "减仓"
            print(f"{i:<4} {signal['code']:<8} {signal['name']:<10} "
                  f"{signal['rsrs']:<7.2f} {signal['momentum']:<7.2f} "
                  f"{signal['strength']:<7.2f} {signal['change']:<7.1f}% {advice}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

建议操作:
• 建议卖出: 综合信号强度 < -0.4
• 减仓:     综合信号强度 -0.4 ~ -0.2  
• 观望:     综合信号强度 > -0.2

操作:
[添加到观察] 将股票加入观察列表
[导出] 导出卖出信号列表
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '添加到观察':
            code = UIComponents.get_input("请输入股票代码", "300001")
            UIComponents.print_success(f"已添加 {code} 到观察列表")
        elif choice == '导出':
            UIComponents.print_success("已导出卖出信号列表")
        
        UIComponents.pause()
    
    def _show_portfolio_management(self):
        """持仓管理"""
        UIComponents.clear_screen()
        UIComponents.print_header("💾 持仓管理")
        
        # 模拟持仓数据
        portfolios = [
            {
                'code': '000001',
                'name': '平安银行',
                'shares': 10000,
                'cost': 18.20,
                'current': 18.45,
                'profit': 2500,
                'profit_rate': 1.37,
                'signal': '持有'
            },
            {
                'code': '000002',
                'name': '万科A',
                'shares': 5000,
                'cost': 25.80,
                'current': 25.50,
                'profit': -1500,
                'profit_rate': -1.16,
                'signal': '减仓'
            }
        ]
        
        print("💾 当前持仓:")
        print("━" * 100)
        print(f"{'代码':<8} {'名称':<10} {'持仓':<8} {'成本价':<8} {'现价':<8} {'盈亏':<8} {'收益率':<8} {'信号'}")
        print("━" * 100)
        
        total_profit = 0
        for position in portfolios:
            profit_emoji = "🟢" if position['profit'] > 0 else "🔴"
            signal_emoji = "🟢" if position['signal'] == '买入' else "🔴" if position['signal'] == '卖出' else "🟡"
            
            print(f"{position['code']:<8} {position['name']:<10} "
                  f"{position['shares']:<7} {position['cost']:<7.2f} "
                  f"{position['current']:<7.2f} {profit_emoji}{position['profit']:<+6} "
                  f"{position['profit_rate']:<+7.1f}% {signal_emoji}{position['signal']:<4}")
            
            total_profit += position['profit']
        
        print(f"\n📊 持仓汇总:")
        print(f"├─ 总持仓: 2只股票")
        print(f"├─ 总市值: ¥{sum(p['shares'] * p['current'] for p in portfolios):,.0f}")
        print(f"├─ 总盈亏: {'🟢' if total_profit > 0 else '🔴'} ¥{total_profit:+,.0f}")
        print(f"└─ 整体收益率: {(total_profit / sum(p['shares'] * p['cost'] for p in portfolios)):.2%}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

操作:
[调仓] 根据信号建议调整持仓
[止损] 设置止损点
[止盈] 设置止盈点
[导出] 导出持仓报告
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '调仓':
            UIComponents.print_info("调仓功能开发中...")
        elif choice == '止损':
            UIComponents.print_info("止损设置功能开发中...")
        elif choice == '止盈':
            UIComponents.print_info("止盈设置功能开发中...")
        elif choice == '导出':
            UIComponents.print_success("已导出持仓报告")
        
        UIComponents.pause()


# 导出模块
__all__ = ['LiveMonitorMenu']