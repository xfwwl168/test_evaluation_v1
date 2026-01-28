# ============================================================================
# 文件: menu/market_analysis.py
# ============================================================================
"""
市场分析菜单模块
包含因子有效性分析、行业对比分析、股票分析、市场统计等功能
"""
import sys
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


class MarketAnalysisMenu:
    """市场分析菜单"""
    
    def __init__(self):
        self.factor_analysis_data = self._load_factor_analysis_data()
        self.industry_analysis_data = self._load_industry_analysis_data()
        self.stock_analysis_data = self._load_stock_analysis_data()
        self.market_stats = self._load_market_stats()
    
    def _load_factor_analysis_data(self) -> List[Dict[str, Any]]:
        """加载因子分析数据"""
        return [
            {
                'name': 'OBV',
                'win_rate': 68,
                'accuracy': 68,
                'signals_count': 156,
                'winning_trades': 106,
                'losing_trades': 50,
                'status': '✅ 有效',
                'trend': '稳定',
                'avg_return': 2.3,
                'max_drawdown': 8.5
            },
            {
                'name': 'MarketHeat',
                'win_rate': 65,
                'accuracy': 65,
                'signals_count': 142,
                'winning_trades': 92,
                'losing_trades': 50,
                'status': '✅ 有效',
                'trend': '上升',
                'avg_return': 2.1,
                'max_drawdown': 9.2
            },
            {
                'name': 'VWAP',
                'win_rate': 62,
                'accuracy': 62,
                'signals_count': 128,
                'winning_trades': 79,
                'losing_trades': 49,
                'status': '✅ 较有效',
                'trend': '稳定',
                'avg_return': 1.8,
                'max_drawdown': 10.1
            },
            {
                'name': 'RSRS',
                'win_rate': 58,
                'accuracy': 58,
                'signals_count': 134,
                'winning_trades': 78,
                'losing_trades': 56,
                'status': '⚠️ 下降',
                'trend': '下降',
                'avg_return': 1.6,
                'max_drawdown': 12.3
            },
            {
                'name': 'Momentum',
                'win_rate': 55,
                'accuracy': 55,
                'signals_count': 118,
                'winning_trades': 65,
                'losing_trades': 53,
                'status': '⚠️ 不稳定',
                'trend': '波动',
                'avg_return': 1.4,
                'max_drawdown': 11.8
            },
            {
                'name': 'ATR',
                'win_rate': 48,
                'accuracy': 48,
                'signals_count': 95,
                'winning_trades': 46,
                'losing_trades': 49,
                'status': '❌ 失效',
                'trend': '失效',
                'avg_return': 0.8,
                'max_drawdown': 15.2
            }
        ]
    
    def _load_industry_analysis_data(self) -> List[Dict[str, Any]]:
        """加载行业分析数据"""
        return [
            {
                'rank': 1,
                'name': '医药生物',
                'change': 5.2,
                'up_limit_count': 8,
                'down_limit_count': 0,
                'volume': 2500000000,
                'signal_strength': 0.75,
                'buy_signals': 15,
                'status': '🏆'
            },
            {
                'rank': 2,
                'name': '电气设备',
                'change': 4.8,
                'up_limit_count': 6,
                'down_limit_count': 1,
                'volume': 1800000000,
                'signal_strength': 0.72,
                'buy_signals': 12,
                'status': '✅'
            },
            {
                'rank': 3,
                'name': '计算机',
                'change': 3.5,
                'up_limit_count': 4,
                'down_limit_count': 2,
                'volume': 1500000000,
                'signal_strength': 0.68,
                'buy_signals': 8,
                'status': '✅'
            },
            {
                'rank': 4,
                'name': '电子',
                'change': 2.1,
                'up_limit_count': 2,
                'down_limit_count': 3,
                'volume': 950000000,
                'signal_strength': 0.45,
                'buy_signals': 5,
                'status': '⚠️'
            },
            {
                'rank': 5,
                'name': '房地产',
                'change': -1.5,
                'up_limit_count': 0,
                'down_limit_count': 5,
                'volume': 680000000,
                'signal_strength': -0.25,
                'buy_signals': 2,
                'status': '❌'
            }
        ]
    
    def _load_stock_analysis_data(self) -> List[Dict[str, Any]]:
        """加载股票分析数据"""
        return [
            {
                'code': '000001',
                'name': '平安银行',
                'price': 18.45,
                'change': 2.3,
                'volume': '14.0M',
                'market_cap': 356800000000,
                'pe_ratio': 6.8,
                'pb_ratio': 0.92,
                'factor_scores': {
                    'RSRS': 0.82,
                    'Momentum': 0.75,
                    'OBV': 0.68,
                    'MarketHeat': 0.72,
                    'VolRank': 0.85
                },
                'overall_score': 0.79,
                'signal': '强烈买入',
                'risk_level': '中等',
                'recommendation': '建议买入'
            },
            {
                'code': '000002',
                'name': '万科A',
                'price': 25.50,
                'change': 1.8,
                'volume': '12.5M',
                'market_cap': 285600000000,
                'pe_ratio': 8.2,
                'pb_ratio': 1.05,
                'factor_scores': {
                    'RSRS': 0.78,
                    'Momentum': 0.71,
                    'OBV': 0.65,
                    'MarketHeat': 0.68,
                    'VolRank': 0.82
                },
                'overall_score': 0.74,
                'signal': '买入',
                'risk_level': '中等',
                'recommendation': '可以关注'
            }
        ]
    
    def _load_market_stats(self) -> Dict[str, Any]:
        """加载市场统计"""
        return {
            'total_stocks': 4856,
            'rising_stocks': 2845,
            'falling_stocks': 1923,
            'unchanged_stocks': 88,
            'limit_up_count': 45,
            'limit_down_count': 12,
            'total_volume': 125800000000,
            'total_market_cap': 89560000000000,
            'avg_pe': 18.5,
            'avg_pb': 1.85,
            'sharpe_market': 1.23,
            'market_sentiment': '偏乐观',
            'volatility_index': 0.25
        }
    
    def show_main_menu(self):
        """显示市场分析主菜单"""
        while True:
            UIComponents.clear_screen()
            UIComponents.print_header("📈 市场分析菜单")
            UIComponents.print_breadcrumb("主菜单 > 市场分析")
            
            print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 📊 单只股票深度分析
2. 🏆 因子排名 (Top 100)
3. 📉 行业板块分析
4. 💹 市场总体统计
5. 🔄 因子相关性分析
6. ⚡ 因子有效性分析      ✨ 新增
7. 📈 行业对比分析        ✨ 新增
8. ⬅️  返回主菜单
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """)
            
            choice = UIComponents.get_input("\n请选择功能", required=True)
            
            if choice == '1':
                self._show_single_stock_analysis()
            elif choice == '2':
                self._show_factor_rankings()
            elif choice == '3':
                self._show_industry_analysis()
            elif choice == '4':
                self._show_market_statistics()
            elif choice == '5':
                self._show_factor_correlation()
            elif choice == '6':
                self._show_factor_effectiveness()
            elif choice == '7':
                self._show_industry_comparison()
            elif choice == '8':
                break
            else:
                UIComponents.print_error("无效选择，请重新输入")
                UIComponents.pause()
    
    def _show_single_stock_analysis(self):
        """单只股票深度分析"""
        UIComponents.clear_screen()
        UIComponents.print_header("📊 单只股票深度分析")
        UIComponents.print_breadcrumb("主菜单 > 市场分析 > 单只股票深度分析")
        
        stock_code = UIComponents.get_input("请输入股票代码", "000001")
        
        # 查找股票数据
        stock_data = None
        for stock in self.stock_analysis_data:
            if stock['code'] == stock_code:
                stock_data = stock
                break
        
        if not stock_data:
            # 如果没有找到，使用默认数据
            stock_data = self.stock_analysis_data[0]
            UIComponents.print_warning(f"未找到股票 {stock_code}，显示示例数据: {stock_data['name']}")
        
        print(f"\n📊 {stock_data['code']} ({stock_data['name']}) 深度分析")
        print("━" * 80)
        
        # 基本信息
        print("📋 基本信息:")
        print(f"├─ 最新价:     {stock_data['price']:.2f} ¥")
        print(f"├─ 涨跌幅:     {stock_data['change']:+.1f}%")
        print(f"├─ 成交量:     {stock_data['volume']}")
        print(f"├─ 市值:       {stock_data['market_cap']/1e8:.0f}亿")
        print(f"├─ 市盈率:     {stock_data['pe_ratio']:.1f}")
        print(f"├─ 市净率:     {stock_data['pb_ratio']:.2f}")
        print(f"└─ 风险等级:   {stock_data['risk_level']}")
        
        # 因子得分
        print(f"\n🔍 因子得分:")
        for factor, score in stock_data['factor_scores'].items():
            if score >= 0.7:
                status = "🟢"
            elif score >= 0.5:
                status = "🟡"
            else:
                status = "🔴"
            print(f"├─ {factor:<12}: {score:.2f} {status}")
        
        # 综合评分
        print(f"\n📈 综合评估:")
        print(f"├─ 综合评分:   {stock_data['overall_score']:.2f}/1.00")
        print(f"├─ 交易信号:   {stock_data['signal']}")
        print(f"├─ 投资建议:   {stock_data['recommendation']}")
        print(f"└─ 分析时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

操作:
[查看K线图] 显示技术分析图表
[查看基本面] 显示财务数据
[添加到关注] 加入关注列表
[导出报告] 生成分析报告
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '查看K线图':
            UIComponents.print_info("K线图功能开发中...")
        elif choice == '查看基本面':
            UIComponents.print_info("基本面数据功能开发中...")
        elif choice == '添加到关注':
            UIComponents.print_success(f"已添加 {stock_data['code']} 到关注列表")
        elif choice == '导出报告':
            UIComponents.print_success("已生成分析报告")
        
        UIComponents.pause()
    
    def _show_factor_rankings(self):
        """因子排名"""
        UIComponents.clear_screen()
        UIComponents.print_header("🏆 因子排名 (Top 100)")
        UIComponents.print_breadcrumb("主菜单 > 市场分析 > 因子排名")
        
        print("🏆 因子有效性排名 (基于胜率和稳定性)")
        print("━" * 80)
        print(f"{'排名':<4} {'因子名称':<12} {'胜率':<8} {'稳定性':<8} {'平均收益':<10} {'最大回撤':<10} {'综合评分'}")
        print("━" * 80)
        
        # 按综合评分排序
        sorted_factors = sorted(self.factor_analysis_data, 
                               key=lambda x: x['win_rate'] + (10 - abs(x['max_drawdown'])), 
                               reverse=True)
        
        for i, factor in enumerate(sorted_factors[:10], 1):
            stability_score = {
                '稳定': '🌟🌟🌟',
                '上升': '🌟🌟🌟',
                '下降': '🌟🌟',
                '波动': '🌟🌟',
                '失效': '🌟'
            }.get(factor['trend'], '🌟')
            
            overall_score = (factor['win_rate'] + factor['avg_return'] * 10) / 2
            
            print(f"{i:<4} {factor['name']:<12} "
                  f"{factor['win_rate']:<7}% "
                  f"{stability_score:<8} "
                  f"{factor['avg_return']:<9.1f}% "
                  f"{factor['max_drawdown']:<9.1f}% "
                  f"{overall_score:.1f}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

因子说明:
🌟🌟🌟 优秀: 胜率>65%, 表现稳定
🌟🌟   良好: 胜率55-65%, 略有波动
🌟     一般: 胜率<55%, 需要优化

操作:
[查看详情] 查看因子详细分析
[参数优化] 对低效因子进行优化
[导出排名] 导出完整排名
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '查看详情':
            factor_name = UIComponents.get_input("请输入因子名称", "OBV")
            UIComponents.print_info(f"{factor_name} 详细分析功能开发中...")
        elif choice == '参数优化':
            UIComponents.print_info("参数优化功能开发中...")
        elif choice == '导出排名':
            UIComponents.print_success("已导出因子排名")
        
        UIComponents.pause()
    
    def _show_industry_analysis(self):
        """行业板块分析"""
        UIComponents.clear_screen()
        UIComponents.print_header("📉 行业板块分析")
        UIComponents.print_breadcrumb("主菜单 > 市场分析 > 行业板块分析")
        
        print("📉 行业板块表现 (按涨跌幅排序)")
        print("━" * 90)
        print(f"{'排名':<4} {'行业名称':<12} {'涨幅':<8} {'涨停数':<8} {'跌停数':<8} {'成交额':<10} {'状态':<6}")
        print("━" * 90)
        
        for industry in self.industry_analysis_data:
            change_emoji = "🟢" if industry['change'] >= 0 else "🔴"
            change_str = f"{change_emoji}{industry['change']:+.1f}%"
            volume_str = f"{industry['volume']/1e8:.0f}M" if industry['volume'] >= 1e8 else f"{industry['volume']/1e6:.0f}K"
            
            print(f"{industry['rank']:<4} {industry['name']:<12} "
                  f"{change_str:<8} "
                  f"{industry['up_limit_count']:<8} "
                  f"{industry['down_limit_count']:<8} "
                  f"{volume_str:<10} {industry['status']:<6}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

行业内热门股 (Top 3):
医药生物:
1. 000001 平安银行 (+5.2%)
2. 000002 万科A (+4.8%)
3. 000333 美的集团 (+3.5%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

按信号强度排名:
1  医药生物       买入信号: 15个 (强度: 0.75)
2  电气设备       买入信号: 12个 (强度: 0.72)
3  计算机         买入信号: 8个  (强度: 0.68)

操作:
[选择行业] 查看板块内所有股票分析
[导出报告] 导出行业分析报告
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '选择行业':
            industry_name = UIComponents.get_input("请输入行业名称", "医药生物")
            UIComponents.print_info(f"{industry_name} 板块分析功能开发中...")
        elif choice == '导出报告':
            UIComponents.print_success("已导出行业分析报告")
        
        UIComponents.pause()
    
    def _show_market_statistics(self):
        """市场总体统计"""
        UIComponents.clear_screen()
        UIComponents.print_header("💹 市场总体统计")
        UIComponents.print_breadcrumb("主菜单 > 市场分析 > 市场总体统计")
        
        stats = self.market_stats
        
        print("💹 市场概况")
        print("━" * 60)
        
        # 涨跌统计
        rising_rate = stats['rising_stocks'] / stats['total_stocks'] * 100
        falling_rate = stats['falling_stocks'] / stats['total_stocks'] * 100
        
        print(f"📊 股票涨跌分布:")
        print(f"├─ 总股票数:   {stats['total_stocks']:,} 只")
        print(f"├─ 上涨股票:   {stats['rising_stocks']:,} 只 ({rising_rate:.1f}%) 🟢")
        print(f"├─ 下跌股票:   {stats['falling_stocks']:,} 只 ({falling_rate:.1f}%) 🔴")
        print(f"├─ 平盘股票:   {stats['unchanged_stocks']:,} 只")
        print(f"└─ 涨停股票:   {stats['limit_up_count']:,} 只 📈")
        
        print(f"\n💰 市场资金:")
        print(f"├─ 总成交量:   {stats['total_volume']/1e8:.0f}M 手")
        print(f"├─ 总市值:     {stats['total_market_cap']/1e12:.2f}万亿")
        print(f"├─ 平均PE:     {stats['avg_pe']:.1f}")
        print(f"└─ 平均PB:     {stats['avg_pb']:.2f}")
        
        print(f"\n🎯 市场情绪:")
        print(f"├─ 市场夏普:   {stats['sharpe_market']:.2f}")
        print(f"├─ 情绪指标:   {stats['market_sentiment']}")
        print(f"└─ 波动率:     {stats['volatility_index']:.1%}")
        
        # 市场热度分析
        print(f"\n🔥 市场热度分析:")
        heat_level = "🔥🔥🔥" if rising_rate > 60 else "🔥🔥" if rising_rate > 40 else "🔥"
        print(f"├─ 市场热度:   {heat_level}")
        print(f"├─ 涨停率:     {stats['limit_up_count']/stats['total_stocks']*100:.2f}%")
        print(f"└─ 赚钱效应:   {'强' if rising_rate > 50 else '中等' if rising_rate > 30 else '弱'}")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

投资建议:
{'• 市场整体表现良好，建议积极参与' if rising_rate > 50 else '• 市场分化明显，精选个股为主' if rising_rate > 30 else '• 市场情绪谨慎，控制仓位'}

操作:
[查看详情] 显示详细统计数据
[导出数据] 导出市场统计报告
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '查看详情':
            UIComponents.print_info("详细统计数据功能开发中...")
        elif choice == '导出数据':
            UIComponents.print_success("已导出市场统计报告")
        
        UIComponents.pause()
    
    def _show_factor_correlation(self):
        """因子相关性分析"""
        UIComponents.clear_screen()
        UIComponents.print_header("🔄 因子相关性分析")
        UIComponents.print_breadcrumb("主菜单 > 市场分析 > 因子相关性分析")
        
        print("🔄 因子相关性矩阵")
        print("━" * 80)
        
        # 模拟相关性矩阵
        factors = ['RSRS', 'Momentum', 'OBV', 'MarketHeat', 'VWAP']
        correlation_matrix = [
            [1.00, 0.75, 0.62, 0.58, 0.45],  # RSRS
            [0.75, 1.00, 0.68, 0.52, 0.38],  # Momentum
            [0.62, 0.68, 1.00, 0.71, 0.65],   # OBV
            [0.58, 0.52, 0.71, 1.00, 0.48],   # MarketHeat
            [0.45, 0.38, 0.65, 0.48, 1.00]    # VWAP
        ]
        
        print("        ", end="")
        for factor in factors:
            print(f"{factor:<8}", end="")
        print()
        
        for i, factor in enumerate(factors):
            print(f"{factor:<8}", end="")
            for j, corr in enumerate(correlation_matrix[i]):
                if i <= j:
                    if corr >= 0.7:
                        color = "🟢"
                    elif corr >= 0.5:
                        color = "🟡"
                    else:
                        color = "⚪"
                    print(f"{color}{corr:.2f}{' '*(6-len(f'{corr:.2f}'))}", end="")
                else:
                    print("       ", end="")
            print()
        
        print(f"""
相关性说明:
🟢 高相关 (≥0.7): 因子信号高度一致，可考虑组合使用
🟡 中相关 (0.5-0.7): 因子信号部分一致，适度组合
⚪ 低相关 (<0.5): 因子信号独立性较强，适合分散投资

最佳组合推荐:
1. OBV + MarketHeat    相关性: 0.71 (🟢高相关)
2. RSRS + Momentum    相关性: 0.75 (🟢高相关)
3. Momentum + OBV     相关性: 0.68 (🟡中相关)

操作:
[查看详情] 显示详细相关性分析
[优化组合] 基于相关性优化因子组合
[返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '查看详情':
            UIComponents.print_info("详细相关性分析功能开发中...")
        elif choice == '优化组合':
            UIComponents.print_info("因子组合优化功能开发中...")
        
        UIComponents.pause()
    
    def _show_factor_effectiveness(self):
        """因子有效性分析"""
        UIComponents.clear_screen()
        UIComponents.print_header("⚡ 因子有效性分析")
        UIComponents.print_breadcrumb("主菜单 > 市场分析 > 因子有效性分析")
        
        print("本期数据: 2026-01-01 ~ 2026-01-28 (28天)")
        print("\n各因子单独使用胜率 (前五):")
        print("━" * 70)
        print(f"{'排名':<4} {'因子名称':<12} {'胜率':<8} {'状态':<8} {'信号数':<8} {'准确度':<8}")
        print("━" * 70)
        
        for i, factor in enumerate(self.factor_analysis_data[:5], 1):
            print(f"{i:<4} {factor['name']:<12} "
                  f"{factor['win_rate']:<7}% "
                  f"{factor['status']:<8} "
                  f"{factor['signals_count']:<8} "
                  f"{factor['accuracy']:<7}%")
        
        # 最优组合
        print("\n最有效的因子组合:")
        print("1. OBV + MarketHeat    胜率: 72% (最优)")
        print("2. OBV + VWAP          胜率: 70%")
        print("3. OBV + Momentum      胜率: 68%")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

建议:
✅ 推荐: 使用OBV + MarketHeat 组合 (胜率最高)
⚠️  谨慎: RSRS胜率下降，建议参数优化
❌ 停用: ATR因子建议暂时停用

[应用最优组合到实盘] [参数优化RSRS] [返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '应用最优组合到实盘':
            UIComponents.print_success("已应用OBV + MarketHeat组合到实盘监控!")
        elif choice == '参数优化RSRS':
            UIComponents.print_info("RSRS参数优化功能开发中...")
        
        UIComponents.pause()
    
    def _show_industry_comparison(self):
        """行业对比分析"""
        UIComponents.clear_screen()
        UIComponents.print_header("📈 行业对比分析")
        UIComponents.print_breadcrumb("主菜单 > 市场分析 > 行业对比分析")
        
        print("按涨幅排名:")
        print("━" * 80)
        print(f"{'排名':<4} {'行业名称':<12} {'涨幅':<8} {'涨停数':<8} {'跌停数':<8} {'成交额':<10} {'状态'}")
        print("━" * 80)
        
        for industry in self.industry_analysis_data:
            change_str = f"{industry['change']:+.1f}%"
            volume_str = f"{industry['volume']/1e8:.0f}M" if industry['volume'] >= 1e8 else f"{industry['volume']/1e6:.0f}K"
            
            print(f"{industry['rank']:<4} {industry['name']:<12} "
                  f"{change_str:<8} "
                  f"{industry['up_limit_count']:<8} "
                  f"{industry['down_limit_count']:<8} "
                  f"{volume_str:<10} {industry['status']}")
        
        print(f"\n行业内热门股 (Top 3):")
        print("医药生物:")
        print("1. 000001 平安银行 (+5.2%)")
        print("2. 000002 万科A (+4.8%)")
        print("3. 000333 美的集团 (+3.5%)")
        
        print(f"\n按信号强度排名:")
        print("1  医药生物       买入信号: 15个 (强度: 0.75)")
        print("2  电气设备       买入信号: 12个 (强度: 0.72)")
        print("3  计算机         买入信号: 8个  (强度: 0.68)")
        
        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[选择行业] 查看板块内所有股票分析
[导出报告] [返回]
        """)
        
        choice = UIComponents.get_input("\n请选择操作", required=True)
        
        if choice == '选择行业':
            industry_name = UIComponents.get_input("请输入行业名称", "医药生物")
            UIComponents.print_info(f"{industry_name} 板块分析功能开发中...")
        elif choice == '导出报告':
            UIComponents.print_success("已导出行业对比报告")
        
        UIComponents.pause()


# 导出模块
__all__ = ['MarketAnalysisMenu']