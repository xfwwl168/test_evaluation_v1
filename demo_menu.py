#!/usr/bin/env python3
"""
高级交互式菜单系统演示脚本
"""

import sys
from pathlib import Path

# 确保项目根目录在路径中
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def demo_menu_system():
    """演示菜单系统"""
    try:
        print("🎯 正在启动高级交互式菜单系统...")
        print("=" * 60)
        
        from menu.main_menu import MainMenu
        
        print("✅ 菜单系统加载成功!")
        print("\n🚀 启动选项:")
        print("1. 进入完整菜单系统")
        print("2. 快速功能演示")
        print("3. 查看系统信息")
        print("4. 退出")
        
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            print("\n🚀 启动完整菜单系统...")
            menu = MainMenu()
            menu.start()
        elif choice == '2':
            print("\n🎮 快速功能演示...")
            demo_functions()
        elif choice == '3':
            print("\n📊 系统信息:")
            show_system_info()
        elif choice == '4':
            print("\n👋 感谢使用!")
            return
        else:
            print("\n❌ 无效选择")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请检查是否正确安装了所有依赖")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

def demo_functions():
    """演示主要功能"""
    print("\n🎮 功能演示菜单:")
    print("-" * 40)
    print("1. 回测引擎演示")
    print("2. 实盘监控演示") 
    print("3. 市场分析演示")
    print("4. 数据管理演示")
    print("5. 系统管理演示")
    
    choice = input("\n请选择功能 (1-5): ").strip()
    
    if choice == '1':
        demo_backtest()
    elif choice == '2':
        demo_live_monitor()
    elif choice == '3':
        demo_market_analysis()
    elif choice == '4':
        demo_data_management()
    elif choice == '5':
        demo_system_management()
    else:
        print("❌ 无效选择")

def demo_backtest():
    """演示回测功能"""
    print("\n🎮 回测引擎演示")
    print("=" * 40)
    
    try:
        from menu.backtest_menu import BacktestMenu
        
        # 创建回测菜单实例
        backtest_menu = BacktestMenu()
        
        print("✅ 回测引擎功能:")
        print("├─ 策略管理: 7个内置策略")
        print("├─ 单策略回测: 完整参数配置")
        print("├─ 多策略对比: 并行回测")
        print("├─ 因子组合配置: 自定义权重")
        print("├─ 参数优化: GridSearch优化")
        print("└─ 回测历史: 历史记录管理")
        
        print("\n📊 可用策略:")
        for i, strategy in enumerate(backtest_menu.strategies, 1):
            print(f"{i}. {strategy['name']} - {strategy['description']} (胜率: {strategy['win_rate']}%)")
        
        print("\n🎯 功能特色:")
        print("├─ 策略复制和自定义")
        print("├─ 详细的回测报告")
        print("├─ 因子权重可视化")
        print("├─ 参数优化结果展示")
        print("└─ 历史对比分析")
        
    except Exception as e:
        print(f"❌ 演示错误: {e}")

def demo_live_monitor():
    """演示实盘监控功能"""
    print("\n📡 实盘监控演示")
    print("=" * 40)
    
    try:
        from menu.live_monitor import LiveMonitorMenu
        
        live_monitor = LiveMonitorMenu()
        
        print("✅ 实盘监控功能:")
        print("├─ 策略配置: 预设+临时策略")
        print("├─ 实时扫描: 全市场信号监控")
        print("├─ 股票跟踪: 深度因子分析")
        print("├─ 信号热力图: 可视化排名")
        print("├─ 卖出信号: 风险预警")
        print("└─ 持仓管理: 实时状态")
        
        print(f"\n🟢 买入信号 ({len(live_monitor.buy_signals)}个):")
        for signal in live_monitor.buy_signals[:3]:
            print(f"├─ {signal['code']} {signal['name']}: RSRS {signal['rsrs']:.2f}, 强度 {signal['strength']:.2f}")
        
        print(f"\n🔴 卖出信号 ({len(live_monitor.sell_signals)}个):")
        for signal in live_monitor.sell_signals:
            print(f"├─ {signal['code']} {signal['name']}: 强度 {signal['strength']:.2f}")
        
        print("\n🎯 功能特色:")
        print("├─ 多策略实时切换")
        print("├─ 信号强度热力图")
        print("├─ 详细因子分析")
        print("├─ 历史信号追踪")
        print("└─ 智能风险预警")
        
    except Exception as e:
        print(f"❌ 演示错误: {e}")

def demo_market_analysis():
    """演示市场分析功能"""
    print("\n📈 市场分析演示")
    print("=" * 40)
    
    try:
        from menu.market_analysis import MarketAnalysisMenu
        
        market_analysis = MarketAnalysisMenu()
        
        print("✅ 市场分析功能:")
        print("├─ 因子有效性分析: 胜率排名")
        print("├─ 行业对比分析: 行业排名")
        print("├─ 股票深度分析: 多维评分")
        print("├─ 因子排名: Top 100")
        print("├─ 行业板块分析: 热度统计")
        print("└─ 市场总体统计: 情绪监控")
        
        print("\n⚡ 因子有效性排名:")
        for i, factor in enumerate(market_analysis.factor_analysis_data[:3], 1):
            print(f"{i}. {factor['name']}: 胜率 {factor['win_rate']}% ({factor['status']})")
        
        print("\n📈 行业排名:")
        for industry in market_analysis.industry_analysis_data[:3]:
            print(f"{industry['rank']}. {industry['name']}: {industry['change']:+.1f}% ({industry['status']})")
        
        print("\n🎯 功能特色:")
        print("├─ 因子有效性量化评估")
        print("├─ 行业对比热力图")
        print("├─ 多维度股票分析")
        print("├─ 市场情绪指标")
        print("└─ 智能投资建议")
        
    except Exception as e:
        print(f"❌ 演示错误: {e}")

def demo_data_management():
    """演示数据管理功能"""
    print("\n📊 数据管理演示")
    print("=" * 40)
    
    try:
        from menu.data_management import DataManagementMenu
        
        data_management = DataManagementMenu()
        
        print("✅ 数据管理功能:")
        print("├─ 数据更新: 四种更新模式")
        print("├─ 数据库管理: 维护优化")
        print("├─ 数据质量: 四维检查")
        print("├─ 备份恢复: 完整体系")
        print("├─ 配置设置: 灵活配置")
        print("└─ 日志管理: 完整记录")
        
        print(f"\n📈 数据库状态:")
        stats = data_management.db_stats
        print(f"├─ 总股票数: {stats['total_stocks']:,}只")
        print(f"├─ 总数据行: {stats['total_rows']:,}行")
        print(f"├─ 数据库大小: {stats['storage']['database_size']:.1f}GB")
        print(f"└─ 数据完整性: {stats['data_quality']['completeness']:.1f}%")
        
        print("\n🔄 更新模式:")
        print("├─ 增量更新: 30-60秒 (推荐)")
        print("├─ 全量更新: 30-90分钟")
        print("├─ 智能更新: 2-5分钟")
        print("└─ 快速更新: 10-20秒")
        
        print("\n🎯 功能特色:")
        print("├─ 智能数据更新")
        print("├─ 数据质量监控")
        print("├─ 自动化备份")
        print("├─ 性能优化")
        print("└─ 完整审计日志")
        
    except Exception as e:
        print(f"❌ 演示错误: {e}")

def demo_system_management():
    """演示系统管理功能"""
    print("\n🔧 系统管理演示")
    print("=" * 40)
    
    try:
        from menu.system_management import SystemManagementMenu
        
        system_management = SystemManagementMenu()
        
        print("✅ 系统管理功能:")
        print("├─ 系统设置: 多项配置")
        print("├─ 日志查看: 多类型管理")
        print("├─ 系统诊断: 全方位检查")
        print("├─ 性能监控: 实时面板")
        print("├─ 安全设置: 完整体系")
        print("└─ 系统维护: 维护工具")
        
        print(f"\n📋 系统状态:")
        system_info = system_management.system_info
        print(f"├─ 平台: {system_info['platform']}")
        print(f"├─ Python版本: {system_info['python_version']}")
        print(f"├─ CPU核心: {system_info['cpu_count']}个")
        print(f"├─ 内存: {system_info['memory_total']/1024/1024/1024:.0f}GB")
        print(f"└─ 运行时间: {system_info['uptime']}")
        
        print(f"\n📝 日志文件 ({len(system_management.log_files)}个):")
        for log_file in system_management.log_files[:3]:
            print(f"├─ {log_file['name']}: {log_file['size']} ({log_file['level']})")
        
        print("\n🎯 功能特色:")
        print("├─ 全方位系统监控")
        print("├─ 智能诊断报告")
        print("├─ 实时性能监控")
        print("├─ 完善安全体系")
        print("└─ 自动化维护")
        
    except Exception as e:
        print(f"❌ 演示错误: {e}")

def show_system_info():
    """显示系统信息"""
    print("\n📊 高级交互式菜单系统信息")
    print("=" * 50)
    print("版本: v3.0 (Option A 完整版)")
    print("开发者: LION_QUANT 2026")
    print("架构: 模块化设计")
    print("状态: ✅ 运行正常")
    
    print("\n📋 功能模块:")
    modules = [
        ("🎮 回测引擎", "策略管理、参数优化、因子配置"),
        ("📡 实盘监控", "实时扫描、信号监控、持仓管理"),
        ("📈 市场分析", "因子分析、行业对比、股票分析"),
        ("📊 数据管理", "数据更新、质量检查、备份恢复"),
        ("🔧 系统管理", "系统设置、日志查看、诊断监控")
    ]
    
    for name, description in modules:
        print(f"├─ {name}: {description}")
    
    print("\n🎯 技术特色:")
    features = [
        "现代化Python架构",
        "模块化设计理念",
        "健壮错误处理",
        "流畅交互体验",
        "丰富视觉设计"
    ]
    
    for feature in features:
        print(f"├─ ✅ {feature}")
    
    print("\n🚀 启动方式:")
    print("├─ 完整菜单: python main.py menu")
    print("├─ 直接运行: python -m menu.main_menu")
    print("└─ 演示模式: python demo_menu.py")

def main():
    """主函数"""
    print("🎯 LION_QUANT 2026 - 高级交互式菜单系统")
    print("=" * 60)
    print("✨ Option A 完整版 - 功能演示")
    print("=" * 60)
    
    try:
        demo_menu_system()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出演示")
    except Exception as e:
        print(f"\n❌ 演示错误: {e}")
    finally:
        print("\n🔚 演示结束")

if __name__ == "__main__":
    main()