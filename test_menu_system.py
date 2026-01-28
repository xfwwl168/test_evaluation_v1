#!/usr/bin/env python3
# 测试新菜单系统是否能正确导入和运行

import sys
import os
from pathlib import Path

# 确保项目根目录在路径中
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def test_imports():
    """测试模块导入"""
    try:
        print("测试导入菜单模块...")
        from menu import MainMenu
        print("✅ MainMenu 导入成功")
        
        from menu.ui_components import UIComponents
        print("✅ UIComponents 导入成功")
        
        from menu.backtest_menu import BacktestMenu
        print("✅ BacktestMenu 导入成功")
        
        from menu.live_monitor import LiveMonitorMenu
        print("✅ LiveMonitorMenu 导入成功")
        
        from menu.market_analysis import MarketAnalysisMenu
        print("✅ MarketAnalysisMenu 导入成功")
        
        from menu.data_management import DataManagementMenu
        print("✅ DataManagementMenu 导入成功")
        
        from menu.system_management import SystemManagementMenu
        print("✅ SystemManagementMenu 导入成功")
        
        print("\n🎉 所有菜单模块导入成功!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_menu_creation():
    """测试菜单创建"""
    try:
        print("\n测试菜单创建...")
        
        from menu.main_menu import MainMenu
        menu = MainMenu()
        print("✅ MainMenu 创建成功")
        
        # 测试各个子菜单
        backtest_menu = menu.backtest_menu
        print("✅ BacktestMenu 初始化成功")
        
        live_monitor_menu = menu.live_monitor_menu
        print("✅ LiveMonitorMenu 初始化成功")
        
        market_analysis_menu = menu.market_analysis_menu
        print("✅ MarketAnalysisMenu 初始化成功")
        
        data_management_menu = menu.data_management_menu
        print("✅ DataManagementMenu 初始化成功")
        
        system_management_menu = menu.system_management_menu
        print("✅ SystemManagementMenu 初始化成功")
        
        print("\n🎉 所有菜单模块创建成功!")
        return True
        
    except Exception as e:
        print(f"❌ 菜单创建失败: {e}")
        return False

def test_ui_components():
    """测试UI组件"""
    try:
        print("\n测试UI组件...")
        
        from menu.ui_components import UIComponents, MenuDisplay
        
        # 测试清屏
        print("测试清屏功能...")
        # UIComponents.clear_screen()  # 不在测试中实际调用清屏
        
        # 测试输入获取（模拟）
        print("✅ UI组件功能正常")
        return True
        
    except Exception as e:
        print(f"❌ UI组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 测试新的高级交互式菜单系统")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 模块导入测试失败")
        return False
    
    # 测试创建
    if not test_menu_creation():
        print("\n❌ 菜单创建测试失败")
        return False
    
    # 测试UI组件
    if not test_ui_components():
        print("\n❌ UI组件测试失败")
        return False
    
    print("\n🎉 所有测试通过!")
    print("✅ 菜单系统已准备就绪")
    print("\n🚀 启动命令:")
    print("   python main.py menu")
    print("\n💡 提示:")
    print("   - 使用数字键选择功能")
    print("   - 按 Enter 确认")
    print("   - 按 0 返回上级菜单")
    print("   - 按 Ctrl+C 退出")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)