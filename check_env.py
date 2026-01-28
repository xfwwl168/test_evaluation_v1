#!/usr/bin/env python
"""
环境检查脚本 - 诊断量化引擎依赖问题
"""
import sys
from pathlib import Path

def print_section(title):
    """打印分节标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

def check_python_version():
    """检查 Python 版本"""
    print_section("Python 环境")
    print(f"Python 版本: {sys.version}")
    print(f"可执行文件: {sys.executable}")
    
    if sys.version_info < (3, 9):
        print("❌ 需要 Python 3.9 或更高版本")
        return False
    else:
        print("✓ Python 版本符合要求")
        return True

def check_packages():
    """检查依赖包"""
    print_section("依赖包检查")
    
    required_packages = {
        'pandas': '>=1.5.0',
        'numpy': '>=1.23.0',
        'duckdb': '>=0.9.0',
        'pytdx': '>=1.72',
        'click': '>=8.0',
        'pytest': '>=7.0 (可选)'
    }
    
    all_ok = True
    for pkg, version in required_packages.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', '未知')
            print(f"  ✓ {pkg:12} {ver:15} (要求 {version})")
        except ImportError:
            print(f"  ✗ {pkg:12} - 未安装! (要求 {version})")
            all_ok = False
    
    return all_ok

def check_directory_structure():
    """检查目录结构"""
    print_section("目录结构检查")
    
    required_dirs = [
        'core',
        'config',
        'utils',
        'strategy',
        'engine',
        'factors',
        'analysis'
    ]
    
    all_ok = True
    for dirname in required_dirs:
        path = Path(dirname)
        exists = path.exists() and path.is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {dirname}/")
        if not exists:
            all_ok = False
    
    return all_ok

def check_key_files():
    """检查关键文件"""
    print_section("关键文件检查")
    
    key_files = [
        ('main.py', '主入口'),
        ('core/__init__.py', '核心模块导出'),
        ('core/updater.py', '数据更新器'),
        ('core/database.py', '数据库接口'),
        ('core/downloader.py', '数据下载器'),
        ('core/node_scanner.py', '节点扫描器'),
        ('config/__init__.py', '配置模块导出'),
        ('config/settings.py', '配置定义'),
        ('utils/__init__.py', '工具模块导出'),
        ('utils/logger.py', '日志工具')
    ]
    
    all_ok = True
    for filepath, desc in key_files:
        path = Path(filepath)
        exists = path.exists() and path.is_file()
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath:30} ({desc})")
        if not exists:
            all_ok = False
    
    return all_ok

def check_imports():
    """测试关键模块导入"""
    print_section("模块导入测试")
    
    test_imports = [
        ('config', 'settings', '配置对象'),
        ('utils.logger', 'setup_logging', '日志设置'),
        ('utils.logger', 'get_logger', '日志获取'),
        ('core.updater', 'DataUpdater', '数据更新器'),
        ('core.database', 'StockDatabase', '数据库接口'),
        ('core.downloader', 'StockDownloader', '数据下载器'),
        ('core.node_scanner', 'NodeScanner', '节点扫描器')
    ]
    
    all_ok = True
    for module_name, obj_name, desc in test_imports:
        try:
            module = __import__(module_name, fromlist=[obj_name])
            obj = getattr(module, obj_name)
            print(f"  ✓ from {module_name} import {obj_name}")
            print(f"    → {desc}")
        except ImportError as e:
            print(f"  ✗ from {module_name} import {obj_name}")
            print(f"    错误: {e}")
            all_ok = False
        except AttributeError as e:
            print(f"  ✗ {obj_name} 不在 {module_name} 中")
            print(f"    错误: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ✗ 导入 {module_name}.{obj_name} 时发生异常")
            print(f"    错误: {type(e).__name__}: {e}")
            all_ok = False
    
    return all_ok

def check_data_directory():
    """检查数据目录"""
    print_section("数据目录检查")
    
    data_dir = Path('data')
    if not data_dir.exists():
        print(f"  ℹ data/ 目录不存在 (首次运行时会自动创建)")
    else:
        print(f"  ✓ data/ 目录存在")
        
        # 检查子目录
        subdirs = ['logs', 'cache']
        for subdir in subdirs:
            path = data_dir / subdir
            status = "✓" if path.exists() else "○"
            print(f"    {status} {subdir}/")
        
        # 检查数据库文件
        db_file = data_dir / 'stocks_daily.db'
        if db_file.exists():
            size_mb = db_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ 数据库文件存在: {size_mb:.2f} MB")
        else:
            print(f"  ℹ 数据库文件不存在 (运行 init 命令创建)")

def print_summary(results):
    """打印总结"""
    print_section("检查总结")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status:10} {check}")
    
    print()
    if all_passed:
        print("🎉 所有检查通过! 可以运行 python main.py init")
    else:
        print("⚠️  部分检查失败，请根据上述信息修复问题")
        print("   最常见的问题:")
        print("   1. 缺少依赖包 → pip install -r requirements.txt")
        print("   2. core/__init__.py 缺失导出 → 参考修复文档")
        print("   3. 文件结构不完整 → 检查项目完整性")

def main():
    """主函数"""
    print("=" * 60)
    print("  量化引擎环境诊断工具 v1.0")
    print("=" * 60)
    
    results = {
        'Python 版本': check_python_version(),
        '依赖包': check_packages(),
        '目录结构': check_directory_structure(),
        '关键文件': check_key_files(),
        '模块导入': check_imports()
    }
    
    check_data_directory()
    print_summary(results)
    
    print("\n" + "=" * 60)
    
    # 返回退出代码
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
