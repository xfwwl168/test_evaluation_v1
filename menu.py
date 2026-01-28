#!/usr/bin/env python
# ============================================================================
# 文件: menu.py
# ============================================================================
"""
量化交易引擎 - 交互式菜单系统 v2.0

功能:
- 系统管理 (初始化/更新/诊断)
- 策略回测 (RSRS/动量/短线/Alpha Hunter)
- 市场分析 (扫描/诊断)
- 实战模式 (模拟交易/信号监控)
- 高级功能 (节点测试/数据库管理)
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# 确保项目根目录在路径中
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印头部"""
    print("=" * 70)
    print("                    🚀 量化交易引擎 v2.0                    ")
    print("=" * 70)


def print_menu():
    """打印主菜单"""
    print("""
📋 主菜单
----------------------------------------------------------------------
  【系统管理】
    1.  📦 初始化数据库 (首次运行必选)
    2.  📈 每日数据更新
    3.  🔧 系统信息查看
    4.  🔍 环境诊断检查

  【策略回测】
    5.  🎯 RSRS 趋势策略回测
    6.  📊 动量策略回测
    7.  ⚡ 短线 RSRS 策略回测 (高胜率)
    8.  🏆 Alpha Hunter 策略回测 
    9.  🚀 Alpha Hunter V2 策略回测 (私募级)    # ← 新增
    10.  ⚙️  自定义回测参数

  【市场分析】
    11. 🔎 全市场扫描 (今日金股)
    12. 🏥 单股诊断分析
    13. 📈 多股对比分析

  【实战模式】
    14. 🎮 模拟交易 (Paper Trading)
    15. 📡 实时信号监控
    16. 📋 今日交易计划生成

  【高级功能】
    17. 🌐 节点速度测试
    18. 💾 数据库管理
    19. 📝 查看日志
    20. 🧪 运行单元测试

  【其他】
    0.  🚪 退出系统
----------------------------------------------------------------------""")


def run_command(cmd: str, show_output: bool = True):
    """运行命令"""
    print(f"\n执行命令: {cmd}")
    print("=" * 70)
    
    if show_output:
        result = subprocess.run(cmd, shell=True)
        return result.returncode
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr


def pause():
    """暂停等待"""
    input("\n按 Enter 继续...")


def get_input(prompt: str, default: str = None) -> str:
    """获取用户输入"""
    if default:
        user_input = input(f"{prompt} (默认: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def get_backtest_params():
    """获取回测参数"""
    print()
    start = get_input("开始日期", "2020-01-01")
    end = get_input("结束日期", "2023-12-31")
    capital = get_input("初始资金", "1000000")
    freq = get_input("调仓频率 D/W/M", "W")
    return start, end, capital, freq


# ============================================================================
# 系统管理功能
# ============================================================================

def menu_init_database():
    """初始化数据库"""
    print_header()
    print("\n📦 初始化数据库")
    print("=" * 70)
    print("⚠️  注意: 首次初始化需要下载全量历史数据，约需 30-60 分钟")
    print("=" * 70)
    
    confirm = input("\n确认开始初始化? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    workers = get_input("并行进程数", "4")
    run_command(f"python main.py init --workers {workers}")


def menu_daily_update():
    """每日更新"""
    print_header()
    print("\n📈 每日数据更新")
    print("=" * 70)
    
    full = input("是否全量更新? (y/n, 默认增量): ").strip().lower()
    
    if full == 'y':
        run_command("python main.py update --full")
    else:
        run_command("python main.py update")


def menu_system_info():
    """系统信息"""
    print_header()
    print("\n🔧 系统信息")
    print("=" * 70)
    run_command("python main.py info")


def menu_env_check():
    """环境检查"""
    print_header()
    print("\n🔍 环境诊断")
    print("=" * 70)
    run_command("python check_env.py")


# ============================================================================
# 策略回测功能
# ============================================================================

def menu_backtest_rsrs():
    """RSRS 策略回测"""
    print_header()
    print("\n🎯 RSRS 趋势策略回测")
    print("=" * 70)
    print("""
策略说明:
  - 基于阻力支撑相对强度 (RSRS) 指标
  - R² 过滤确保信号有效性
  - 量价共振增强信号质量
  - 吊灯止损保护利润
  - 适合中长线趋势跟踪
""")
    
    start, end, capital, freq = get_backtest_params()
    run_command(f"python main.py backtest --strategy rsrs --start {start} --end {end} --capital {capital} --freq {freq}")


def menu_backtest_momentum():
    """动量策略回测"""
    print_header()
    print("\n📊 动量策略回测")
    print("=" * 70)
    print("""
策略说明:
  - 选择过去 N 日涨幅最大的股票
  - 波动率倒数加权仓位管理
  - 均值回归作为离场信号
  - 适合趋势明显的市场环境
""")
    
    start, end, capital, freq = get_backtest_params()
    run_command(f"python main.py backtest --strategy momentum --start {start} --end {end} --capital {capital} --freq {freq}")


def menu_backtest_short_term():
    """短线策略回测"""
    print_header()
    print("\n⚡ 短线 RSRS 策略回测 (高胜率)")
    print("=" * 70)
    print("""
策略说明:
  - 修正版 RSRS (R² 加权 + 偏度修正)
  - 严格入场: RSRS>0.7 + R²>0.8 + 放量突破 + 趋势共振
  - 动态离场: 固定止损 3% + ATR 移动止盈 + MA5 趋势
  - 波动率仓位管理
  - 适合短线操作 (持仓 1-5 天)
""")
    
    start, end, capital, freq = get_backtest_params()
    
    # 短线策略默认日度检查
    if freq == 'W':
        print("\n💡 提示: 短线策略建议使用日度调仓 (D)")
        freq = get_input("调仓频率 D/W/M", "D")
    
    run_command(f"python main.py backtest --strategy short_term_rsrs --start {start} --end {end} --capital {capital} --freq {freq}")


def menu_backtest_alpha_hunter():
    """Alpha Hunter 策略回测"""
    print_header()
    print("\n🏆 Alpha Hunter 策略回测 (私募级)")
    print("=" * 70)
    print("""
策略说明:
  - 目标: 年化 >30%, 回撤 <10%
  - 5重入场条件: RSRS + R² + MA趋势 + 换手率 + 压力距离
  - T+1 必杀卖出: 开盘15分钟未涨2%且跌破昨收
  - 动态移动锁利: 每+3%利润 → 止损上移2%
  - Kelly 准则仓位管理
  - 最大持仓 2 天
""")
    
    start, end, capital, freq = get_backtest_params()
    
    # Alpha Hunter 必须日度
    if freq != 'D':
        print("\n⚠️ Alpha Hunter 策略必须使用日度调仓")
        freq = 'D'
    
    # 使用自定义脚本运行
    script_path = ROOT_DIR / "examples" / "run_alpha_hunter.py"
    if script_path.exists():
        run_command(f"python {script_path}")
    else:
        run_command(f"python main.py backtest --strategy alpha_hunter_v1 --start {start} --end {end} --capital {capital} --freq D")


def menu_backtest_alpha_hunter_v2():
    """Alpha Hunter V2 策略回测"""
    print_header()
    print("\n🚀 Alpha Hunter V2 策略回测 (私募级)")
    print("=" * 70)
    print("""
策略说明:
  • 自适应 RSRS (市场状态感知偏度修正)
  • 5重入场条件: RSRS + R² + 信号质量 + MA趋势 + 压力距离
  • T+1 必杀卖出: 开盘15分钟涨幅<2% + 跌破昨收
  • 动态移动锁利: 每+3%利润 → 止损上移2%
  • Kelly 准则动态仓位
  • 目标: 年化>30%, 回撤<10%
""")

    start, end, capital, freq = get_backtest_params()

    if freq != 'D':
        print("\n⚠️ Alpha Hunter V2 必须使用日度调仓")
        freq = 'D'

    # 运行示例脚本
    script_path = ROOT_DIR / "examples" / "run_alpha_hunter_v2.py"
    if script_path.exists():
        run_command(f"python {script_path}")
    else:
        run_command(
            f"python main.py backtest --strategy alpha_hunter_v2 --start {start} --end {end} --capital {capital} --freq D")


def menu_backtest_custom():
    """自定义回测"""
    print_header()
    print("\n⚙️ 自定义回测参数")
    print("=" * 70)
    
    print("\n可用策略:")
    print("  1. rsrs            - RSRS 趋势策略")
    print("  2. momentum        - 动量策略")
    print("  3. short_term_rsrs - 短线 RSRS 策略")
    print("  4. alpha_hunter_v1 - Alpha Hunter 策略")
    
    strategy = get_input("\n策略名称", "rsrs")
    start, end, capital, freq = get_backtest_params()
    
    # 高级参数
    print("\n高级参数 (直接回车使用默认值):")
    top_n = get_input("选股数量 top_n", "30")
    
    cmd = f"python main.py backtest --strategy {strategy} --start {start} --end {end} --capital {capital} --freq {freq}"
    run_command(cmd)


# ============================================================================
# 市场分析功能
# ============================================================================

def menu_market_scan():
    """全市场扫描"""
    print_header()
    print("\n🔎 全市场扫描")
    print("=" * 70)
    
    date = get_input("扫描日期 (默认: 今天, YYYY-MM-DD)", "")
    top_n = get_input("显示数量", "50")
    
    cmd = f"python main.py scan --top {top_n}"
    if date:
        cmd += f" --date {date}"
    
    run_command(cmd)


def menu_stock_diagnose():
    """单股诊断"""
    print_header()
    print("\n🏥 单股诊断")
    print("=" * 70)
    
    code = get_input("请输入股票代码 (如 000001)")
    
    if not code:
        print("❌ 请输入有效的股票代码")
        return
    
    run_command(f"python main.py diagnose {code}")


def menu_multi_stock_compare():
    """多股对比"""
    print_header()
    print("\n📈 多股对比分析")
    print("=" * 70)
    
    codes = get_input("请输入股票代码 (用逗号分隔, 如 000001,600519,000858)")
    
    if not codes:
        print("❌ 请输入有效的股票代码")
        return
    
    code_list = [c.strip() for c in codes.split(',')]
    
    print(f"\n正在分析 {len(code_list)} 只股票...")
    for code in code_list:
        print(f"\n{'='*70}")
        print(f"📊 {code}")
        print('='*70)
        run_command(f"python main.py diagnose {code}")


# ============================================================================
# 实战模式功能
# ============================================================================

def menu_paper_trading():
    """模拟交易"""
    print_header()
    print("\n🎮 模拟交易 (Paper Trading)")
    print("=" * 70)
    print("""
模拟交易模式说明:
  - 使用真实行情数据
  - 模拟订单执行 (不产生真实交易)
  - 记录交易日志和绩效
  - 支持多策略同时运行
""")
    
    print("\n可用策略:")
    print("  1. rsrs            - RSRS 趋势策略")
    print("  2. momentum        - 动量策略") 
    print("  3. short_term_rsrs - 短线 RSRS 策略")
    print("  4. alpha_hunter_v1 - Alpha Hunter 策略")
    
    strategy = get_input("\n选择策略", "short_term_rsrs")
    capital = get_input("模拟资金", "1000000")
    
    print("\n⏳ 启动模拟交易...")
    print("=" * 70)
    
    # 创建模拟交易脚本
    script = f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
from core.database import StockDatabase
from strategy import StrategyRegistry
from engine.backtest import BacktestEngine

# 获取最近的交易数据
db = StockDatabase()
stats = db.get_stats()

if stats.get('total_rows', 0) == 0:
    print("❌ 数据库为空，请先初始化")
    sys.exit(1)

end_date = str(stats.get('max_date', datetime.now().strftime('%Y-%m-%d')))[:10]
start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')

print(f"📅 模拟交易区间: {{start_date}} ~ {{end_date}}")
print(f"💰 模拟资金: {{int({capital}):,}}")
print(f"📊 使用策略: {strategy}")
print()

# 运行回测作为模拟
engine = BacktestEngine(initial_capital={capital})

try:
    strategy_cls = StrategyRegistry.get("{strategy}")
    engine.add_strategy(strategy_cls())
except Exception as e:
    print(f"❌ 策略加载失败: {{e}}")
    sys.exit(1)

results = engine.run(start_date, end_date, rebalance_freq='D')

# 显示最近交易
for name, result in results.items():
    trades = result.get_trades()
    if not trades.empty:
        print("\\n📋 最近交易记录:")
        print(trades.tail(10).to_string())
'''
    
    # 写入临时文件并执行
    temp_script = ROOT_DIR / "temp_paper_trading.py"
    temp_script.write_text(script, encoding='utf-8')
    
    run_command(f"python {temp_script}")
    
    # 清理
    temp_script.unlink(missing_ok=True)

    def menu_signal_monitor():
        """实时信号监控 (完善版)"""
        print_header()
        print("\n📡 实时信号监控")
        print("=" * 70)
        print("""
    信号监控模式:
      - 使用真实策略逻辑生成信号
      - 区分入场信号(🟢)和离场信号(🔴)
      - 跟踪虚拟持仓状态
      - 支持多策略并行监控
    """)

        print("\n可用策略:")
        print("  1. rsrs            - RSRS 趋势策略")
        print("  2. momentum        - 动量策略")
        print("  3. short_term_rsrs - 短线 RSRS 策略")
        print("  4. alpha_hunter_v1 - Alpha Hunter 策略")
        print("  5. all             - 全部策略")

        choice = get_input("\n选择策略 (多个用逗号分隔)", "3")

        # 解析策略选择
        strategy_map = {
            '1': ['rsrs'],
            '2': ['momentum'],
            '3': ['short_term_rsrs'],
            '4': ['alpha_hunter_v1'],
            '5': None  # None = 全部
        }

        if choice in strategy_map:
            strategies = strategy_map[choice]
        else:
            # 支持直接输入策略名
            strategies = [s.strip() for s in choice.split(',')]

        interval = int(get_input("扫描间隔 (秒)", "60"))

        strategy_names = strategies if strategies else "全部"
        print(f"\n⏳ 启动信号监控...")
        print(f"   策略: {strategy_names}")
        print(f"   间隔: {interval} 秒")
        print("   按 Ctrl+C 停止")
        print("=" * 70)

        # 检查 live 模块是否存在
        live_module_exists = (ROOT_DIR / "live" / "signal_monitor.py").exists()

        if live_module_exists:
            # 使用完整的信号监控模块
            script = f'''
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from live.signal_monitor import run_signal_monitor

    run_signal_monitor(
        strategies={strategies!r},
        interval={interval}
    )
    '''
        else:
            # 回退到简化版本 (使用扫描器)
            script = f'''
    import sys
    import time
    from pathlib import Path
    from datetime import datetime
    sys.path.insert(0, str(Path(__file__).parent))

    from analysis.scanner import MarketScanner

    scanner = MarketScanner()
    strategies = {strategies!r}
    interval = {interval}

    print("📡 信号监控已启动 (简化模式)")
    print(f"   策略: {{strategies if strategies else '全市场扫描'}}")
    print("=" * 70)

    scan_count = 0
    while True:
        try:
            scan_count += 1
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\\n[{{now}}] 第 {{scan_count}} 次扫描...")

            result = scanner.scan(top_n=10)

            if not result.empty:
                print("\\n🌟 发现信号:")
                print(result.to_string())
            else:
                print("   暂无符合条件的信号")

            print(f"\\n⏳ 等待下次扫描 ({{interval}}秒)...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\\n\\n👋 监控已停止")
            break
        except Exception as e:
            print(f"\\n⚠️ 扫描出错: {{e}}")
            time.sleep(10)
    '''

        temp_script = ROOT_DIR / "temp_monitor.py"
        temp_script.write_text(script, encoding='utf-8')

        try:
            run_command(f"python {temp_script}")
        finally:
            temp_script.unlink(missing_ok=True)
    
    print("\n选择监控策略:")
    print("  1. short_term_rsrs - 短线信号")
    print("  2. alpha_hunter_v1 - Alpha 信号")
    print("  3. all             - 全部策略")
    
    choice = get_input("选择", "1")
    
    if choice == '1':
        strategy = 'short_term_rsrs'
    elif choice == '2':
        strategy = 'alpha_hunter_v1'
    else:
        strategy = 'all'
    
    print(f"\n⏳ 启动信号监控 (策略: {strategy})...")
    print("按 Ctrl+C 停止监控")
    print("=" * 70)
    
    # 信号监控脚本
    script = f'''
import sys
import time
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

from analysis.scanner import MarketScanner

scanner = MarketScanner()

print("📡 信号监控已启动")
print("=" * 70)

scan_count = 0
while True:
    try:
        scan_count += 1
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\\n[{{now}}] 第 {{scan_count}} 次扫描...")
        
        # 执行扫描
        result = scanner.scan(top_n=10)
        
        if not result.empty:
            print("\\n🌟 发现信号:")
            print(result.to_string())
        else:
            print("   暂无符合条件的信号")
        
        # 等待 60 秒
        print("\\n⏳ 等待下次扫描 (60秒)...")
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\\n\\n👋 监控已停止")
        break
    except Exception as e:
        print(f"\\n⚠️ 扫描出错: {{e}}")
        time.sleep(10)
'''
    
    temp_script = ROOT_DIR / "temp_signal_monitor.py"
    temp_script.write_text(script, encoding='utf-8')
    
    run_command(f"python {temp_script}")
    
    temp_script.unlink(missing_ok=True)


def menu_trading_plan():
    """今日交易计划"""
    print_header()
    print("\n📋 今日交易计划生成")
    print("=" * 70)
    
    print("\n选择策略:")
    print("  1. short_term_rsrs - 短线策略")
    print("  2. alpha_hunter_v1 - Alpha Hunter")
    print("  3. rsrs            - RSRS 趋势")
    
    choice = get_input("选择", "1")
    
    strategies = {
        '1': 'short_term_rsrs',
        '2': 'alpha_hunter_v1', 
        '3': 'rsrs'
    }
    strategy = strategies.get(choice, 'short_term_rsrs')
    
    capital = get_input("可用资金", "1000000")
    
    print(f"\n⏳ 生成交易计划 (策略: {strategy})...")
    print("=" * 70)
    
    script = f'''
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

from analysis.scanner import MarketScanner

scanner = MarketScanner()
today = datetime.now().strftime('%Y-%m-%d')

print(f"📅 日期: {{today}}")
print(f"💰 可用资金: {{int({capital}):,}}")
print(f"📊 策略: {strategy}")
print()

# 扫描候选股
print("🔍 扫描市场...")
result = scanner.scan(top_n=20)

if result.empty:
    print("\\n❌ 今日无符合条件的交易标的")
else:
    print("\\n" + "=" * 70)
    print("📋 今日交易计划")
    print("=" * 70)
    
    capital = {capital}
    max_positions = 10
    position_size = capital / max_positions
    
    print(f"\\n💰 单笔仓位: {{position_size:,.0f}}")
    print(f"📊 最大持仓: {{max_positions}} 只")
    print()
    
    print("【买入候选】")
    print("-" * 70)
    for i, row in result.head(10).iterrows():
        code = row.get('代码', row.get('code', 'N/A'))
        price = row.get('收盘价', row.get('close', 0))
        score = row.get('综合评分', row.get('alpha_score', 0))
        
        if price > 0:
            shares = int(position_size / price / 100) * 100
            print(f"  {{i:>2}}. {{code}} | 价格: {{price:>8.2f}} | 评分: {{score:>6.4f}} | 建议: {{shares}} 股")
    
    print()
    print("⚠️ 提示: 以上仅为参考，请结合实际情况决策")
    print("=" * 70)
'''
    
    temp_script = ROOT_DIR / "temp_trading_plan.py"
    temp_script.write_text(script, encoding='utf-8')
    
    run_command(f"python {temp_script}")
    
    temp_script.unlink(missing_ok=True)


# ============================================================================
# 高级功能
# ============================================================================

def menu_node_test():
    """节点速度测试"""
    print_header()
    print("\n🌐 TDX 节点速度测试")
    print("=" * 70)
    
    script = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.node_scanner import TDXNodeScanner

print("⏳ 正在测试节点速度...")
print()

scanner = TDXNodeScanner(timeout=3.0)
results = scanner.scan_threaded()

print("📊 节点测试结果 (按延迟排序):")
print("-" * 60)
print(f"{'排名':<4} {'节点名称':<12} {'地址':<20} {'延迟(ms)':<10} {'状态'}")
print("-" * 60)

for i, node in enumerate(results[:10], 1):
    status = "✓" if node['status'] == 'ok' else "✗"
    latency = f"{node['latency_ms']:.1f}" if node['latency_ms'] > 0 else "超时"
    print(f"{i:<4} {node['name']:<12} {node['host']:<20} {latency:<10} {status}")

print("-" * 60)
print(f"\\n共测试 {len(results)} 个节点, 可用 {sum(1 for n in results if n['status'] == 'ok')} 个")
'''
    
    temp_script = ROOT_DIR / "temp_node_test.py"
    temp_script.write_text(script, encoding='utf-8')
    
    run_command(f"python {temp_script}")
    
    temp_script.unlink(missing_ok=True)


def menu_database_manage():
    """数据库管理"""
    print_header()
    print("\n💾 数据库管理")
    print("=" * 70)
    
    print("\n操作选项:")
    print("  1. 查看数据库统计")
    print("  2. 压缩数据库 (VACUUM)")
    print("  3. 检查数据完整性")
    print("  4. 导出数据 (CSV)")
    print("  5. 返回")
    
    choice = get_input("\n选择操作", "1")
    
    if choice == '1':
        script = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import StockDatabase
from config import settings

db = StockDatabase(str(settings.path.DB_PATH))
stats = db.get_stats()

print("\\n📊 数据库统计")
print("-" * 50)
print(f"  数据库路径: {settings.path.DB_PATH}")
print(f"  总记录数:   {stats.get('total_rows', 0):,}")
print(f"  股票数量:   {stats.get('unique_stocks', 0):,}")
print(f"  交易日数:   {stats.get('trading_days', 0):,}")
print(f"  日期范围:   {stats.get('date_range', ('N/A', 'N/A'))}")
print(f"  文件大小:   {stats.get('db_size_mb', 0):.2f} MB")
print("-" * 50)
'''
        temp_script = ROOT_DIR / "temp_db_stats.py"
        temp_script.write_text(script, encoding='utf-8')
        run_command(f"python {temp_script}")
        temp_script.unlink(missing_ok=True)
        
    elif choice == '2':
        print("\n⏳ 压缩数据库...")
        script = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import StockDatabase
from config import settings
import os

db_path = str(settings.path.DB_PATH)
before_size = os.path.getsize(db_path) / 1024 / 1024 if os.path.exists(db_path) else 0

db = StockDatabase(db_path)
db.vacuum()

after_size = os.path.getsize(db_path) / 1024 / 1024

print(f"\\n✓ 压缩完成")
print(f"  压缩前: {before_size:.2f} MB")
print(f"  压缩后: {after_size:.2f} MB")
print(f"  节省:   {before_size - after_size:.2f} MB ({(1 - after_size/before_size)*100:.1f}%)")
'''
        temp_script = ROOT_DIR / "temp_vacuum.py"
        temp_script.write_text(script, encoding='utf-8')
        run_command(f"python {temp_script}")
        temp_script.unlink(missing_ok=True)
        
    elif choice == '3':
        print("\n⏳ 检查数据完整性...")
        script = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.updater import DataUpdater

updater = DataUpdater()
report = updater.check_integrity()

print("\\n📋 数据完整性报告")
print("-" * 50)
print(f"  总记录数:     {report.get('total_rows', 0):,}")
print(f"  股票数量:     {report.get('stocks', 0):,}")
print(f"  交易日数:     {report.get('trading_days', 0):,}")
print(f"  不完整股票:   {report.get('incomplete_stocks', 0):,}")
print("-" * 50)

if report.get('incomplete_stocks', 0) > 0:
    print("\\n⚠️ 发现不完整数据，建议运行全量更新")
else:
    print("\\n✓ 数据完整性良好")
'''
        temp_script = ROOT_DIR / "temp_check.py"
        temp_script.write_text(script, encoding='utf-8')
        run_command(f"python {temp_script}")
        temp_script.unlink(missing_ok=True)
        
    elif choice == '4':
        code = get_input("股票代码 (如 000001, 留空导出全部)", "")
        output = get_input("输出文件名", "export.csv")
        
        print(f"\n⏳ 导出数据到 {output}...")
        script = f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import StockDatabase
from config import settings

db = StockDatabase(str(settings.path.DB_PATH))

code = "{code}"
if code:
    df = db.get_stock_history(code)
    print(f"导出 {{code}}: {{len(df)}} 条记录")
else:
    with db.connect() as conn:
        df = conn.execute("SELECT * FROM daily_bars LIMIT 100000").fetchdf()
    print(f"导出全部数据 (前 100000 条)")

df.to_csv("{output}", index=False, encoding='utf-8-sig')
print(f"\\n✓ 已导出到 {output}")
'''
        temp_script = ROOT_DIR / "temp_export.py"
        temp_script.write_text(script, encoding='utf-8')
        run_command(f"python {temp_script}")
        temp_script.unlink(missing_ok=True)


def menu_view_logs():
    """查看日志"""
    print_header()
    print("\n📝 查看日志")
    print("=" * 70)
    
    from config import settings
    log_dir = settings.path.LOG_DIR
    
    if not log_dir.exists():
        print("❌ 日志目录不存在")
        return
    
    log_files = sorted(log_dir.glob("*.log"), reverse=True)
    
    if not log_files:
        print("❌ 没有日志文件")
        return
    
    print("\n可用日志文件:")
    for i, f in enumerate(log_files[:10], 1):
        size = f.stat().st_size / 1024
        print(f"  {i}. {f.name} ({size:.1f} KB)")
    
    choice = get_input("\n选择文件编号", "1")
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(log_files):
            log_file = log_files[idx]
            lines = get_input("显示最后 N 行", "50")
            
            print(f"\n📄 {log_file.name} (最后 {lines} 行)")
            print("=" * 70)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                for line in all_lines[-int(lines):]:
                    print(line.rstrip())
    except:
        print("❌ 无效选择")


def menu_run_tests():
    """运行测试"""
    print_header()
    print("\n🧪 运行单元测试")
    print("=" * 70)
    
    print("\n测试选项:")
    print("  1. 全部测试")
    print("  2. 因子测试")
    print("  3. 引擎测试")
    print("  4. 策略测试")
    
    choice = get_input("选择", "1")
    
    test_map = {
        '1': 'tests/',
        '2': 'tests/test_factors.py',
        '3': 'tests/test_engine.py',
        '4': 'tests/test_short_term_strategy.py'
    }
    
    test_path = test_map.get(choice, 'tests/')
    run_command(f"python -m pytest {test_path} -v")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("请选择功能 [0-20]: ").strip()
        
        clear_screen()
        
        try:
            if choice == '0':
                print("\n👋 感谢使用，再见!")
                break
            elif choice == '1':
                menu_init_database()
            elif choice == '2':
                menu_daily_update()
            elif choice == '3':
                menu_system_info()
            elif choice == '4':
                menu_env_check()
            elif choice == '5':
                menu_backtest_rsrs()
            elif choice == '6':
                menu_backtest_momentum()
            elif choice == '7':
                menu_backtest_short_term()
            elif choice == '8':
                menu_backtest_alpha_hunter()
            elif choice == '9':
                menu_backtest_alpha_hunter_v2()  # ← 新增
            elif choice == '10':
                menu_backtest_custom()
            elif choice == '11':
                menu_market_scan()
            elif choice == '12':
                menu_stock_diagnose()
            elif choice == '13':
                menu_multi_stock_compare()
            elif choice == '14':
                menu_paper_trading()
            elif choice == '15':
                menu_signal_monitor()
            elif choice == '16':
                menu_trading_plan()
            elif choice == '17':
                menu_node_test()
            elif choice == '18':
                menu_database_manage()
            elif choice == '19':
                menu_view_logs()
            elif choice == '20':
                menu_run_tests()
            else:
                print("无效选择，请重试")
        except KeyboardInterrupt:
            print("\n\n操作已取消")
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        pause()


if __name__ == "__main__":
    main()