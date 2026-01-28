# ============================================================================
# 文件: main.py (向量化增强版)
# ============================================================================
#!/usr/bin/env python
"""
量化引擎主入口 - 向量化增强版
"""
import click
import sys
from pathlib import Path

# 确保模块可导入
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils.logger import setup_logging, get_logger
from config import settings


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='详细日志')
@click.pass_context
def cli(ctx, verbose: bool):
    """
    🚀 量化交易引擎 v2.0 (向量化增强版)
    
    使用示例:
    
    \b
    # 初始化数据库
    python main.py init
    
    \b
    # 每日更新
    python main.py update
    
    \b
    # 运行回测（原引擎）
    python main.py backtest --strategy rsrs --start 2020-01-01
    
    \b
    # 快速回测（向量化引擎，10-50x加速）
    python main.py fastbacktest --strategy momentum --start 2020-01-01 --end 2023-12-31
    
    \b
    # 性能对比测试
    python main.py benchmark --strategy momentum --start 2022-01-01 --end 2023-12-31
    
    \b
    # 市场扫描
    python main.py scan --top 30
    
    \b
    # 单股诊断
    python main.py diagnose 000001
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(level='DEBUG' if verbose else 'INFO')


@cli.command()
@click.option('--workers', '-w', default=None, type=int, help='并行进程数')
@click.pass_context
def init(ctx, workers: int):
    """初始化数据库 - 全量下载"""
    from core.updater import DataUpdater
    
    click.echo("📦 初始化数据库...")
    updater = DataUpdater()
    stats = updater.full_update(n_workers=workers)
    click.echo(f"✅ 完成! 下载 {stats['downloaded']} 只股票")


@cli.command()
@click.option('--full', is_flag=True, help='全量更新')
@click.pass_context
def update(ctx, full: bool):
    """更新数据"""
    from core.updater import DataUpdater
    
    click.echo(f"📈 {'全量' if full else '增量'}更新...")
    updater = DataUpdater()
    
    if full:
        stats = updater.full_update()
    else:
        stats = updater.incremental_update()
    
    click.echo(f"✅ 完成! 更新 {stats.get('updated', stats.get('written', 0))} 条")


# ==================== 原始回测引擎 ====================
@cli.command()
@click.option('--strategy', '-s', default='rsrs', help='策略名称 (rsrs/momentum)')
@click.option('--start', default='2020-01-01', help='开始日期')
@click.option('--end', default='2023-12-31', help='结束日期')
@click.option('--capital', default=1000000, type=float, help='初始资金')
@click.option('--freq', default='W', help='调仓频率 (D/W/M)')
@click.pass_context
def backtest(ctx, strategy: str, start: str, end: str, capital: float, freq: str):
    """运行回测（原始引擎）"""
    from engine.backtest import BacktestEngine
    from strategy.rsrs_strategy import RSRSStrategy
    from strategy.momentum_strategy import MomentumStrategy
    
    click.echo(f"🚀 运行回测: {strategy} (原始引擎)")
    
    engine = BacktestEngine(initial_capital=capital)
    
    if strategy == 'rsrs':
        engine.add_strategy(RSRSStrategy())
    elif strategy == 'momentum':
        engine.add_strategy(MomentumStrategy())
    else:
        click.echo(f"❌ 未知策略: {strategy}")
        return
    
    results = engine.run(start, end, rebalance_freq=freq)


# ==================== 向量化快速回测 ====================
@cli.command()
@click.option('--strategy', '-s', required=True, help='策略: momentum/rsrs/composite/reversal')
@click.option('--start', required=True, help='开始日期 YYYY-MM-DD')
@click.option('--end', required=True, help='结束日期 YYYY-MM-DD')
@click.option('--capital', default=1000000, type=float, help='初始资金')
@click.option('--freq', default='W', help='调仓频率 (D/W/M)')
@click.option('--top-n', default=10, type=int, help='持仓数量')
@click.option('--codes', default=None, help='股票池（逗号分隔，如 000001,000002）')
@click.option('--save-plot', is_flag=True, help='保存权益曲线图')
@click.option('--save-csv', is_flag=True, help='保存详细结果到CSV')
@click.pass_context
def fastbacktest(ctx, strategy: str, start: str, end: str, capital: float, 
                 freq: str, top_n: int, codes: str, save_plot: bool, save_csv: bool):
    """
    快速回测（向量化引擎，10-50x加速）
    
    支持的策略:
    - momentum: 动量策略
    - rsrs: RSRS择时
    - composite: 组合Alpha
    - reversal: 短期反转
    
    示例:
    \b
    python main.py fastbacktest --strategy momentum --start 2020-01-01 --end 2023-12-31
    python main.py fastbacktest --strategy rsrs --start 2020-01-01 --end 2023-12-31 --save-plot
    python main.py fastbacktest --strategy composite --start 2020-01-01 --end 2023-12-31 --top-n 20
    """
    import time
    from engine.vectorized_backtest_engine import VectorizedBacktestEngine, BacktestConfig
    from pathlib import Path
    
    click.echo("=" * 70)
    click.echo(f"⚡ 向量化快速回测")
    click.echo("=" * 70)
    click.echo(f"策略:       {strategy}")
    click.echo(f"周期:       {start} → {end}")
    click.echo(f"持仓数:     {top_n} 只")
    click.echo(f"调仓频率:   {freq}")
    click.echo(f"初始资金:   {capital:,.0f}")
    click.echo("=" * 70)
    
    # 解析股票池
    stock_codes = codes.split(',') if codes else None
    if stock_codes:
        click.echo(f"股票池:     {len(stock_codes)} 只")
    
    # 计时
    t0 = time.time()
    
    # 创建配置
    config = BacktestConfig(
        initial_capital=capital,
        rebalance_freq=freq,
        top_n=top_n
    )
    
    # 创建引擎
    engine = VectorizedBacktestEngine(config=config)
    
    try:
        # 执行回测
        click.echo("\n[1/4] 加载数据...")
        engine.load_data(start, end, codes=stock_codes, use_parallel=True)
        
        click.echo("[2/4] 计算因子...")
        if strategy == 'momentum':
            engine.compute_factors('momentum', period=20)
        elif strategy == 'rsrs':
            engine.compute_factors('rsrs', window=18, n=600)
        elif strategy == 'composite':
            engine.compute_factors('composite')
        elif strategy == 'reversal':
            # 使用反转因子（短期跌幅大的）
            import pandas as pd
            from engine.vectorized_backtest_engine import VectorizedFactors
            engine.factors = VectorizedFactors.reversal(engine.data.returns, period=5)
        else:
            click.echo(f"❌ 未知策略: {strategy}")
            click.echo("支持的策略: momentum, rsrs, composite, reversal")
            return
        
        click.echo("[3/4] 生成信号...")
        engine.generate_signals(method='topN', top_n=top_n)
        
        click.echo("[4/4] 运行回测...")
        results = engine.run_backtest()
        
        elapsed = time.time() - t0
        
        # 显示结果
        engine.print_results(results)
        
        # 保存图表
        if save_plot:
            output_dir = Path('data/outputs')
            output_dir.mkdir(exist_ok=True, parents=True)
            
            plot_path = output_dir / f"equity_{strategy}_{start}_{end}.png"
            engine.plot_equity_curve(save_path=str(plot_path))
            click.echo(f"\n📊 权益曲线: {plot_path}")
        
        # 保存CSV
        if save_csv:
            output_dir = Path('data/outputs')
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # 保存权益曲线
            equity_path = output_dir / f"equity_{strategy}_{start}_{end}.csv"
            engine.equity_curve.to_csv(equity_path)
            
            # 保存持仓历史
            positions_path = output_dir / f"positions_{strategy}_{start}_{end}.csv"
            engine.positions_history.to_csv(positions_path)
            
            click.echo(f"💾 数据已保存:")
            click.echo(f"   - {equity_path}")
            click.echo(f"   - {positions_path}")
        
        click.echo(f"\n⏱️  总耗时: {elapsed:.1f}秒")
        click.echo("=" * 70)
        
    except Exception as e:
        click.echo(f"\n❌ 回测失败: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()


# ==================== 性能对比测试 ====================
@cli.command()
@click.option('--strategy', '-s', default='momentum', help='策略名称')
@click.option('--start', default='2022-01-01', help='开始日期')
@click.option('--end', default='2023-12-31', help='结束日期')
@click.option('--capital', default=1000000, type=float, help='初始资金')
@click.option('--top-n', default=10, type=int, help='持仓数量')
@click.pass_context
def benchmark(ctx, strategy: str, start: str, end: str, capital: float, top_n: int):
    """
    性能对比测试（原引擎 vs 向量化引擎）
    
    示例:
    \b
    python main.py benchmark --strategy momentum --start 2022-01-01 --end 2023-12-31
    """
    import time
    
    click.echo("=" * 70)
    click.echo("性能对比测试: 原引擎 vs 向量化引擎")
    click.echo("=" * 70)
    click.echo(f"策略: {strategy}")
    click.echo(f"周期: {start} → {end}")
    click.echo("=" * 70)
    
    # 测试向量化引擎
    click.echo("\n[测试1] 向量化引擎...")
    click.echo("-" * 70)
    
    from engine.vectorized_backtest_engine import VectorizedBacktestEngine, BacktestConfig
    
    config = BacktestConfig(initial_capital=capital, top_n=top_n)
    engine_new = VectorizedBacktestEngine(config=config)
    
    t0 = time.time()
    
    engine_new.load_data(start, end)
    click.echo(f"  ✓ 加载数据: {time.time()-t0:.1f}s")
    
    t1 = time.time()
    engine_new.compute_factors(strategy)
    click.echo(f"  ✓ 计算因子: {time.time()-t1:.1f}s")
    
    t2 = time.time()
    engine_new.generate_signals(method='topN', top_n=top_n)
    click.echo(f"  ✓ 生成信号: {time.time()-t2:.1f}s")
    
    t3 = time.time()
    results_new = engine_new.run_backtest()
    click.echo(f"  ✓ 运行回测: {time.time()-t3:.1f}s")
    
    time_new = time.time() - t0
    click.echo(f"\n总耗时: {time_new:.1f}秒")
    
    # 显示结果
    click.echo("\n向量化引擎结果:")
    click.echo("-" * 70)
    click.echo(f"年化收益: {results_new['annual_return']:.2%}")
    click.echo(f"夏普比率: {results_new['sharpe_ratio']:.2f}")
    click.echo(f"最大回撤: {results_new['max_drawdown']:.2%}")
    
    # 测试原引擎（可选，如果想对比）
    try:
        click.echo("\n\n[测试2] 原引擎...")
        click.echo("-" * 70)
        
        from engine.backtest import BacktestEngine
        from strategy.momentum_strategy import MomentumStrategy
        from strategy.rsrs_strategy import RSRSStrategy
        
        engine_old = BacktestEngine(initial_capital=capital)
        
        if strategy == 'momentum':
            engine_old.add_strategy(MomentumStrategy())
        elif strategy == 'rsrs':
            engine_old.add_strategy(RSRSStrategy())
        else:
            raise ValueError(f"原引擎不支持策略: {strategy}")
        
        t0 = time.time()
        results_old = engine_old.run(start, end, rebalance_freq='W')
        time_old = time.time() - t0
        
        click.echo(f"\n总耗时: {time_old:.1f}秒")
        
        # 对比
        click.echo("\n\n" + "=" * 70)
        click.echo("性能对比")
        click.echo("=" * 70)
        click.echo(f"{'引擎':12} {'耗时':>12} {'加速比':>12}")
        click.echo("-" * 70)
        click.echo(f"{'原引擎':12} {time_old:>10.1f}s {1.0:>11.1f}x")
        click.echo(f"{'向量化引擎':12} {time_new:>10.1f}s {time_old/time_new:>11.1f}x")
        click.echo("=" * 70)
        
        if time_old / time_new > 10:
            click.echo("🚀 性能提升: 超过10倍加速!")
        elif time_old / time_new > 5:
            click.echo("⚡ 性能提升: 5-10倍加速")
        elif time_old / time_new > 2:
            click.echo("✨ 性能提升: 2-5倍加速")
        else:
            click.echo("💡 性能提升: 略有加速")
    
    except Exception as e:
        click.echo(f"\n⚠️  原引擎测试跳过: {e}")
        click.echo("（原引擎可能不支持该策略或配置）")


# ==================== 市场扫描 ====================
@cli.command()
@click.option('--date', '-d', default=None, help='扫描日期')
@click.option('--top', '-n', default=50, type=int, help='输出数量')
@click.pass_context
def scan(ctx, date: str, top: int):
    """全市场扫描"""
    from analysis.scanner import MarketScanner
    from analysis.report import ReportGenerator
    
    click.echo("🔍 扫描市场...")
    
    scanner = MarketScanner()
    result = scanner.scan(target_date=date, top_n=top)
    
    if not result.empty:
        ReportGenerator.print_golden_stocks(result)
    else:
        click.echo("未找到符合条件的股票")


# ==================== 单股诊断 ====================
@cli.command()
@click.argument('code')
@click.pass_context
def diagnose(ctx, code: str):
    """单股诊断"""
    from analysis.stock_doctor import StockDoctor
    
    click.echo(f"🔬 诊断 {code}...")
    
    doctor = StockDoctor()
    result = doctor.diagnose(code)
    report = doctor.generate_report(result)
    click.echo(report)


# ==================== 系统信息 ====================
@cli.command()
@click.pass_context
def info(ctx):
    """显示系统信息"""
    click.echo("=" * 60)
    click.echo("📊 量化引擎信息 v2.0 (向量化增强版)")
    click.echo("=" * 60)
    click.echo(f"数据库:     {settings.path.DB_PATH}")
    click.echo(f"日志目录:   {settings.path.LOG_DIR}")
    click.echo(f"初始资金:   {settings.backtest.INITIAL_CAPITAL:,.0f}")
    click.echo(f"RSRS窗口:   {settings.factor.RSRS_WINDOW}")
    click.echo("\n新增功能:")
    click.echo("  ⚡ 向量化回测引擎 (10-50x加速)")
    click.echo("  📊 多因子组合策略")
    click.echo("  🎨 权益曲线可视化")
    click.echo("  💾 详细结果导出")
    click.echo("=" * 60)


# ==================== 批量回测 ====================
@cli.command()
@click.option('--strategies', default='momentum,rsrs,composite', help='策略列表（逗号分隔）')
@click.option('--start', default='2020-01-01', help='开始日期')
@click.option('--end', default='2023-12-31', help='结束日期')
@click.option('--capital', default=1000000, type=float, help='初始资金')
@click.option('--top-n', default=10, type=int, help='持仓数量')
@click.pass_context
def batchtest(ctx, strategies: str, start: str, end: str, capital: float, top_n: int):
    """
    批量回测多个策略
    
    示例:
    \b
    python main.py batchtest --strategies momentum,rsrs,composite --start 2020-01-01 --end 2023-12-31
    """
    import time
    import pandas as pd
    from engine.vectorized_backtest_engine import VectorizedBacktestEngine, BacktestConfig
    
    strategy_list = strategies.split(',')
    
    click.echo("=" * 70)
    click.echo(f"批量回测: {len(strategy_list)} 个策略")
    click.echo("=" * 70)
    
    results_summary = []
    
    for i, strategy in enumerate(strategy_list, 1):
        click.echo(f"\n[{i}/{len(strategy_list)}] 回测策略: {strategy}")
        click.echo("-" * 70)
        
        config = BacktestConfig(initial_capital=capital, top_n=top_n)
        engine = VectorizedBacktestEngine(config=config)
        
        try:
            t0 = time.time()
            
            engine.load_data(start, end)
            engine.compute_factors(strategy)
            engine.generate_signals(method='topN', top_n=top_n)
            results = engine.run_backtest()
            
            elapsed = time.time() - t0
            
            results_summary.append({
                '策略': strategy,
                '年化收益': f"{results['annual_return']:.2%}",
                '夏普比率': f"{results['sharpe_ratio']:.2f}",
                '最大回撤': f"{results['max_drawdown']:.2%}",
                '胜率': f"{results['win_rate']:.1%}",
                '耗时': f"{elapsed:.1f}s"
            })
            
            click.echo(f"✓ 年化收益: {results['annual_return']:.2%}")
            click.echo(f"✓ 夏普比率: {results['sharpe_ratio']:.2f}")
            click.echo(f"✓ 最大回撤: {results['max_drawdown']:.2%}")
            click.echo(f"✓ 耗时: {elapsed:.1f}s")
            
        except Exception as e:
            click.echo(f"✗ 失败: {e}")
            results_summary.append({
                '策略': strategy,
                '年化收益': 'N/A',
                '夏普比率': 'N/A',
                '最大回撤': 'N/A',
                '胜率': 'N/A',
                '耗时': 'N/A'
            })
    
    # 汇总表
    click.echo("\n\n" + "=" * 70)
    click.echo("批量回测汇总")
    click.echo("=" * 70)
    
    df_summary = pd.DataFrame(results_summary)
    click.echo(df_summary.to_string(index=False))
    
    # 保存结果
    from pathlib import Path
    output_dir = Path('data/outputs')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    summary_path = output_dir / f"batch_summary_{start}_{end}.csv"
    df_summary.to_csv(summary_path, index=False)
    
    click.echo(f"\n💾 汇总结果已保存: {summary_path}")
    click.echo("=" * 70)


if __name__ == "__main__":
    cli()
