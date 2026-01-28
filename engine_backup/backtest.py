# ============================================================================
# 文件: engine/backtest.py
# ============================================================================
"""
回测引擎 - 核心调度器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
from datetime import datetime
import logging
import sys

from .matcher import MatchEngine, Order, OrderStatus
from .portfolio import PortfolioManager, Position
from strategy import BaseStrategy, StrategyContext, Signal, OrderSide
from core.database import StockDatabase
from config import settings


class ProgressBar:
    """简易进度条"""

    def __init__(self, total: int, prefix: str = '', width: int = 40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0

    def update(self, current: int = None, suffix: str = ''):
        """更新进度"""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        if self.total <= 0:
            return

        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)

        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent:>6.1%} ({self.current}/{self.total}) {suffix}')
        sys.stdout.flush()

        if self.current >= self.total:
            print()  # 换行

    def finish(self):
        """完成"""
        self.update(self.total)


class BacktestEngine:
    """
    回测引擎 - 策略评测核心

    架构:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         BacktestEngine                              │
    │                                                                     │
    │  ┌───────────────┐     ┌───────────────┐     ┌─────────────────┐   │
    │  │  DataLoader   │────►│   Strategy    │────►│  MatchEngine    │   │
    │  │  时间步数据    │     │   信号生成    │     │  订单撮合       │   │
    │  └───────────────┘     └───────────────┘     └────────┬────────┘   │
    │         │                                              │            │
    │         │              ┌───────────────┐               │            │
    │         └─────────────►│  Portfolio    │◄──────────────┘            │
    │                        │  持仓/权益    │                            │
    │                        └───────┬───────┘                            │
    │                                │                                    │
    │                        ┌───────▼───────┐                            │
    │                        │  Analyzer     │                            │
    │                        │  绩效分析     │                            │
    │                        └───────────────┘                            │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
            self,
            db_path: str = None,
            initial_capital: float = None,
            commission_rate: float = None,
            slippage_rate: float = None,
            show_progress: bool = True  # 新增: 是否显示进度条
    ):
        self.db_path = db_path or str(settings.path.DB_PATH)
        self.initial_capital = initial_capital or settings.backtest.INITIAL_CAPITAL
        self.show_progress = show_progress

        # 组件
        self.db = StockDatabase(self.db_path)
        self.match_engine = MatchEngine(
            commission_rate=commission_rate,
            slippage_rate=slippage_rate
        )

        # 策略容器 (支持多策略对比)
        self.strategies: Dict[str, Tuple[BaseStrategy, PortfolioManager]] = {}

        # 数据
        self.trading_dates: List[str] = []
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._market_data: pd.DataFrame = None

        self.logger = logging.getLogger("BacktestEngine")

    def add_strategy(self, strategy: BaseStrategy) -> 'BacktestEngine':
        """添加策略"""
        portfolio = PortfolioManager(self.initial_capital)
        self.strategies[strategy.name] = (strategy, portfolio)
        self.logger.info(f"Added strategy: {strategy.name}")
        return self

    def run(
            self,
            start_date: str,
            end_date: str,
            codes: List[str] = None,
            rebalance_freq: str = None
    ) -> Dict[str, 'BacktestResult']:
        """
        运行回测

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期
            codes: 股票池 (None=全市场)
            rebalance_freq: 调仓频率 (D/W/M)

        Returns:
            {strategy_name: BacktestResult}
        """
        rebalance_freq = rebalance_freq or settings.backtest.REBALANCE_FREQ

        # ===== 数据库空检查 =====
        stats = self.db.get_stats()
        if stats.get('total_rows', 0) == 0:
            self.logger.error("数据库为空！请先运行 'python main.py init' 初始化数据")
            raise RuntimeError(
                "数据库为空！请先运行以下命令初始化:\n"
                "  python main.py init\n"
                "或在菜单中选择 [1. 📦 初始化数据库]"
            )

        self.logger.info(f"Starting backtest: {start_date} to {end_date}")
        self.logger.info(f"Strategies: {list(self.strategies.keys())}")

        # 1. 加载数据 (带进度)
        self._load_data(start_date, end_date, codes)

        # 检查是否加载到数据
        if len(self.trading_dates) == 0:
            self.logger.error(f"指定日期范围 {start_date} ~ {end_date} 内没有数据")
            raise RuntimeError(f"指定日期范围内没有数据，请检查日期或先初始化数据库")

        # 2. 初始化策略 & 预计算因子 (带进度)
        self._initialize_strategies()

        # 3. 获取调仓日期
        rebalance_dates = self._get_rebalance_dates(rebalance_freq)

        # 4. 逐日回测 (带进度条)
        total_days = len(self.trading_dates)
        self.logger.info(f"Running {total_days} trading days...")

        if self.show_progress:
            progress = ProgressBar(total_days, prefix='📈 回测进度')

        for i, current_date in enumerate(self.trading_dates):
            current_data = self._get_daily_data(current_date)
            is_rebalance = current_date in rebalance_dates

            for name, (strategy, portfolio) in self.strategies.items():
                # 更新市值
                portfolio.update_market_value(current_data)

                # 调仓日生成信号
                if is_rebalance:
                    context = self._build_context(current_date, current_data, portfolio, strategy)
                    signals = strategy.generate_signals(context)

                    if signals:
                        self._execute_signals(signals, current_data, current_date, portfolio, strategy)

                # 记录权益
                portfolio.record_snapshot(current_date)

                # 日终回调
                strategy.on_day_end(context if is_rebalance else None)

            # 更新进度条
            if self.show_progress:
                # 计算当前收益用于显示
                equity_info = ""
                for name, (_, portfolio) in self.strategies.items():
                    pnl = (portfolio.total_equity / self.initial_capital - 1) * 100
                    equity_info = f"| 收益: {pnl:+.2f}%"
                    break
                progress.update(i + 1, equity_info)

        print()  # 进度条结束换行

        # 5. 生成结果
        results = {}
        for name, (strategy, portfolio) in self.strategies.items():
            results[name] = BacktestResult(
                strategy_name=name,
                portfolio=portfolio,
                initial_capital=self.initial_capital
            )
            results[name].print_summary()

        return results

    def _load_data(self, start_date: str, end_date: str, codes: List[str]) -> None:
        """加载数据"""
        if self.show_progress:
            print("📊 加载市场数据...")
        self.logger.info("Loading market data...")

        # 扩展开始日期 (需要历史数据计算因子)
        extended_start = pd.to_datetime(start_date) - pd.DateOffset(years=3)
        extended_start_str = extended_start.strftime('%Y-%m-%d')

        # 从数据库加载
        with self.db.connect() as conn:
            # 先获取总数用于进度显示
            if codes:
                placeholders = ','.join(['?' for _ in codes])
                count_query = f"SELECT COUNT(DISTINCT code) FROM daily_bars WHERE code IN ({placeholders})"
                total_stocks = conn.execute(count_query, codes).fetchone()[0]

                query = f"""
                    SELECT code, market, date, open, high, low, close, vol, amount
                    FROM daily_bars
                    WHERE code IN ({placeholders})
                      AND date BETWEEN ? AND ?
                    ORDER BY code, date
                """
                self._market_data = conn.execute(query, codes + [extended_start_str, end_date]).fetchdf()
            else:
                count_query = "SELECT COUNT(DISTINCT code) FROM daily_bars"
                total_stocks = conn.execute(count_query).fetchone()[0]

                query = """
                    SELECT code, market, date, open, high, low, close, vol, amount
                    FROM daily_bars
                    WHERE date BETWEEN ? AND ?
                    ORDER BY code, date
                """
                self._market_data = conn.execute(query, [extended_start_str, end_date]).fetchdf()

        if self._market_data.empty:
            self.logger.warning("No data loaded from database")
            if self.show_progress:
                print("   ⚠️ 未加载到任何数据")
            return

        if self.show_progress:
            print(f"   ✓ 从数据库加载 {len(self._market_data):,} 条记录")

        # 计算涨跌停
        if self.show_progress:
            print("   ⏳ 计算涨跌停标记...")
        self._market_data = self._add_limit_flags(self._market_data)

        # 交易日列表 (只取回测区间)
        all_dates = self._market_data['date'].unique()
        self.trading_dates = sorted([
            str(d)[:10] if hasattr(d, 'strftime') else str(d)[:10]
            for d in all_dates
            if start_date <= str(d)[:10] <= end_date
        ])

        # 按股票缓存历史数据 (带进度)
        unique_codes = self._market_data['code'].unique()
        total_codes = len(unique_codes)

        if self.show_progress:
            progress = ProgressBar(total_codes, prefix='   📦 缓存股票数据')

        for i, code in enumerate(unique_codes):
            stock_df = self._market_data[self._market_data['code'] == code].copy()
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            stock_df.set_index('date', inplace=True)
            self._data_cache[code] = stock_df

            if self.show_progress and (i + 1) % 100 == 0:
                progress.update(i + 1)

        if self.show_progress:
            progress.finish()

        self.logger.info(f"Loaded {len(self._data_cache)} stocks, {len(self.trading_dates)} trading days")
        if self.show_progress:
            print(f"   ✓ 已加载 {len(self._data_cache)} 只股票, {len(self.trading_dates)} 个交易日")

    def _add_limit_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加涨跌停标记"""
        df = df.copy()
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        df['limit_up'] = (df['prev_close'] * 1.1).round(2)
        df['limit_down'] = (df['prev_close'] * 0.9).round(2)
        df['is_limit_up'] = df['close'] >= df['limit_up'] - 0.01
        df['is_limit_down'] = df['close'] <= df['limit_down'] + 0.01
        return df

    def _initialize_strategies(self) -> None:
        """初始化策略并预计算因子"""
        if self.show_progress:
            print("🔧 初始化策略...")
        self.logger.info("Initializing strategies and computing factors...")

        for name, (strategy, _) in self.strategies.items():
            if self.show_progress:
                print(f"   ⏳ 初始化策略: {name}")
            strategy.initialize()

            # 计算因子
            if self._data_cache:
                if self.show_progress:
                    print(f"   ⏳ 计算因子 ({len(self._data_cache)} 只股票)...")

                factors = strategy.compute_factors(self._data_cache)
                strategy._factors = factors

                self.logger.info(f"  {name}: computed {len(factors)} factors")
                if self.show_progress:
                    print(f"   ✓ {name}: 计算了 {len(factors)} 个因子")
            else:
                strategy._factors = {}
                self.logger.warning(f"  {name}: no data to compute factors")

    def _get_daily_data(self, date: str) -> pd.DataFrame:
        """获取当日数据"""
        if self._market_data is None or self._market_data.empty:
            return pd.DataFrame()

        date_col = self._market_data['date']
        if hasattr(date_col.iloc[0], 'strftime'):
            mask = date_col.dt.strftime('%Y-%m-%d') == date
        else:
            mask = date_col.astype(str).str[:10] == date

        return self._market_data[mask].copy()

    def _get_rebalance_dates(self, freq: str) -> set:
        """获取调仓日期"""
        if not self.trading_dates:
            return set()

        dates = pd.to_datetime(self.trading_dates)

        if freq == 'D':
            return set(self.trading_dates)

        df = pd.DataFrame({'date': dates})

        if freq == 'W':
            df['period'] = df['date'].dt.isocalendar().week.astype(str) + '-' + df['date'].dt.year.astype(str)
        elif freq == 'M':
            df['period'] = df['date'].dt.to_period('M').astype(str)
        else:
            return set(self.trading_dates)

        last_dates = df.groupby('period')['date'].last()
        return set(last_dates.dt.strftime('%Y-%m-%d').tolist())

    def _build_context(
            self,
            current_date: str,
            current_data: pd.DataFrame,
            portfolio: PortfolioManager,
            strategy: BaseStrategy
    ) -> StrategyContext:
        """构建策略上下文"""
        # 获取历史数据
        history = {}
        for code in current_data['code'].unique():
            if code in self._data_cache:
                hist = self._data_cache[code]
                history[code] = hist[hist.index <= current_date].tail(250)

        # 持仓转换
        positions = {
            code: pos.quantity
            for code, pos in portfolio.positions.items()
        }

        return StrategyContext(
            current_date=current_date,
            current_data=current_data,
            history_data=history,
            factors=getattr(strategy, '_factors', {}),
            positions=positions,
            cash=portfolio.cash,
            total_equity=portfolio.total_equity
        )

    def _execute_signals(
            self,
            signals: List[Signal],
            market_data: pd.DataFrame,
            current_date: str,
            portfolio: PortfolioManager,
            strategy: BaseStrategy
    ) -> None:
        """执行信号"""
        # 转换为目标权重
        target_weights = self._signals_to_weights(signals)

        # 计算订单
        orders = portfolio.calculate_rebalance_orders(
            target_weights, market_data, current_date, self.match_engine
        )

        # 撮合执行
        for order in orders:
            code_data = market_data[market_data['code'] == order.code]

            if code_data.empty:
                continue

            position = portfolio.get_position(order.code)

            matched = self.match_engine.match(
                order, code_data.iloc[0], position, current_date
            )

            if matched.status == OrderStatus.FILLED:
                portfolio.apply_order(matched, current_date)
                strategy.on_order_filled(matched)
            else:
                strategy.on_order_rejected(matched, matched.reject_reason)

    def _signals_to_weights(self, signals: List[Signal]) -> Dict[str, float]:
        """信号转权重"""
        weights = {}

        # 分离买卖信号
        buy_signals = [s for s in signals if s.side == OrderSide.BUY]
        sell_signals = [s for s in signals if s.side == OrderSide.SELL]

        # 卖出信号: 权重=0
        for s in sell_signals:
            weights[s.code] = 0.0

        # 买入信号: 归一化权重
        if buy_signals:
            total = sum(s.weight for s in buy_signals)
            reserve = settings.backtest.CASH_RESERVE

            for s in buy_signals:
                weights[s.code] = (s.weight / total) * (1 - reserve) if total > 0 else 0

        return weights

    def compare_strategies(self) -> pd.DataFrame:
        """对比策略净值"""
        curves = {}

        for name, (_, portfolio) in self.strategies.items():
            df = portfolio.get_equity_df()
            if not df.empty:
                curves[name] = df['equity'] / self.initial_capital

        return pd.DataFrame(curves)


class BacktestResult:
    """回测结果"""

    def __init__(
            self,
            strategy_name: str,
            portfolio: PortfolioManager,
            initial_capital: float
    ):
        self.strategy_name = strategy_name
        self.portfolio = portfolio
        self.initial_capital = initial_capital

        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """计算绩效指标"""
        df = self.portfolio.get_equity_df()

        if df.empty:
            self.metrics = {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'sharpe': 0,
                'sortino': 0,
                'calmar': 0,
                'win_rate': 0,
                'total_trades': 0
            }
            return

        equity = df['equity'].values
        returns = df['daily_return'].values

        # 收益
        total_return = (equity[-1] / self.initial_capital) - 1
        n_years = len(equity) / 252
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        # 风险
        max_drawdown = df['drawdown'].max() if 'drawdown' in df.columns else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

        # 比率
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10) if returns.std() > 0 else 0
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        neg_returns = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / (neg_returns.std() + 1e-10) if len(neg_returns) > 0 else sharpe

        # 交易统计
        trades = self.portfolio.get_trades_df()
        win_rate = 0
        if not trades.empty:
            daily_win = (returns > 0).sum() / max(len(returns), 1)
            win_rate = daily_win

        self.metrics = {
            'total_return': round(total_return, 4),
            'annual_return': round(annual_return, 4),
            'max_drawdown': round(max_drawdown, 4),
            'volatility': round(volatility, 4),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'calmar': round(calmar, 3),
            'win_rate': round(win_rate, 4),
            'total_trades': len(trades) if not trades.empty else 0
        }

    def print_summary(self) -> None:
        """打印摘要"""
        m = self.metrics
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    📊 回测结果: {self.strategy_name:<30}║
╠══════════════════════════════════════════════════════════════════╣
║  总收益:      {m.get('total_return', 0):>10.2%}    年化收益:    {m.get('annual_return', 0):>10.2%}   ║
║  最大回撤:    {m.get('max_drawdown', 0):>10.2%}    波动率:      {m.get('volatility', 0):>10.2%}   ║
║  夏普比率:    {m.get('sharpe', 0):>10.3f}    卡玛比率:    {m.get('calmar', 0):>10.3f}   ║
║  索提诺:      {m.get('sortino', 0):>10.3f}    日胜率:      {m.get('win_rate', 0):>10.2%}   ║
║  交易次数:    {m.get('total_trades', 0):>10d}                                    ║
╚══════════════════════════════════════════════════════════════════╝
""")

    def get_equity_curve(self) -> pd.DataFrame:
        return self.portfolio.get_equity_df()

    def get_trades(self) -> pd.DataFrame:
        return self.portfolio.get_trades_df()