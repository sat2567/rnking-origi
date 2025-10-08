from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from .ranking import calculate_composite_score


def compute_cagr(series: pd.Series, periods_per_year: int = 252) -> float:
    if series.empty:
        return np.nan
    total_return = series.iloc[-1] / series.iloc[0] - 1 if series.iloc[0] != 0 else np.nan
    n_periods = len(series)
    years = n_periods / periods_per_year
    if years <= 0 or np.isnan(total_return):
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


def compute_max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    roll_max = series.cummax()
    drawdown = series / roll_max - 1.0
    return drawdown.min()


def compute_annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return np.nan
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.06, periods_per_year: int = 252) -> float:
    if returns.empty:
        return np.nan
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / vol


def run_backtest(
    nav_data: pd.DataFrame,
    benchmark_series: pd.Series,
    metrics_weights: Dict[str, float],
    risk_free_rate: float = 0.06,
    rebalance_freq: str = 'M',  # 'M' monthly, 'Q' quarterly, '6M' semi-annual
    lookback_days: int = 252,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a simple top-N composite score backtest.

    - At each rebalance date, compute composite scores using the previous `lookback_days` of data.
    - Select the top_n funds and hold equal-weighted until next rebalance.

    Returns:
      equity_df: columns ['Strategy', 'Benchmark'] cumulative index values (start=1.0)
      metrics_df: summary performance metrics for Strategy and Benchmark
    """
    if nav_data is None or nav_data.empty:
        raise ValueError("nav_data is empty")
    if benchmark_series is None or benchmark_series.empty:
        raise ValueError("benchmark_series is empty")

    # Ensure datetime index
    nav = nav_data.sort_index().copy()
    bench = benchmark_series.sort_index().copy()

    # Align dates
    common_index = nav.index.intersection(bench.index)
    nav = nav.loc[common_index]
    bench = bench.loc[common_index]

    # Compute daily returns
    nav_rets = nav.pct_change().dropna(how='all')
    bench_rets = bench.pct_change().dropna()

    if nav_rets.empty or bench_rets.empty:
        raise ValueError("Not enough data to compute returns for backtest")

    # Rebalance dates
    if rebalance_freq not in ['M', 'Q', '6M']:
        raise ValueError("rebalance_freq must be 'M', 'Q', or '6M'")
    if rebalance_freq == 'M':
        rebal_dates = nav_rets.resample('M').last().index
    elif rebalance_freq == 'Q':
        rebal_dates = nav_rets.resample('Q').last().index
    else:  # '6M' semi-annual
        # Use two-quarter periods to approximate 6 months at quarter-ends
        rebal_dates = nav_rets.resample('2Q').last().index

    # Iterate rebalances
    strategy_daily = pd.Series(index=nav_rets.index, dtype=float)
    current_weights: Optional[pd.Series] = None
    prev_rebal_date = None

    holdings_records = []
    for i, date in enumerate(rebal_dates):
        # Determine lookback window end
        end_date = date
        start_date = end_date - pd.Timedelta(days=lookback_days)
        window_idx = nav.index[(nav.index > start_date) & (nav.index <= end_date)]
        if len(window_idx) < 20:
            # Skip if insufficient history
            continue
        nav_window = nav.loc[window_idx]
        bench_window = bench.loc[window_idx]

        # Compute composite scores on the window
        scores_df = calculate_composite_score(
            nav_data=nav_window,
            benchmark_series=bench_window,
            risk_free_rate=risk_free_rate,
            metrics_weights=metrics_weights,
        )
        if scores_df.empty or 'composite_score' not in scores_df.columns:
            continue

        top_funds = scores_df['composite_score'].sort_values(ascending=False).head(top_n).index
        if len(top_funds) == 0:
            continue

        # Equal weights among selected funds
        current_weights = pd.Series(1.0 / len(top_funds), index=top_funds)

        # Record holdings for this rebalance date
        holdings_records.append({
            'date': pd.to_datetime(date),
            'funds': ', '.join(list(top_funds))
        })

        # Determine holding period (until next rebalance)
        start_hold = date
        end_hold = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else nav_rets.index[-1]
        hold_idx = nav_rets.index[(nav_rets.index > start_hold) & (nav_rets.index <= end_hold)]
        if len(hold_idx) == 0:
            continue

        # Compute portfolio daily returns during holding period
        rets_slice = nav_rets.loc[hold_idx, current_weights.index].copy()
        # Drop columns that may be missing all data in the slice
        rets_slice = rets_slice.dropna(axis=1, how='all')
        # Align weights with available columns
        w = current_weights.reindex(rets_slice.columns).dropna()
        if w.empty:
            continue
        w = w / w.sum()
        strat_rets = (rets_slice.fillna(0) @ w)
        strategy_daily.loc[hold_idx] = strat_rets.values

    strategy_daily = strategy_daily.dropna()

    # Build cumulative equity curves (start at 1.0)
    strat_equity = (1 + strategy_daily).cumprod()
    bench_equity = (1 + bench_rets.loc[strategy_daily.index]).cumprod()

    equity_df = pd.DataFrame({
        'Strategy': strat_equity,
        'Benchmark': bench_equity
    })

    # Holdings log
    holdings_df = pd.DataFrame(holdings_records)
    if not holdings_df.empty:
        holdings_df = holdings_df.sort_values('date').reset_index(drop=True)

    # Metrics
    periods_per_year = 252
    metrics = {
        'CAGR': [
            compute_cagr(equity_df['Strategy'], periods_per_year),
            compute_cagr(equity_df['Benchmark'], periods_per_year),
        ],
        'Volatility': [
            compute_annual_volatility(strategy_daily, periods_per_year),
            compute_annual_volatility(bench_rets.loc[strategy_daily.index], periods_per_year),
        ],
        'Sharpe': [
            compute_sharpe(strategy_daily, risk_free_rate, periods_per_year),
            compute_sharpe(bench_rets.loc[strategy_daily.index], risk_free_rate, periods_per_year),
        ],
        'Max Drawdown': [
            compute_max_drawdown(equity_df['Strategy']),
            compute_max_drawdown(equity_df['Benchmark']),
        ],
        'Total Return': [
            equity_df['Strategy'].iloc[-1] - 1 if not equity_df['Strategy'].empty else np.nan,
            equity_df['Benchmark'].iloc[-1] - 1 if not equity_df['Benchmark'].empty else np.nan,
        ]
    }
    metrics_df = pd.DataFrame(metrics, index=['Strategy', 'Benchmark'])

    return equity_df, metrics_df, holdings_df
