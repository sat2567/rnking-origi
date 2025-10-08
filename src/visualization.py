"""
Visualization utilities for mutual fund analysis.
"""
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Color palette
COLORS = px.colors.qualitative.Plotly
DEFAULT_COLOR = COLORS[0]
BENCHMARK_COLOR = '#808080'  # Gray for benchmark

def plot_cumulative_returns(
    nav_data: pd.DataFrame,
    benchmark_series: Optional[pd.Series] = None,
    title: str = "Cumulative Returns",
    height: int = 500
) -> go.Figure:
    """
    Plot cumulative returns for multiple funds and optionally a benchmark.
    
    Args:
        nav_data: DataFrame with NAV data (columns = funds, index = dates)
        benchmark_series: Optional Series with benchmark values (same index as nav_data)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Calculate cumulative returns
    cum_returns = (1 + nav_data.pct_change().fillna(0)).cumprod()
    
    fig = go.Figure()
    
    # Add fund traces
    for i, col in enumerate(cum_returns.columns):
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col],
            mode='lines',
            name=col,
            line=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate='%{y:.2f}<extra>%{x|%Y-%m-%d}</extra>'
        ))
    
    # Add benchmark trace if provided
    if benchmark_series is not None and not benchmark_series.empty:
        # Align benchmark with the funds' dates
        aligned_benchmark = benchmark_series.reindex(nav_data.index, method='ffill')
        bench_cum_returns = (1 + aligned_benchmark.pct_change().fillna(0)).cumprod()
        
        fig.add_trace(go.Scatter(
            x=bench_cum_returns.index,
            y=bench_cum_returns.values,
            mode='lines',
            name='Benchmark',
            line=dict(color=BENCHMARK_COLOR, dash='dash'),
            hovertemplate='%{y:.2f}<extra>Benchmark\n%{x|%Y-%m-%d}</extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of ₹1",
        hovermode="x unified",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".2f")
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def plot_drawdown(
    nav_data: pd.DataFrame,
    benchmark_series: Optional[pd.Series] = None,
    title: str = "Drawdown",
    height: int = 400
) -> go.Figure:
    """
    Plot drawdown for multiple funds and optionally a benchmark.
    
    Args:
        nav_data: DataFrame with NAV data (columns = funds, index = dates)
        benchmark_series: Optional Series with benchmark values (same index as nav_data)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Calculate drawdowns
    def get_drawdown(series):
        cum_returns = (1 + series.pct_change().fillna(0)).cumprod()
        rolling_max = cum_returns.cummax()
        return (cum_returns - rolling_max) / rolling_max
    
    drawdowns = nav_data.apply(get_drawdown)
    
    fig = go.Figure()
    
    # Add fund traces
    for i, col in enumerate(drawdowns.columns):
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns[col],
            mode='lines',
            name=col,
            line=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate='%{y:.1%}<extra>%{x|%Y-%m-%d}</extra>',
            fill='tozeroy'
        ))
    
    # Add benchmark drawdown if provided
    if benchmark_series is not None and not benchmark_series.empty:
        # Align benchmark with the funds' dates
        aligned_benchmark = benchmark_series.reindex(nav_data.index, method='ffill')
        bench_drawdown = get_drawdown(aligned_benchmark)
        
        fig.add_trace(go.Scatter(
            x=bench_drawdown.index,
            y=bench_drawdown.values,
            mode='lines',
            name='Benchmark',
            line=dict(color=BENCHMARK_COLOR, dash='dash'),
            hovertemplate='%{y:.1%}<extra>Benchmark\n%{x|%Y-%m-%d}</extra>',
            fill='tozeroy'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x",
        height=height,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".1%")
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    
    return fig

def plot_rolling_volatility(
    nav_data: pd.DataFrame,
    window: int = 63,  # 3 months
    benchmark_series: Optional[pd.Series] = None,
    title: str = "Rolling Volatility (Annualized)",
    height: int = 400
) -> go.Figure:
    """
    Plot rolling volatility for multiple funds and optionally a benchmark.
    
    Args:
        nav_data: DataFrame with NAV data (columns = funds, index = dates)
        window: Rolling window in days
        benchmark_series: Optional Series with benchmark values (same index as nav_data)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Calculate daily returns
    returns = nav_data.pct_change().dropna()
    
    # Calculate rolling volatility (annualized)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    fig = go.Figure()
    
    # Add fund traces
    for i, col in enumerate(rolling_vol.columns):
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[col],
            mode='lines',
            name=col,
            line=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate='%{y:.1%}<extra>%{x|%Y-%m-%d}</extra>'
        ))
    
    # Add benchmark volatility if provided
    if benchmark_series is not None and not benchmark_series.empty:
        # Align benchmark with the funds' dates
        aligned_benchmark = benchmark_series.reindex(nav_data.index, method='ffill')
        bench_returns = aligned_benchmark.pct_change().dropna()
        bench_rolling_vol = bench_returns.rolling(window=window).std() * np.sqrt(252)
        
        fig.add_trace(go.Scatter(
            x=bench_rolling_vol.index,
            y=bench_rolling_vol.values,
            mode='lines',
            name='Benchmark',
            line=dict(color=BENCHMARK_COLOR, dash='dash'),
            hovertemplate='%{y:.1%}<extra>Benchmark\n%{x|%Y-%m-%d}</extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        hovermode="x",
        height=height,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".1%")
    
    return fig

def plot_rolling_sharpe_ratio(
    nav_data: pd.DataFrame,
    window: int = 252,  # 1 year
    risk_free_rate: float = 0.06,
    title: str = "Rolling Sharpe Ratio",
    height: int = 400
) -> go.Figure:
    """
    Plot rolling Sharpe ratio for multiple funds.
    
    Args:
        nav_data: DataFrame with NAV data (columns = funds, index = dates)
        window: Rolling window in days
        risk_free_rate: Annual risk-free rate (default: 0.06)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Calculate daily returns
    returns = nav_data.pct_change().dropna()
    
    # Calculate rolling Sharpe ratio
    excess_returns = returns - (risk_free_rate / 252)
    rolling_sharpe = (excess_returns.rolling(window=window).mean() / 
                     returns.rolling(window=window).std() * 
                     np.sqrt(252))
    
    fig = go.Figure()
    
    # Add fund traces
    for i, col in enumerate(rolling_sharpe.columns):
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe[col],
            mode='lines',
            name=col,
            line=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate='%{y:.2f}<extra>%{x|%Y-%m-%d}</extra>'
        ))
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling Sharpe Ratio",
        hovermode="x",
        height=height,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def plot_rolling_beta(
    nav_data: pd.DataFrame,
    benchmark_series: pd.Series,
    window: int = 252,  # 1 year
    title: str = "Rolling Beta to Benchmark",
    height: int = 400
) -> go.Figure:
    """
    Plot rolling beta for multiple funds relative to a benchmark.
    
    Args:
        nav_data: DataFrame with NAV data (columns = funds, index = dates)
        benchmark_series: Series with benchmark values (same index as nav_data)
        window: Rolling window in days
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Calculate returns
    returns = nav_data.pct_change().dropna()
    bench_returns = benchmark_series.pct_change().dropna()
    
    # Calculate rolling beta
    rolling_beta = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for col in returns.columns:
        # Calculate rolling covariance and variance
        rolling_cov = returns[col].rolling(window=window).cov(bench_returns)
        rolling_var = bench_returns.rolling(window=window).var()
        rolling_beta[col] = rolling_cov / rolling_var
    
    fig = go.Figure()
    
    # Add fund traces
    for i, col in enumerate(rolling_beta.columns):
        fig.add_trace(go.Scatter(
            x=rolling_beta.index,
            y=rolling_beta[col],
            mode='lines',
            name=col,
            line=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate='%{y:.2f}<extra>%{x|%Y-%m-%d}</extra>'
        ))
    
    # Add horizontal line at 1
    fig.add_hline(y=1, line_dash="dash", line_color="black", opacity=0.3)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling Beta",
        hovermode="x",
        height=height,
        showlegend=True,
    )
    
    return fig


def plot_ranking_table(
    metrics_df: pd.DataFrame,
    title: str = "Top Mutual Funds by Composite Score",
    height: int = 600
) -> go.Figure:
    """
    Create a table showing the ranking of mutual funds based on various metrics.
    
    Args:
        metrics_df: DataFrame with metrics from calculate_composite_score
        title: Table title
        height: Table height in pixels
        
    Returns:
        Plotly Figure object with the ranking table
    """
    # Format numbers for display
    formatted_df = metrics_df.copy()
    
    # Format percentages (0-1 to percentage with 2 decimal places)
    pct_cols = ['annual_return', 'annual_volatility', 'max_drawdown', 'alpha']
    for col in pct_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    
    # Format other metrics
    if 'sharpe_ratio' in formatted_df.columns:
        formatted_df['sharpe_ratio'] = formatted_df['sharpe_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
    if 'sortino_ratio' in formatted_df.columns:
        formatted_df['sortino_ratio'] = formatted_df['sortino_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
    if 'information_ratio' in formatted_df.columns:
        formatted_df['information_ratio'] = formatted_df['information_ratio'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
    if 'tracking_error' in formatted_df.columns:
        formatted_df['tracking_error'] = formatted_df['tracking_error'].apply(
            lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
        )
    if 'beta' in formatted_df.columns:
        formatted_df['beta'] = formatted_df['beta'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
    if 'composite_score' in formatted_df.columns:
        formatted_df['composite_score'] = formatted_df['composite_score'].apply(
            lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
        )
    
    # Reset index to show fund names as a column
    formatted_df = formatted_df.reset_index().rename(columns={'index': 'Fund'})
    
    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>' + str(col) + '</b>' for col in formatted_df.columns],
            fill_color='#2c3e50',
            font=dict(color='white', size=12, family="Arial, sans-serif"),
            align=['left'] + ['center'] * (len(formatted_df.columns) - 1),
            height=40
        ),
        cells=dict(
            values=[formatted_df[col] for col in formatted_df.columns],
            fill_color='white',
            align=['left'] + ['center'] * (len(formatted_df.columns) - 1),
            font=dict(family="Arial, sans-serif", size=11),
            height=35
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top',
            font=dict(size=16, family="Arial, sans-serif")
        ),
        height=min(800, 100 + 35 * len(formatted_df)),
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_returns_distribution(
    nav_data: pd.DataFrame,
    benchmark_series: Optional[pd.Series] = None,
    title: str = "Distribution of Daily Returns",
    height: int = 400
) -> go.Figure:
    """
    Plot distribution of daily returns for multiple funds and optionally a benchmark.
    
    Args:
        nav_data: DataFrame with NAV data (columns = funds, index = dates)
        benchmark_series: Optional Series with benchmark values (same index as nav_data)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Calculate daily returns
    returns = nav_data.pct_change().dropna()
    
    fig = go.Figure()
    
    # Add fund histograms
    for i, col in enumerate(returns.columns):
        fig.add_trace(go.Histogram(
            x=returns[col],
            name=col,
            opacity=0.7,
            marker_color=COLORS[i % len(COLORS)],
            nbinsx=100,
            hovertemplate='Range: %{x:.2%}<br>Count: %{y}'  
        ))
    
    # Add benchmark histogram if provided
    if benchmark_series is not None and not benchmark_series.empty:
        # Align benchmark with the funds' dates
        aligned_benchmark = benchmark_series.reindex(nav_data.index, method='ffill')
        bench_returns = aligned_benchmark.pct_change().dropna()
        
        fig.add_trace(go.Histogram(
            x=bench_returns,
            name='Benchmark',
            opacity=0.3,
            marker_color=BENCHMARK_COLOR,
            nbinsx=100,
            hovertemplate='Range: %{x:.2%}<br>Count: %{y}<extra>Benchmark</extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        barmode='overlay',
        height=height,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Format x-axis as percentage
    fig.update_xaxes(tickformat=".1%")
    
    return fig
