"""
Ranking utilities for mutual fund analysis.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from . import metrics

def calculate_composite_score(
    nav_data: pd.DataFrame,
    benchmark_series: pd.Series,
    risk_free_rate: float = 0.06,
    metrics_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Calculate a composite score for each fund based on multiple metrics.
    
    Args:
        nav_data: DataFrame with NAV data (columns = funds, index = dates)
        benchmark_series: Series with benchmark values (same index as nav_data)
        risk_free_rate: Annual risk-free rate (default: 0.06)
        metrics_weights: Dictionary of metric names and their weights (sum to 1.0)
        
    Returns:
        DataFrame with metrics and composite score for each fund
    """
    if metrics_weights is None:
        metrics_weights = {
            'sharpe_ratio': 0.4,
            'sortino_ratio': 0.4,
            'information_ratio': 0.1,
            'max_drawdown': -0.1  # Negative weight because lower is better
        }
    
    # Calculate all metrics
    results = {}
    
    for fund in nav_data.columns:
        fund_returns = nav_data[fund].pct_change().dropna()
        benchmark_returns = benchmark_series.pct_change().dropna()
        
        # Align dates
        common_idx = fund_returns.index.intersection(benchmark_returns.index)
        fund_returns = fund_returns[common_idx]
        aligned_benchmark = benchmark_returns[common_idx]
        
        # Calculate metrics
        fund_metrics = {
            'annual_return': metrics.calculate_annualized_return(fund_returns),
            'annual_volatility': metrics.calculate_volatility(fund_returns),
            'sharpe_ratio': metrics.calculate_sharpe_ratio(fund_returns, risk_free_rate=risk_free_rate),
            'sortino_ratio': metrics.calculate_sortino_ratio(fund_returns, risk_free_rate=risk_free_rate),
            'max_drawdown': metrics.calculate_max_drawdown(nav_data[fund]),
            'information_ratio': metrics.calculate_information_ratio(fund_returns, aligned_benchmark),
            'tracking_error': metrics.calculate_tracking_error(fund_returns, aligned_benchmark)
        }
        results[fund] = fund_metrics
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Normalize metrics (0-1 scaling)
    normalized_df = metrics_df.copy()
    for col in metrics_weights.keys():
        if col in normalized_df.columns:
            if metrics_weights[col] >= 0:  # Higher is better
                normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / \
                                   (normalized_df[col].max() - normalized_df[col].min() + 1e-10)
            else:  # Lower is better (like max_drawdown)
                normalized_df[col] = 1 - ((normalized_df[col] - normalized_df[col].min()) / 
                                       (normalized_df[col].max() - normalized_df[col].min() + 1e-10))
    
    # Calculate composite score
    composite_scores = pd.Series(0.0, index=normalized_df.index)
    for metric, weight in metrics_weights.items():
        if metric in normalized_df.columns:
            composite_scores += normalized_df[metric] * abs(weight)
    
    # Add composite score to metrics
    metrics_df['composite_score'] = composite_scores
    
    # Sort by composite score
    metrics_df = metrics_df.sort_values('composite_score', ascending=False)
    
    return metrics_df

def get_top_funds(
    metrics_df: pd.DataFrame,
    n: int = 10,
    sort_by: str = 'composite_score'
) -> pd.DataFrame:
    """
    Get top N funds based on a specific metric.
    
    Args:
        metrics_df: DataFrame with metrics from calculate_composite_score
        n: Number of top funds to return
        sort_by: Metric to sort by (default: 'composite_score')
        
    Returns:
        DataFrame with top N funds and their metrics
    """
    if sort_by not in metrics_df.columns:
        sort_by = 'composite_score'
    
    return metrics_df.nlargest(n, sort_by)
