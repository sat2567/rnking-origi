"""
Financial metrics calculations with robust edge case handling.
"""
from typing import Tuple
import numpy as np
import pandas as pd

def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from daily returns.
    
    Args:
        returns: Series of daily returns
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Annualized return or np.nan for invalid inputs
    """
    if len(returns) < 2:
        return np.nan
        
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1

def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from daily returns.
    
    Args:
        returns: Series of daily returns
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Annualized volatility or np.nan for invalid inputs
    """
    if len(returns) < 2:
        return np.nan
        
    vol = returns.std()
    if vol <= 0:
        return np.nan
        
    return vol * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio from daily returns.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 0.06)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Sharpe ratio or np.nan for invalid inputs
    """
    if len(returns) < 2:
        return np.nan
        
    vol = returns.std()
    if vol <= 0:
        return np.nan
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return excess_returns.mean() / vol * np.sqrt(periods_per_year)

def calculate_sortino_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio from daily returns.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 0.06)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Sortino ratio or np.nan for invalid inputs
    """
    if len(returns) < 2:
        return np.nan
        
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.nan
        
    downside_vol = downside_returns.std()
    if downside_vol <= 0:
        return np.nan
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return excess_returns.mean() / downside_vol * np.sqrt(periods_per_year)

def calculate_max_drawdown(nav_series: pd.Series) -> float:
    """
    Calculate maximum drawdown from NAV series.
    
    Args:
        nav_series: Series of NAV values
        
    Returns:
        Maximum drawdown (as negative percentage) or np.nan for invalid inputs
    """
    if len(nav_series) < 2:
        return np.nan
        
    if nav_series.min() <= 0:
        return np.nan
        
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    return drawdown.min()

def calculate_information_ratio(
    fund_returns: pd.Series, 
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio from fund and benchmark returns.
    
    Args:
        fund_returns: Series of fund returns
        benchmark_returns: Series of benchmark returns (same length as fund_returns)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Information ratio or np.nan for invalid inputs
    """
    if len(fund_returns) < 2 or len(fund_returns) != len(benchmark_returns):
        return np.nan
        
    active_returns = fund_returns - benchmark_returns
    active_vol = active_returns.std()
    if active_vol <= 0:
        return np.nan
        
    return active_returns.mean() / active_vol * np.sqrt(periods_per_year)

def calculate_tracking_error(
    fund_returns: pd.Series, 
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate tracking error from fund and benchmark returns.
    
    Args:
        fund_returns: Series of fund returns
        benchmark_returns: Series of benchmark returns (same length as fund_returns)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Annualized tracking error or np.nan for invalid inputs
    """
    if len(fund_returns) < 2 or len(fund_returns) != len(benchmark_returns):
        return np.nan
        
    active_returns = fund_returns - benchmark_returns
    active_vol = active_returns.std()
    if active_vol <= 0:
        return np.nan
        
    return active_vol * np.sqrt(periods_per_year)

def calculate_alpha_beta(
    fund_returns: pd.Series, 
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252
) -> Tuple[float, float]:
    """
    Calculate alpha and beta from fund and benchmark returns.
    
    Args:
        fund_returns: Series of fund returns
        benchmark_returns: Series of benchmark returns (same length as fund_returns)
        risk_free_rate: Annual risk-free rate (default: 0.06)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Tuple of (alpha, beta) or (np.nan, np.nan) for invalid inputs
    """
    if len(fund_returns) < 2 or len(fund_returns) != len(benchmark_returns):
        return np.nan, np.nan
        
    # Calculate excess returns
    fund_excess = fund_returns - (risk_free_rate / periods_per_year)
    bench_excess = benchmark_returns - (risk_free_rate / periods_per_year)
    
    # Calculate covariance and variance
    cov_matrix = np.cov(fund_excess, bench_excess)
    if cov_matrix.size < 4:
        return np.nan, np.nan
        
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # Annualize alpha
    alpha = fund_excess.mean() - beta * bench_excess.mean()
    alpha *= periods_per_year
    
    return alpha, beta
