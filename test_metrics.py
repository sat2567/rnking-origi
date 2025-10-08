"""
Test script for metrics.py edge cases
"""
import numpy as np
import pandas as pd
from src import metrics

# Test cases
def test_edge_cases():
    print("Testing edge cases...")
    
    # Empty returns
    empty_returns = pd.Series([])
    
    # Single return
    single_return = pd.Series([0.01])
    
    # Zero volatility
    zero_vol_returns = pd.Series([0.01, 0.01, 0.01])
    
    # Negative NAV values
    negative_nav = pd.Series([100, 95, 90, -85])
    
    # Test all metrics
    test_cases = [
        ("Annualized Return", metrics.calculate_annualized_return, empty_returns),
        ("Annualized Return", metrics.calculate_annualized_return, single_return),
        ("Volatility", metrics.calculate_volatility, zero_vol_returns),
        ("Sharpe Ratio", metrics.calculate_sharpe_ratio, zero_vol_returns),
        ("Sortino Ratio", metrics.calculate_sortino_ratio, zero_vol_returns),
        ("Max Drawdown", metrics.calculate_max_drawdown, negative_nav),
    ]
    
    for name, func, data in test_cases:
        result = func(data)
        print(f"{name}: {'NaN' if np.isnan(result) else result}")
    
    print("All edge case tests completed!")

if __name__ == "__main__":
    test_edge_cases()
