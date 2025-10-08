    if len(returns) < 2:
        return np.nan
        
    vol = returns.std()
    if vol <= 0:  # Guard against zero/negative volatility
        return np.nan
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return excess_returns.mean() / vol * np.sqrt(periods_per_year)
