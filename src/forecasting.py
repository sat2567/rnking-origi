'''from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def to_monthly_returns(nav_data: pd.DataFrame) -> pd.DataFrame:
    """Convert daily NAV price data to monthly simple returns per fund.
    Expects nav_data indexed by datetime with fund columns of prices (NAVs).
    """
    if nav_data.empty:
        return pd.DataFrame(index=nav_data.index)
    # Last price each month
    monthly_prices = nav_data.resample('M').last()
    monthly_rets = monthly_prices.pct_change().dropna(how='all')
    return monthly_rets


def ar1_fit(series: pd.Series) -> Tuple[float, float]:
    """Fit AR(1) via OLS: r_t = c + phi * r_{t-1} + eps
    Returns (c, phi). If insufficient data, fall back to (mean, 0).
    """
    s = series.dropna().astype(float)
    if len(s) < 3:
        mu = float(s.mean()) if len(s) > 0 else 0.0
        return mu, 0.0
    x = s.shift(1).dropna()
    y = s.loc[x.index]
    var_x = float(np.var(x, ddof=0))
    if var_x == 0 or np.isnan(var_x):
        mu = float(s.mean())
        return mu, 0.0
    cov_xy = float(np.cov(x, y, ddof=0)[0, 1])
    phi = cov_xy / var_x
    # intercept c = mean(y) - phi * mean(x)
    c = float(y.mean() - phi * x.mean())
    return c, phi


def ar1_forecast_path(last_value: float, c: float, phi: float, horizon: int) -> np.ndarray:
    """Generate h-step ahead AR(1) forecasts recursively."""
    f = np.zeros(horizon, dtype=float)
    prev = last_value
    for h in range(horizon):
        next_val = c + phi * prev
        f[h] = next_val
        prev = next_val
    return f


def mean_forecast_path(mu: float, horizon: int) -> np.ndarray:
    return np.full(horizon, float(mu))


def _gb_requirements():
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for Gradient Boosting forecasts. Please install 'scikit-learn'."
        ) from e


def _rolling_features_from_list(values: list[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    feats = {}
    # lags
    feats['lag1'] = arr[-1] if len(arr) >= 1 else 0.0
    # rolling means
    for w in (3, 6, 12):
        window = arr[-w:] if len(arr) >= w else arr
        feats[f'mean_{w}'] = float(np.nanmean(window)) if window.size else 0.0
        feats[f'std_{w}'] = float(np.nanstd(window)) if window.size else 0.0
    # drawdown based on cumulative product
    if arr.size:
        equity = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        feats['curr_dd'] = float(dd[-1])
        feats['min_dd_12'] = float(np.min(dd[-12:])) if dd.size >= 1 else 0.0
    else:
        feats['curr_dd'] = 0.0
        feats['min_dd_12'] = 0.0
    return feats


def _build_supervised_frame(ser: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Create X (features at t-1) and y (return at t) for training."""
    s = ser.dropna().astype(float)
    if len(s) < 18:
        return pd.DataFrame(), pd.Series(dtype=float)
    vals = s.values.tolist()
    rows = []
    y = []
    for t in range(1, len(vals)):
        feats = _rolling_features_from_list(vals[:t])
        rows.append(feats)
        y.append(vals[t])
    X = pd.DataFrame(rows, index=s.index[1:])
    y = pd.Series(y, index=s.index[1:])
    return X, y


def gb_fit_and_forecast(series: pd.Series, horizon: int = 6) -> np.ndarray:
    """Fit GradientBoostingRegressor on engineered features and forecast horizon steps ahead recursively."""
    _gb_requirements()
    from sklearn.ensemble import GradientBoostingRegressor

    s = series.dropna().astype(float)
    if len(s) < 18:
        # Fallback to mean if not enough data
        mu = float(s.mean()) if len(s) else 0.0
        return mean_forecast_path(mu, horizon)

    X, y = _build_supervised_frame(s)
    if X.empty or y.empty:
        mu = float(s.mean()) if len(s) else 0.0
        return mean_forecast_path(mu, horizon)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    # Roll-forward simulation
    vals = s.values.tolist()
    preds = []
    for _ in range(horizon):
        feats = _rolling_features_from_list(vals)
        x_next = pd.DataFrame([feats])
        y_hat = float(model.predict(x_next)[0])
        preds.append(y_hat)
        vals.append(y_hat)
        # keep list from growing unbounded; cap at last 36 months
        if len(vals) > 36:
            vals = vals[-36:]
    return np.array(preds, dtype=float)


def _lstm_requirements():
    """Check if TensorFlow/Keras is available for LSTM models."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # noqa: F401
    except Exception as e:
        raise ImportError(
            "TensorFlow is required for LSTM forecasts. Please install 'tensorflow'."
        ) from e


def _create_sequences(data: np.ndarray, lookback: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training.
    
    Args:
        data: 1D array of monthly returns
        lookback: number of past months to use as input
    
    Returns:
        X: 3D array (samples, lookback, 1)
        y: 1D array of target values
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def lstm_fit_and_forecast(series: pd.Series, horizon: int = 6, lookback: int = 12) -> np.ndarray:
    """
    Fit an LSTM model on monthly returns and forecast horizon steps ahead.
    
    Args:
        series: pandas Series of monthly returns
        horizon: number of months to forecast
        lookback: number of past months to use as input sequence
    
    Returns:
        Array of forecasted monthly returns
    """
    _lstm_requirements()
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    import tensorflow as tf
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    s = series.dropna().astype(float)
    
    # Need at least lookback + 10 data points for meaningful training
    if len(s) < lookback + 10:
        # Fallback to mean forecast
        mu = float(s.mean()) if len(s) else 0.0
        return mean_forecast_path(mu, horizon)
    
    # Prepare data
    data = s.values
    
    # Normalize data (important for LSTM)
    data_mean = np.mean(data)
    data_std = np.std(data)
    if data_std == 0:
        data_std = 1.0
    data_normalized = (data - data_mean) / data_std
    
    # Create sequences
    X, y = _create_sequences(data_normalized, lookback)
    
    if len(X) < 5:
        # Not enough sequences, fallback
        mu = float(s.mean()) if len(s) else 0.0
        return mean_forecast_path(mu, horizon)
    
    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split into train/val (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Build optimized LSTM model
    model = keras.Sequential([
        # First LSTM layer with dropout
        layers.LSTM(64, activation='tanh', return_sequences=True, 
                   input_shape=(lookback, 1)),
        layers.Dropout(0.2),
        
        # Second LSTM layer
        layers.LSTM(32, activation='tanh', return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Early stopping to prevent overfitting
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # Train model (suppress verbose output)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=min(32, len(X_train)),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Generate forecasts recursively
    predictions = []
    current_sequence = data_normalized[-lookback:].copy()
    
    for _ in range(horizon):
        # Reshape for prediction
        X_pred = current_sequence.reshape(1, lookback, 1)
        
        # Predict next value (normalized)
        y_pred_norm = model.predict(X_pred, verbose=0)[0, 0]
        
        # Denormalize
        y_pred = y_pred_norm * data_std + data_mean
        predictions.append(float(y_pred))
        
        # Update sequence (roll forward with normalized prediction)
        current_sequence = np.append(current_sequence[1:], y_pred_norm)
    
    return np.array(predictions, dtype=float)


def forecast_next_6m_returns(
    nav_data: pd.DataFrame,
    models: Dict[str, str] = None,
    horizon_months: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast next 6 months returns for each fund using various models.

    models: mapping {fund_name: model_name} or a single model for all.
            supported models: 'AR1', 'MEAN', 'GB', 'LSTM'

    Returns:
      monthly_forecasts: DataFrame (index future months, columns funds) with monthly return forecasts.
      summary: DataFrame with per-fund cumulative forecast return over horizon.
    """
    if models is None:
        models = {}

    mrets = to_monthly_returns(nav_data)
    if mrets.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Build future month index starting next calendar month after last
    last_month = mrets.index.max()
    start_future = (last_month + pd.offsets.MonthEnd(1))
    future_index = pd.date_range(start=start_future, periods=horizon_months, freq='M')

    forecasts = pd.DataFrame(index=future_index, columns=mrets.columns, dtype=float)
    for col in mrets.columns:
        ser = mrets[col]
        model = models.get(col, models.get('*', 'AR1')).upper()
        if model == 'MEAN':
            mu = float(ser.mean()) if ser.notna().any() else 0.0
            path = mean_forecast_path(mu, horizon_months)
        elif model == 'AR1':
            c, phi = ar1_fit(ser)
            last_val = float(ser.dropna().iloc[-1]) if ser.notna().any() else 0.0
            path = ar1_forecast_path(last_value=last_val, c=c, phi=phi, horizon=horizon_months)
        elif model == 'GB':
            path = gb_fit_and_forecast(ser, horizon=horizon_months)
        elif model == 'LSTM':
            path = lstm_fit_and_forecast(ser, horizon=horizon_months)
        else:
            # default fallback
            c, phi = ar1_fit(ser)
            last_val = float(ser.dropna().iloc[-1]) if ser.notna().any() else 0.0
            path = ar1_forecast_path(last_value=last_val, c=c, phi=phi, horizon=horizon_months)
        forecasts[col] = path

    # Summary cumulative simple return over horizon: prod(1+r) - 1
    cumret = (1 + forecasts).prod(axis=0) - 1
    summary = pd.DataFrame({
        'forecast_6m_return': cumret
    }).sort_values('forecast_6m_return', ascending=False)

    return forecasts, summary'''
