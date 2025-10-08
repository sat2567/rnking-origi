"""
Data loading and preprocessing utilities for mutual fund analysis.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mutual_fund_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and preprocess mutual fund NAV data from a CSV file.
    
    Args:
        file_path: Path to the mutual fund NAV CSV file
        
    Returns:
        DataFrame containing the processed mutual fund data with columns:
        - date: datetime index
        - scheme_name: name of the mutual fund scheme
        - nav: net asset value
    """
    try:
        logger.info(f"Loading mutual fund data from {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Standardize column names (case-insensitive, strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Ensure we have the expected columns
        expected_columns = ['Date', 'Scheme Name', 'NAV']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"CSV must contain these columns: {expected_columns}")
        
        # Convert date column to datetime with dayfirst=True for DD-MM-YYYY format
        df['date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Drop rows with invalid dates
        if df['date'].isna().any():
            logger.warning(f"Dropped {df['date'].isna().sum()} rows with invalid dates")
            df = df.dropna(subset=['date'])
        
        # Rename and select only needed columns
        df = df.rename(columns={
            'Scheme Name': 'scheme_name',
            'NAV': 'nav'
        })
        
        # Convert NAV to numeric, handle any non-numeric values
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        
        # Drop rows with invalid NAV values
        if df['nav'].isna().any():
            logger.warning(f"Dropped {df['nav'].isna().sum()} rows with invalid NAV values")
            df = df.dropna(subset=['nav'])
        
        # Sort by date
        df = df.sort_values('date')
        
        df = df.drop_duplicates(['date', 'scheme_name'], keep='last')
        
        logger.info(f"Successfully loaded {len(df)} NAV records for {df['scheme_name'].nunique()} funds")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Select and return only the required columns
        return df[['date', 'scheme_name', 'nav']]
            
        # Rename columns for consistency
        df = df.rename(columns={
            'Date': 'date',
            'Scheme Name': 'scheme_name',
            'NAV': 'nav'
        })
        
        # Convert date strings to datetime (handling DD-MM-YYYY format)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce', format='%d-%m-%Y')
        
        # Drop any rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Convert NAV to numeric, handle any non-numeric values
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        
        # Drop rows with invalid NAV values
        df = df.dropna(subset=['nav'])
        
        # Handle duplicate dates (keep last)
        df = df.drop_duplicates(['date', 'scheme_name'], keep='last')
        
        logger.info(f"Successfully loaded {len(df)} NAV records for {df['scheme_name'].nunique()} funds")
        return df
        
    except Exception as e:
        logger.error(f"Error loading mutual fund data: {str(e)}")
        raise


def load_benchmark_data(file_path: Union[str, Path]) -> pd.Series:
    """
    Load and preprocess benchmark index data (e.g., NIFTY).
    
    Args:
        file_path: Path to the benchmark index CSV file
        
    Returns:
        Series with date index and benchmark values
    """
    try:
        logger.info(f"Loading benchmark data from {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Create a mapping of lowercase column names to original names
        col_map = {col.lower(): col for col in df.columns}
        
        # Standardize column names (case-insensitive)
        df.columns = df.columns.str.strip().str.lower()
        
        # Map common column names to standard names
        date_col = next((col for col in ['date', 'date_', 'timestamp', 'time'] 
                        if col in df.columns), None)
        
        close_col = next((col for col in ['close', 'price', 'value', 'adj close', 'adj_close'] 
                         if col in df.columns), None)
        
        # If we found standard columns, rename them
        if date_col:
            df = df.rename(columns={date_col: 'date'})
        if close_col and close_col != 'close':
            df = df.rename(columns={close_col: 'close'})
        
        # If we couldn't find standard columns, try to identify them
        if 'date' not in df.columns:
            # Look for date-like columns
            date_cols = []
            for col in df.columns:
                try:
                    # Try to convert to datetime to see if it's a date column
                    sample = df[col].dropna().sample(min(10, len(df)))
                    if pd.to_datetime(sample, errors='coerce').notna().all():
                        date_cols.append(col)
                except:
                    continue
            
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'date'})
            else:
                # If no date column found, try to use the index if it's a datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={'index': 'date'})
                else:
                    raise ValueError("No date column found in benchmark data")
        
        if 'close' not in df.columns:
            # Try to find a numeric column to use as close price
            numeric_cols = [col for col in df.columns if col != 'date' and pd.api.types.is_numeric_dtype(df[col])]
            if numeric_cols:
                df = df.rename(columns={numeric_cols[0]: 'close'})
            else:
                raise ValueError("No numeric column found in benchmark data")
        
        # Convert date column to datetime
        date_parsers = [
            lambda x: pd.to_datetime(x, dayfirst=True, errors='coerce'),
            lambda x: pd.to_datetime(x, dayfirst=False, errors='coerce'),
            lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce'),
            lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'),
            lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'),
            lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='coerce')
        ]
        
        # Find the best date parser for the date column
        best_parser = max(
            date_parsers,
            key=lambda p: (lambda s: (s.dropna().nunique(), len(s.dropna())))(p(df['date']))
        )
        
        df['date'] = best_parser(df['date'])
        
        # Ensure we have valid data
        df = df.dropna(subset=['date', 'close'])
        
        # Convert close price to numeric
        df['close'] = pd.to_numeric(
            df['close'].astype(str).str.replace('[^\d.-]', '', regex=True),
            errors='coerce'
        )
        
        # Sort by date and remove duplicates
        df = df.sort_values('date')
        df = df.drop_duplicates('date', keep='last')
        
        # Set index to date and keep only close column for the series
        series = df.set_index('date')['close']
        
        logger.info(f"Successfully loaded {len(series)} benchmark records from {series.index.min()} to {series.index.max()}")
        return series
        
    except Exception as e:
        logger.error(f"Error loading benchmark data: {str(e)}")
        raise


def preprocess_data(
    mf_data: pd.DataFrame, 
    benchmark_series: pd.Series,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess and align mutual fund and benchmark data.
    
    Args:
        mf_data: Raw mutual fund data from load_mutual_fund_data
        benchmark_series: Raw benchmark data from load_benchmark_data
        start_date: Optional start date (YYYY-MM-DD or datetime)
        end_date: Optional end date (YYYY-MM-DD or datetime)
        
    Returns:
        Tuple of (nav_pivot, benchmark_series_aligned)
        - nav_pivot: DataFrame with date index and fund NAVs as columns
        - benchmark_series_aligned: Benchmark values aligned with nav_pivot index
    """
    try:
        logger.info("Preprocessing and aligning data...")
        
        # Convert to datetime if strings are provided
        if start_date and isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Filter mutual fund data by date range
        if start_date is not None:
            mf_data = mf_data[mf_data['date'] >= start_date]
        if end_date is not None:
            mf_data = mf_data[mf_data['date'] <= end_date]
        
        if mf_data.empty:
            raise ValueError("No mutual fund data available for the selected date range")
        
        # Pivot to get NAVs by date and fund
        nav_pivot = mf_data.pivot(index='date', columns='scheme_name', values='nav')
        
        # Sort index
        nav_pivot = nav_pivot.sort_index()
        
        # Forward fill missing values (same NAV as previous day)
        nav_pivot = nav_pivot.ffill()
        
        # Filter benchmark data by date range
        if start_date is not None:
            benchmark_series = benchmark_series[benchmark_series.index >= start_date]
        if end_date is not None:
            benchmark_series = benchmark_series[benchmark_series.index <= end_date]
        
        if benchmark_series.empty:
            raise ValueError("No benchmark data available for the selected date range")
        
        # Align benchmark with mutual fund data dates
        common_dates = nav_pivot.index.intersection(benchmark_series.index)
        
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between mutual fund and benchmark data")
        
        nav_pivot = nav_pivot.loc[common_dates]
        benchmark_series_aligned = benchmark_series[common_dates]
        
        logger.info(f"Preprocessing complete. Data available from {common_dates[0]} to {common_dates[-1]}")
        
        return nav_pivot, benchmark_series_aligned
        
    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        raise
