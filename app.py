import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

# Add src directory to path
sys_path = str(Path(__file__).parent / "src")
if sys_path not in sys.path:
    sys.path.append(sys_path)

# Import local modules
from src.data_loader import load_mutual_fund_data, load_benchmark_data, preprocess_data
from src import metrics
from src.metrics import calculate_alpha_beta, calculate_information_ratio
from src import visualization
import src.ranking as ranking
from src.ranking import calculate_composite_score, get_top_funds

# Set page config
st.set_page_config(
    page_title="Advanced Fund Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    .stDataFrame {
        font-size: 0.9em;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Constants
DEFAULT_RISK_FREE_RATE = 0.06  # 6% annual risk-free rate

def main():
    st.title("📊 Advanced Fund Analysis Dashboard")
    st.caption("Analyze and compare mutual fund performance with advanced metrics")

    # Sidebar for file uploads and settings
    st.sidebar.header("Data Input")

    # File uploaders
    mf_file = st.sidebar.file_uploader(
        "Upload Mutual Fund NAV Data (CSV)",
        type="csv",
        key="mf_uploader"
    )

    nifty_file = st.sidebar.file_uploader(
        "Upload NIFTY Benchmark Data (CSV)",
        type="csv",
        key="nifty_uploader"
    )

    # Use sample data checkbox
    use_sample_data = st.sidebar.checkbox("Use sample data", value=True)

    # Date range selection
    st.sidebar.header("Analysis Period")

    # Default date range (can be adjusted based on data)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*5)  # 5 years by default

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [default_start, default_end],
        format="YYYY-MM-DD"
    )

    # Risk-free rate input
    risk_free_rate = st.sidebar.number_input(
        "Annual Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0,
        step=0.1
    ) / 100  # Convert to decimal

    # Load data based on user selection
    if use_sample_data or (mf_file is not None and nifty_file is not None):
        try:
            with st.spinner("Loading and processing data..."):
                if use_sample_data:
                    # Use the sample data files in the current directory
                    mf_path = Path("largecap_regular_growth_raw_10yrs.csv")
                    nifty_path = Path("nifty100_filtered_data.csv")

                    if not mf_path.exists() or not nifty_path.exists():
                        st.error("Sample data files not found. Please upload your own data files.")
                        return
                else:
                    # Save uploaded files temporarily
                    mf_path = Path("temp_mf_data.csv")
                    with open(mf_path, "wb") as f:
                        f.write(mf_file.getbuffer())

                    nifty_path = Path("temp_nifty_data.csv")
                    with open(nifty_path, "wb") as f:
                        f.write(nifty_file.getbuffer())

                # Load data using data_loader
                mf_data = load_mutual_fund_data(mf_path)
                benchmark_data = load_benchmark_data(nifty_path)

                # Process date range
                if len(date_range) == 2:
                    start_date = pd.to_datetime(date_range[0]).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(date_range[1]).strftime('%Y-%m-%d')
                else:
                    start_date = None
                    end_date = None

                # Preprocess data
                nav_data, benchmark_series = preprocess_data(
                    mf_data,
                    benchmark_data,
                    start_date=start_date,
                    end_date=end_date
                )

                # Display data info
                st.sidebar.success(f"Data loaded: {len(nav_data)} days, {len(nav_data.columns)} funds")

                # Fund selection with Select All option
                all_funds = sorted(nav_data.columns.tolist())

                # Add a 'Select All' checkbox
                select_all = st.sidebar.checkbox("Select All Funds", value=True)

                if select_all:
                    selected_funds = all_funds
                    st.sidebar.info(f"All {len(all_funds)} funds selected")
                else:
                    selected_funds = st.sidebar.multiselect(
                        "Or select specific funds:",
                        all_funds,
                        default=all_funds[:min(5, len(all_funds))]
                    )

                if not selected_funds:
                    st.warning("Please select at least one fund")
                    return

                # Filter data for selected funds
                filtered_nav_data = nav_data[selected_funds]

                # Create tabs for different sections
                tab1, tab2 = st.tabs(["Performance Metrics", "Ranking Metrics"])

                with tab1:
                    st.header("Performance Metrics")

                    ranking_criteria = st.selectbox(
                        "Ranking Criteria:",
                        ["Composite Score", "Momentum", "Consistency", "Risk-Adjusted Returns"],
                        key="performance_ranking"
                    )

                    # Calculate metrics weights based on selected criteria
                    if ranking_criteria == "Composite Score":
                        perf_metrics_weights = {
                            'sharpe_ratio': 0.25,
                            'sortino_ratio': 0.25,
                            'information_ratio': 0.2,
                            'max_drawdown': -0.1,  # Negative weight because lower is better
                            'annual_return': 0.2,
                            'annual_volatility': -0.1,  # Negative weight for lower volatility
                            'tracking_error': -0.1  # Negative weight for lower tracking error
                        }
                    elif ranking_criteria == "Momentum":
                        perf_metrics_weights = {
                            'annual_return': 0.4,
                            'sharpe_ratio': 0.3,
                            'beta': -0.2,  # Negative weight for lower beta
                            'max_drawdown': -0.1
                        }
                    elif ranking_criteria == "Consistency":
                        perf_metrics_weights = {
                            'sortino_ratio': 0.4,
                            'information_ratio': 0.3,
                            'max_drawdown': -0.2,
                            'annual_volatility': -0.1  # Negative weight for lower volatility
                        }
                    else:  # Risk-Adjusted Returns
                        perf_metrics_weights = {
                            'sharpe_ratio': 0.4,
                            'sortino_ratio': 0.4,
                            'max_drawdown': -0.2
                        }

                    # Calculate and display performance metrics
                    with st.spinner("Calculating performance metrics..."):
                        perf_metrics = calculate_composite_score(
                            filtered_nav_data,
                            benchmark_series,
                            risk_free_rate=risk_free_rate,
                            metrics_weights=perf_metrics_weights
                        )

                        # Display key performance indicators
                        st.subheader(f"Key Performance Indicators - {ranking_criteria}")

                        # Show top 3 funds based on composite score
                        top_performers = perf_metrics.nlargest(3, 'composite_score')

                        col1, col2, col3 = st.columns(3)

                        for i, (fund_name, metrics) in enumerate(top_performers.iterrows()):
                            with [col1, col2, col3][i]:
                                st.metric(
                                    label=f"{fund_name[:20]}..." if len(fund_name) > 20 else fund_name,
                                    value=f"{metrics['composite_score']:.3f}",
                                    delta=f"{metrics['annual_return']*100:.1f}%" if pd.notnull(metrics['annual_return']) else 'N/A'
                                )

                        # Display detailed metrics table
                        st.subheader("Detailed Performance Metrics")

                        # Format metrics for display
                        display_perf = perf_metrics.copy()
                        pct_cols = ['annual_return', 'annual_volatility', 'max_drawdown', 'tracking_error', 'alpha']
                        for col in pct_cols:
                            if col in display_perf.columns:
                                display_perf[col] = display_perf[col].apply(
                                    lambda x: f"{x*100:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
                                )

                        ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'information_ratio', 'beta', 'composite_score']
                        for col in ratio_cols:
                            if col in display_perf.columns:
                                display_perf[col] = display_perf[col].apply(
                                    lambda x: f"{x:.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
                                )

                        st.dataframe(
                            display_perf,
                            use_container_width=True,
                            height=min(600, 100 + 35 * len(display_perf))
                        )

                with tab2:
                    st.header("🏆 Fund Rankings")

                    # Add ranking criteria selector
                    ranking_criteria = st.selectbox(
                        "Select Ranking Criteria:",
                        ["Composite Score", "Momentum", "Consistency", "Risk-Adjusted Returns"],
                        index=0
                    )

                    # Calculate rankings based on selected criteria
                    with st.spinner("Calculating rankings..."):
                        if ranking_criteria == "Composite Score":
                            metrics_weights = {
                                'sharpe_ratio': 0.25,
                                'sortino_ratio': 0.25,
                                'information_ratio': 0.2,
                                'max_drawdown': -0.1,  # Negative weight because lower is better
                                'annual_return': 0.2,
                                'annual_volatility': -0.1,  # Negative weight for lower volatility
                                'tracking_error': -0.1  # Negative weight for lower tracking error
                            }
                        elif ranking_criteria == "Momentum":
                            metrics_weights = {
                                'annual_return': 0.4,
                                'sharpe_ratio': 0.3,
                                'beta': -0.2,  # Negative weight for lower beta
                                'max_drawdown': -0.1
                            }
                        elif ranking_criteria == "Consistency":
                            metrics_weights = {
                                'sortino_ratio': 0.4,
                                'information_ratio': 0.3,
                                'max_drawdown': -0.2,
                                'annual_volatility': -0.1  # Negative weight for lower volatility
                            }
                        else:  # Risk-Adjusted Returns
                            metrics_weights = {
                                'sharpe_ratio': 0.4,
                                'sortino_ratio': 0.4,
                                'max_drawdown': -0.2
                            }

                        # Calculate composite scores and get rankings
                        metrics_df = calculate_composite_score(
                            filtered_nav_data,
                            benchmark_series,
                            risk_free_rate=risk_free_rate,
                            metrics_weights=metrics_weights
                        )

                        # Get all funds with rankings
                        top_funds_df = get_top_funds(metrics_df, n=len(selected_funds))

                        # Add rank column
                        top_funds_df.insert(0, 'Rank', range(1, len(top_funds_df) + 1))

                        # Format the metrics for better display
                        display_metrics = top_funds_df.copy()

                        # Format percentages
                        pct_cols = ['annual_return', 'annual_volatility', 'max_drawdown', 'tracking_error', 'alpha']
                        for col in pct_cols:
                            if col in display_metrics.columns:
                                display_metrics[col] = display_metrics[col].apply(
                                    lambda x: f"{x*100:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
                                )

                        # Format ratios
                        ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'information_ratio', 'beta']
                        for col in ratio_cols:
                            if col in display_metrics.columns:
                                display_metrics[col] = display_metrics[col].apply(
                                    lambda x: f"{x:.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
                                )

                        # Display rankings with more information
                        st.subheader(f"Fund Rankings by {ranking_criteria}")
                        st.dataframe(
                            display_metrics,
                            use_container_width=True,
                            height=min(800, 100 + 35 * len(display_metrics)),
                            column_config={
                                'Rank': st.column_config.NumberColumn("Rank", width="small"),
                                'composite_score': st.column_config.NumberColumn("Score", format="%.4f"),
                            }
                        )

                        # Add download button for rankings
                        csv = top_funds_df.to_csv(index=True)
                        st.download_button(
                            label="📥 Download Full Rankings",
                            data=csv,
                            file_name=f"fund_rankings_{ranking_criteria.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"download_rankings_{ranking_criteria}"
                        )

                    # Calculate alpha and beta for each fund if benchmark is available
                    if benchmark_series is not None and not benchmark_series.empty:
                        alphas = []
                        betas = []
                        for fund in selected_funds:
                            fund_returns = filtered_nav_data[fund].pct_change().dropna()
                            alpha, beta = calculate_alpha_beta(
                                fund_returns,
                                benchmark_series.pct_change().dropna(),
                                risk_free_rate=risk_free_rate
                            )
                            alphas.append(alpha)
                            betas.append(beta)

                        # Add alpha and beta to the metrics
                        display_metrics['alpha'] = alphas
                        display_metrics['beta'] = betas

                        # Calculate information ratio for each fund
                        info_ratios = []
                        for fund in selected_funds:
                            fund_returns = filtered_nav_data[fund].pct_change().dropna()
                            info_ratio = calculate_information_ratio(
                                fund_returns,
                                benchmark_series.pct_change().dropna()
                            )
                            info_ratios.append(info_ratio)

                        display_metrics['information_ratio'] = info_ratios

                # Create metrics DataFrame from display_metrics
                metrics_df = display_metrics.copy()

                # Define columns for formatting
                percent_cols = ['annual_return', 'annual_volatility', 'max_drawdown', 'tracking_error', 'alpha']
                decimal_cols = ['sharpe_ratio', 'sortino_ratio', 'information_ratio', 'beta', 'composite_score']

                # Format percentage columns
                for col in percent_cols:
                    if col in metrics_df.columns:
                        metrics_df[col] = metrics_df[col].apply(
                            lambda x: f"{float(x)*100:.2f}%" if pd.notnull(x) and str(x).replace('.', '').replace('-', '').replace(' ', '').isdigit() else str(x)
                        )

                # Format decimal columns
                for col in decimal_cols:
                    if col in metrics_df.columns:
                        metrics_df[col] = metrics_df[col].apply(
                            lambda x: f"{float(x):.2f}" if pd.notnull(x) and str(x).replace('.', '').replace('-', '').replace(' ', '').replace('e', '').replace('+', '').isdigit() else str(x)
                        )

                # Prepare quarterly rankings for the last 3 years
                if not nav_data.empty:
                    # Get the last 3 years of quarterly dates
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.DateOffset(years=3)

                    # Generate all quarters in the 3-year period
                    all_quarters = pd.date_range(
                        start=start_date,
                        end=end_date,
                        freq='Q'
                    )

                    # Create a DataFrame with all funds and all quarters
                    columns = [f"{q.year} Q{q.quarter}" for q in all_quarters]
                    rankings_df = pd.DataFrame(index=selected_funds, columns=columns, dtype='float64')

                    # Debug: Print data availability
                    print(f"\n=== Data Availability Check ===")
                    print(f"Date range: {nav_data.index.min()} to {nav_data.index.max()}")
                    print(f"Selected funds: {len(selected_funds)} funds")
                    print(f"First few dates in data: {nav_data.index[:5].tolist()}")
                    print(f"First few funds: {selected_funds[:5]}")
                    print("\n=== Processing Quarters ===")

                    # Calculate rankings for each quarter
                    quarters_with_data = 0

                    for date in all_quarters:
                        # Get data for this specific quarter only
                        start_date = date - pd.offsets.QuarterEnd() + pd.offsets.Day(1)  # Start of quarter
                        q_data = nav_data[(nav_data.index >= start_date) & (nav_data.index <= date)]

                        # Debug info for this quarter
                        print(f"\nProcessing {date.year} Q{date.quarter} ({start_date} to {date}):")
                        print(f"  - Found {len(q_data)} data points")

                        # Skip if no data at all
                        if q_data.empty:
                            print("  - No data for this quarter")
                            continue

                        # Get funds with at least 2 data points in this quarter
                        valid_funds = [col for col in selected_funds if col in q_data.columns and q_data[col].count() >= 2]

                        print(f"  - Funds with enough data: {len(valid_funds)}/{len(selected_funds)}")

                        if not valid_funds:
                            print("  - No funds with enough data for this quarter")
                            continue  # No funds with enough data for this quarter

                        try:
                            # Calculate composite scores for this quarter
                            # Ensure we have benchmark data for this period
                            q_benchmark = benchmark_series[(benchmark_series.index >= start_date) &
                                                         (benchmark_series.index <= date)]

                            if q_benchmark.empty or len(q_benchmark) < 2:
                                print(f"  - Not enough benchmark data for {date.year} Q{date.quarter}")
                                continue

                            print(f"  - Calculating composite scores for {len(valid_funds)} funds")

                            q_metrics = calculate_composite_score(
                                nav_data=q_data[valid_funds],  # Only use funds with enough data
                                benchmark_series=q_benchmark,
                                risk_free_rate=risk_free_rate,
                                metrics_weights=metrics_weights
                            )

                            if not q_metrics.empty and 'composite_score' in q_metrics.columns:
                                rank_col = f"{date.year} Q{date.quarter}"
                                # Initialize series with NaN for all funds
                                ranked = pd.Series(index=selected_funds, dtype='float64')
                                # Fill in ranks for funds with valid data
                                ranked[q_metrics.index] = q_metrics['composite_score'].rank(ascending=False, method='min')
                                # Add to our rankings DataFrame
                                rankings_df[rank_col] = ranked.astype('Int64')
                                quarters_with_data += 1

                        except Exception as e:
                            print(f"Error calculating quarterly composite for {date}: {str(e)}")
                            continue

                    if quarters_with_data == 0:
                        print("\n=== No quarters with valid data found. Summary: ===")
                        print(f"- Total quarters checked: {len(all_quarters)}")
                        print(f"- Data date range: {nav_data.index.min()} to {nav_data.index.max()}")
                        print(f"- Number of funds with data: {len([f for f in selected_funds if f in nav_data.columns])}/{len(selected_funds)}")
                        print(f"- First few dates in data: {nav_data.index[:5].tolist()}")
                        st.warning("Insufficient historical data to generate quarterly rankings. Need at least one quarter with valid data.")
                        return

                    # Convert to numeric, ensuring all values are properly converted
                    rankings_df = rankings_df.apply(pd.to_numeric, errors='coerce')

                    # Only convert to Int64 if all values are finite numbers or NaN
                    def safe_convert_to_int(series):
                        # Check if all non-NA values are whole numbers
                        if (series.dropna() % 1 == 0).all():
                            return series.astype('Int64')
                        return series

                    rankings_df = rankings_df.apply(safe_convert_to_int)

                    # Drop columns that are all NaN
                    rankings_df = rankings_df.dropna(axis=1, how='all')

                    # Sort columns chronologically
                    if not rankings_df.empty and len(rankings_df.columns) > 0:
                        rankings_df = rankings_df[sorted(rankings_df.columns)]

                    if 'rankings_df' in locals() and not rankings_df.empty:
                        # Add current rank as the last column if we have composite scores
                        if 'composite_score' in metrics_df.columns:
                            current_rank = metrics_df['composite_score'].rank(ascending=False).astype(int)
                            rankings_df['Current Rank'] = current_rank

                        # Display the rankings table
                        st.subheader("Quarterly Rankings (Lower is Better)")
                        st.dataframe(
                            rankings_df,
                            use_container_width=True,
                            height=min(800, 100 + 35 * len(rankings_df)),
                            column_config={
                                col: st.column_config.NumberColumn(
                                    col,
                                    format="%d",
                                    help=f"Ranking as of {col} (1 = best)"
                                ) for col in rankings_df.columns
                            }
                        )
                    else:
                        st.warning("Insufficient historical data to generate quarterly rankings. Need at least 3 months of data.")

                    # Add download button for rankings
                    csv = rankings_df.to_csv()
                    st.download_button(
                        label="📥 Download Quarterly Rankings",
                        data=csv,
                        file_name="quarterly_rankings.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    else:
        st.warning(
            "Please upload both Mutual Fund and NIFTY benchmark CSV files "
            "or select 'Use sample data' to begin analysis."
        )

        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            ### Mutual Fund Data Format
            Your mutual fund NAV data should be a CSV file with at least these columns:
            - `date`: Date of the NAV (YYYY-MM-DD)
            - `scheme_name`: Name of the mutual fund scheme
            - `nav`: Net Asset Value of the fund

            ### Benchmark Data Format
            Your benchmark data (e.g., NIFTY) should be a CSV file with:
            - `date`: Date of the index value (YYYY-MM-DD)
            - `close` or `value`: Closing value of the index
            """)

if __name__ == "__main__":
    main()
