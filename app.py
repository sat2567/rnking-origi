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
from src.backtesting import run_backtest
from src.ranking import calculate_composite_score, get_top_funds
#from src.forecasting import forecast_next_6m_returns, to_monthly_returns

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

    # Custom Weights Section - keeping sidebar version for backward compatibility
    st.sidebar.header("📊 Custom Weights Configuration (Legacy)")

    use_custom_weights = st.sidebar.checkbox("Use Custom Weights (Legacy)", value=False, help="Enable to set your own weights for composite score calculation")

    if use_custom_weights:
        st.sidebar.subheader("Adjust Metric Weights")

        # Create columns for better layout
        col1, col2 = st.sidebar.columns(2)

        with col1:
            sharpe_weight = st.sidebar.slider("Sharpe Ratio", -1.0, 1.0, 0.25, 0.01,
                                            help="Risk-adjusted return measure")
            sortino_weight = st.sidebar.slider("Sortino Ratio", -1.0, 1.0, 0.25, 0.01,
                                             help="Downside risk-adjusted return")
            info_weight = st.sidebar.slider("Information Ratio", -1.0, 1.0, 0.2, 0.01,
                                          help="Risk-adjusted excess return")
            annual_return_weight = st.sidebar.slider("Annual Return", -1.0, 1.0, 0.2, 0.01,
                                                   help="Total annual return")

        with col2:
            max_drawdown_weight = st.sidebar.slider("Max Drawdown", -1.0, 1.0, -0.1, 0.01,
                                                  help="Maximum peak-to-trough decline (negative = better when lower)")
            volatility_weight = st.sidebar.slider("Annual Volatility", -1.0, 1.0, -0.1, 0.01,
                                                help="Standard deviation of returns (negative = better when lower)")
            tracking_error_weight = st.sidebar.slider("Tracking Error", -1.0, 1.0, -0.1, 0.01,
                                                    help="Deviation from benchmark (negative = better when lower)")
            beta_weight = st.sidebar.slider("Beta", -1.0, 1.0, 0.0, 0.01,
                                          help="Market sensitivity (negative = better when lower for risk reduction)")

        # Custom weights dictionary
        custom_weights = {
            'sharpe_ratio': sharpe_weight,
            'sortino_ratio': sortino_weight,
            'information_ratio': info_weight,
            'max_drawdown': max_drawdown_weight,
            'annual_return': annual_return_weight,
            'annual_volatility': volatility_weight,
            'tracking_error': tracking_error_weight,
            'beta': beta_weight
        }

        # Display total weight sum
        total_weight = sum(abs(w) for w in custom_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.sidebar.warning(f"⚠️ Total absolute weights: {total_weight:.2f} (recommended: ~1.0)")
        else:
            st.sidebar.success(f"✅ Total absolute weights: {total_weight:.2f}")

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

                # Custom Weights Controls - prominently displayed in main dashboard
                st.header("⚙️ Custom Composite Score Weights")

                # Quick preset buttons
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if st.button("Conservative", help="Focus on low risk and stability"):
                        conservative_weights = {
                            'sharpe_ratio': 0.3, 'sortino_ratio': 0.3, 'max_drawdown': -0.4,
                            'annual_volatility': -0.3, 'information_ratio': 0.2,
                            'annual_return': 0.1, 'tracking_error': -0.2, 'beta': -0.2
                        }
                        st.session_state.custom_weights = conservative_weights
                        st.rerun()
                with col2:
                    if st.button("Balanced", help="Balanced risk-return approach"):
                        balanced_weights = {
                            'sharpe_ratio': 0.25, 'sortino_ratio': 0.25, 'information_ratio': 0.2,
                            'annual_return': 0.2, 'max_drawdown': -0.15, 'annual_volatility': -0.1,
                            'tracking_error': -0.1, 'beta': 0.0
                        }
                        st.session_state.custom_weights = balanced_weights
                        st.rerun()
                with col3:
                    if st.button("Growth", help="Focus on high returns"):
                        growth_weights = {
                            'annual_return': 0.4, 'sharpe_ratio': 0.3, 'information_ratio': 0.2,
                            'beta': 0.1, 'max_drawdown': -0.1, 'annual_volatility': 0.0,
                            'sortino_ratio': 0.1, 'tracking_error': 0.0
                        }
                        st.session_state.custom_weights = growth_weights
                        st.rerun()
                with col4:
                    if st.button("Risk-Averse", help="Minimize all risks"):
                        risk_averse_weights = {
                            'max_drawdown': -0.35, 'annual_volatility': -0.25, 'tracking_error': -0.2,
                            'beta': -0.2, 'sharpe_ratio': 0.2, 'sortino_ratio': 0.15,
                            'information_ratio': 0.1, 'annual_return': 0.05
                        }
                        st.session_state.custom_weights = risk_averse_weights
                        st.rerun()
                with col5:
                    if st.button("Reset", help="Reset to default balanced weights"):
                        if 'custom_weights' in st.session_state:
                            del st.session_state.custom_weights
                        st.rerun()

                # Main custom weights interface
                use_custom_weights_main = st.checkbox(
                    "🎛️ **Enable Advanced Custom Weights**",
                    value=bool(st.session_state.get('custom_weights_active', False)),
                    help="Unlock full control over composite score calculation"
                )

                if use_custom_weights_main:
                    st.session_state.custom_weights_active = True

                    # Get current weights from session state or defaults
                    current_weights = st.session_state.get('custom_weights', {
                        'sharpe_ratio': 0.25, 'sortino_ratio': 0.25, 'information_ratio': 0.2,
                        'annual_return': 0.2, 'max_drawdown': -0.1, 'annual_volatility': -0.1,
                        'tracking_error': -0.1, 'beta': 0.0
                    })

                    st.markdown("### 🎚️ Fine-tune Your Strategy")
                    st.markdown("Adjust each metric's importance in the composite score (-1.0 to +1.0):")

                    # Create an expandable section for advanced controls
                    with st.expander("🔧 Advanced Weight Controls", expanded=True):
                        # Two-column layout for weights
                        weight_col1, weight_col2 = st.columns(2)

                        with weight_col1:
                            st.markdown("**📈 Return & Risk Metrics**")
                            cw_sharpe = st.slider("Sharpe Ratio", -1.0, 1.0, current_weights.get('sharpe_ratio', 0.25), 0.01,
                                                help="Risk-adjusted return measure")
                            cw_sortino = st.slider("Sortino Ratio", -1.0, 1.0, current_weights.get('sortino_ratio', 0.25), 0.01,
                                                 help="Downside risk-adjusted return")
                            cw_info = st.slider("Information Ratio", -1.0, 1.0, current_weights.get('information_ratio', 0.2), 0.01,
                                              help="Risk-adjusted excess return")
                            cw_return = st.slider("Annual Return", -1.0, 1.0, current_weights.get('annual_return', 0.2), 0.01,
                                                help="Total annual return")

                        with weight_col2:
                            st.markdown("**⚠️ Risk Metrics (Negative = Better)**")
                            cw_drawdown = st.slider("Max Drawdown", -1.0, 1.0, current_weights.get('max_drawdown', -0.1), 0.01,
                                                  help="Maximum peak-to-trough decline")
                            cw_volatility = st.slider("Annual Volatility", -1.0, 1.0, current_weights.get('annual_volatility', -0.1), 0.01,
                                                    help="Standard deviation of returns")
                            cw_tracking = st.slider("Tracking Error", -1.0, 1.0, current_weights.get('tracking_error', -0.1), 0.01,
                                                  help="Deviation from benchmark")
                            cw_beta = st.slider("Beta", -1.0, 1.0, current_weights.get('beta', 0.0), 0.01,
                                              help="Market sensitivity")

                        # Update session state with new weights
                        new_custom_weights = {
                            'sharpe_ratio': cw_sharpe,
                            'sortino_ratio': cw_sortino,
                            'information_ratio': cw_info,
                            'annual_return': cw_return,
                            'max_drawdown': cw_drawdown,
                            'annual_volatility': cw_volatility,
                            'tracking_error': cw_tracking,
                            'beta': cw_beta
                        }
                        st.session_state.custom_weights = new_custom_weights

                        # Display weight summary
                        total_weight = sum(abs(w) for w in new_custom_weights.values())
                        weight_status = "✅ Good" if abs(total_weight - 1.0) < 0.1 else "⚠️ Adjust for balance"
                        st.metric("Total Weight Magnitude", f"{total_weight:.2f}", delta=weight_status)

                        # Weight visualization
                        weight_df = pd.DataFrame({
                            'Metric': list(new_custom_weights.keys()),
                            'Weight': list(new_custom_weights.values())
                        })
                        weight_df['Metric'] = weight_df['Metric'].str.replace('_', ' ').str.title()

                        # Create a bar chart of weights
                        fig = px.bar(weight_df, x='Metric', y='Weight',
                                   title='Current Weight Distribution',
                                   color='Weight', color_continuous_scale='RdYlGn',
                                   range_color=[-1, 1])
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.session_state.custom_weights_active = False
                    if 'custom_weights' in st.session_state:
                        del st.session_state.custom_weights

                st.markdown("---")  # Separator

                # Create tabs for different sections
                tab1, tab2, tab3,  tab5 = st.tabs([
                    "Performance Metrics",
                    "Ranking Metrics",
                    "Backtesting",
                  #  "Predictive Analytics",
                    "Documentation"
                ])

                with tab1:
                    st.header("Performance Metrics")

                    # Ranking criteria selector - only show if not using custom weights from main dashboard
                    if not st.session_state.get('custom_weights_active', False) and not use_custom_weights:
                        ranking_criteria = st.selectbox(
                            "Ranking Criteria:",
                            ["Composite Score", "Momentum", "Consistency", "Risk-Adjusted Returns"],
                            key="performance_ranking"
                        )
                        criteria_text = ranking_criteria
                    elif st.session_state.get('custom_weights_active', False):
                        ranking_criteria = "Custom Weights"
                        criteria_text = "Custom Weights"
                        st.info("🔧 Using custom weights from main dashboard for composite score calculation")
                    elif use_custom_weights:
                        ranking_criteria = "Legacy Custom Weights"
                        criteria_text = "Legacy Custom Weights"
                        st.info("🔧 Using legacy custom weights from sidebar")
                    else:
                        ranking_criteria = "Default"
                        criteria_text = "Default"

                    # Calculate metrics weights based on selected criteria or custom weights
                    if st.session_state.get('custom_weights_active', False) and 'custom_weights' in st.session_state:
                        perf_metrics_weights = st.session_state.custom_weights
                    elif use_custom_weights:
                        perf_metrics_weights = custom_weights
                    elif ranking_criteria == "Composite Score":
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
                        st.subheader(f"Key Performance Indicators - {criteria_text}")

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

                    # Add ranking criteria selector - only show if not using custom weights from main dashboard
                    if not st.session_state.get('custom_weights_active', False) and not use_custom_weights:
                        ranking_criteria = st.selectbox(
                            "Select Ranking Criteria:",
                            ["Composite Score", "Momentum", "Consistency", "Risk-Adjusted Returns"],
                            index=0
                        )
                    elif st.session_state.get('custom_weights_active', False):
                        ranking_criteria = "Custom Weights"
                        st.info("🔧 Using custom weights from main dashboard for ranking calculation")
                    elif use_custom_weights:
                        ranking_criteria = "Legacy Custom Weights"
                        st.info("🔧 Using legacy custom weights from sidebar for ranking calculation")
                    else:
                        ranking_criteria = "Default"

                    # Calculate rankings based on selected criteria or custom weights
                    with st.spinner("Calculating rankings..."):
                        if st.session_state.get('custom_weights_active', False) and 'custom_weights' in st.session_state:
                            metrics_weights = st.session_state.custom_weights
                        elif use_custom_weights:
                            metrics_weights = custom_weights
                        elif ranking_criteria == "Composite Score":
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
                    
                    # Quarterly Rankings Section
                    st.divider()
                    st.subheader("📊 Quarterly Composite Score Rankings")
                    st.caption(f"Rankings based on: **{ranking_criteria}** criteria (Lower rank = Better performance)")
                    
                    if not nav_data.empty:
                        # Get the last 3 years of quarterly dates
                        end_date = pd.Timestamp.now()
                        start_date_qtr = end_date - pd.DateOffset(years=3)

                        # Generate all quarters in the 3-year period
                        all_quarters = pd.date_range(
                            start=start_date_qtr,
                            end=end_date,
                            freq='Q'
                        )

                        # Create a DataFrame with all funds and all quarters
                        columns = [f"{q.year} Q{q.quarter}" for q in all_quarters]
                        rankings_df = pd.DataFrame(index=selected_funds, columns=columns, dtype='float64')

                        # Optional on-screen debug logs
                        debug_logs = st.checkbox("Show quarterly rankings debug logs", value=False)
                        def dbg(msg: str):
                            if debug_logs:
                                st.text(str(msg))
                        
                        if debug_logs:
                            dbg("=== Data Availability Check ===")
                            dbg(f"Date range: {nav_data.index.min()} to {nav_data.index.max()}")
                            dbg(f"Selected funds: {len(selected_funds)} funds")
                            dbg(f"Using ranking criteria: {ranking_criteria}")
                            dbg(f"Metrics weights: {metrics_weights}")
                            dbg("=== Processing Quarters ===")

                        # Calculate rankings for each quarter
                        quarters_with_data = 0

                        for date in all_quarters:
                            # Get data for this specific quarter only
                            start_qtr = date - pd.offsets.QuarterEnd() + pd.offsets.Day(1)  # Start of quarter
                            q_data = nav_data[(nav_data.index >= start_qtr) & (nav_data.index <= date)]

                            if debug_logs:
                                dbg(f"Processing {date.year} Q{date.quarter} ({start_qtr} to {date}):")
                                dbg(f"  - Found {len(q_data)} data points")

                            # Skip if no data at all
                            if q_data.empty:
                                if debug_logs:
                                    dbg("  - No data for this quarter")
                                continue

                            # Get funds with at least 2 data points in this quarter
                            valid_funds = [col for col in selected_funds if col in q_data.columns and q_data[col].count() >= 2]

                            if debug_logs:
                                dbg(f"  - Funds with enough data: {len(valid_funds)}/{len(selected_funds)}")

                            if not valid_funds:
                                if debug_logs:
                                    dbg("  - No funds with enough data for this quarter")
                                continue

                            try:
                                # Calculate composite scores for this quarter using SAME criteria
                                q_benchmark = benchmark_series[(benchmark_series.index >= start_qtr) &
                                                             (benchmark_series.index <= date)]

                                if q_benchmark.empty or len(q_benchmark) < 2:
                                    if debug_logs:
                                        dbg(f"  - Not enough benchmark data for {date.year} Q{date.quarter}")
                                    continue

                                if debug_logs:
                                    dbg(f"  - Calculating composite scores for {len(valid_funds)} funds")

                                q_metrics = calculate_composite_score(
                                    nav_data=q_data[valid_funds],
                                    benchmark_series=q_benchmark,
                                    risk_free_rate=risk_free_rate,
                                    metrics_weights=metrics_weights  # Uses the same weights from ranking criteria!
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
                                if debug_logs:
                                    dbg(f"Error calculating quarterly composite for {date}: {str(e)}")
                                continue

                        if quarters_with_data == 0:
                            st.warning("Insufficient historical data to generate quarterly rankings. Need at least one quarter with valid data.")
                        else:
                            # Convert to numeric
                            rankings_df = rankings_df.apply(pd.to_numeric, errors='coerce')

                            # Convert to Int64 where appropriate
                            def safe_convert_to_int(series):
                                if (series.dropna() % 1 == 0).all():
                                    return series.astype('Int64')
                                return series

                            rankings_df = rankings_df.apply(safe_convert_to_int)

                            # Drop columns that are all NaN
                            rankings_df = rankings_df.dropna(axis=1, how='all')

                            # Sort columns chronologically
                            if not rankings_df.empty and len(rankings_df.columns) > 0:
                                rankings_df = rankings_df[sorted(rankings_df.columns)]

                            if not rankings_df.empty:
                                # Add current rank as the last column
                                if 'composite_score' in metrics_df.columns:
                                    current_rank = metrics_df['composite_score'].rank(ascending=False).astype(int)
                                    rankings_df['Current Rank'] = current_rank

                                # Display the rankings table
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

                                # Add download button for quarterly rankings
                                qtr_csv = rankings_df.to_csv()
                                st.download_button(
                                    label="📥 Download Quarterly Rankings",
                                    data=qtr_csv,
                                    file_name=f"quarterly_rankings_{ranking_criteria.lower().replace(' ', '_')}.csv",
                                    mime="text/csv",
                                    key="download_quarterly_rankings"
                                )
                            else:
                                st.warning("Insufficient historical data to generate quarterly rankings.")

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

                # Backtesting Tab
                with tab3:
                    st.header("📈 Backtesting")

                    # Determine default backtest date range from data
                    bt_min = pd.to_datetime(nav_data.index.min()) if not nav_data.empty else pd.Timestamp.now() - pd.DateOffset(years=3)
                    bt_max = pd.to_datetime(nav_data.index.max()) if not nav_data.empty else pd.Timestamp.now()

                    bt_col1, bt_col2 = st.columns(2)
                    with bt_col1:
                        bt_dates = st.date_input(
                            "Backtest Date Range",
                            [bt_min, bt_max],
                            min_value=bt_min.to_pydatetime(),
                            max_value=bt_max.to_pydatetime(),
                            format="YYYY-MM-DD",
                            key="bt_date_range"
                        )
                        rebalance = st.selectbox("Rebalance Frequency", ["Monthly", "Quarterly", "Semi-Annual (6M)"], index=0)
                    with bt_col2:
                        lookback_days = st.slider("Lookback Window (days)", min_value=90, max_value=756, value=252, step=10)
                        max_top = max(1, min(10, len(selected_funds)))
                        top_n = st.slider("Top N Funds", min_value=1, max_value=max_top, value=min(5, max_top))

                    # Choose weights source
                    st.subheader("Weights Source")
                    if st.session_state.get('custom_weights_active', False) and 'custom_weights' in st.session_state:
                        st.success("Using dashboard custom weights")
                        bt_weights = st.session_state.custom_weights
                    elif use_custom_weights:
                        st.info("Using legacy sidebar custom weights")
                        bt_weights = custom_weights
                    else:
                        bt_criteria = st.selectbox(
                            "Ranking Criteria for Backtest",
                            ["Composite Score", "Momentum", "Consistency", "Risk-Adjusted Returns"],
                            key="bt_ranking_criteria"
                        )
                        if bt_criteria == "Composite Score":
                            bt_weights = {
                                'sharpe_ratio': 0.25,
                                'sortino_ratio': 0.25,
                                'information_ratio': 0.2,
                                'max_drawdown': -0.1,
                                'annual_return': 0.2,
                                'annual_volatility': -0.1,
                                'tracking_error': -0.1
                            }
                        elif bt_criteria == "Momentum":
                            bt_weights = {
                                'annual_return': 0.4,
                                'sharpe_ratio': 0.3,
                                'beta': -0.2,
                                'max_drawdown': -0.1
                            }
                        elif bt_criteria == "Consistency":
                            bt_weights = {
                                'sortino_ratio': 0.4,
                                'information_ratio': 0.3,
                                'max_drawdown': -0.2,
                                'annual_volatility': -0.1
                            }

                    # Inline fine-tune weights (overrides selection above when enabled)
                    fine_tune = st.checkbox("🎚️ Fine-tune weights here", value=False, help="Adjust weights just for this backtest run")
                    if fine_tune:
                        base = bt_weights.copy()
                        with st.expander("🔧 Fine-tune Weight Sliders", expanded=True):
                            c1, c2 = st.columns(2)
                            with c1:
                                w_sharpe = st.slider("Sharpe Ratio", -1.0, 1.0, float(base.get('sharpe_ratio', 0.25)), 0.01)
                                w_sortino = st.slider("Sortino Ratio", -1.0, 1.0, float(base.get('sortino_ratio', 0.25)), 0.01)
                                w_info = st.slider("Information Ratio", -1.0, 1.0, float(base.get('information_ratio', 0.2)), 0.01)
                                w_return = st.slider("Annual Return", -1.0, 1.0, float(base.get('annual_return', 0.2)), 0.01)
                            with c2:
                                w_drawdown = st.slider("Max Drawdown", -1.0, 1.0, float(base.get('max_drawdown', -0.1)), 0.01)
                                w_vol = st.slider("Annual Volatility", -1.0, 1.0, float(base.get('annual_volatility', -0.1)), 0.01)
                                w_track = st.slider("Tracking Error", -1.0, 1.0, float(base.get('tracking_error', -0.1)), 0.01)
                                w_beta = st.slider("Beta", -1.0, 1.0, float(base.get('beta', 0.0)), 0.01)

                            bt_weights = {
                                'sharpe_ratio': w_sharpe,
                                'sortino_ratio': w_sortino,
                                'information_ratio': w_info,
                                'annual_return': w_return,
                                'max_drawdown': w_drawdown,
                                'annual_volatility': w_vol,
                                'tracking_error': w_track,
                                'beta': w_beta,
                            }

                            # Summary + visualization
                            mag = sum(abs(v) for v in bt_weights.values())
                            st.metric("Total Weight Magnitude", f"{mag:.2f}", delta=("✅ ~1.0" if abs(mag-1.0) < 0.1 else "⚠️ consider ~1.0"))
                            wdf = pd.DataFrame({
                                'Metric': list(bt_weights.keys()),
                                'Weight': list(bt_weights.values())
                            })
                            wdf['Metric'] = wdf['Metric'].str.replace('_',' ').str.title()
                            wfig = px.bar(wdf, x='Metric', y='Weight', title='Backtest Weight Distribution', color='Weight', color_continuous_scale='RdYlGn', range_color=[-1,1])
                            wfig.update_layout(height=260, margin=dict(t=40,b=10))
                            st.plotly_chart(wfig, use_container_width=True)

                    # Preset save/load controls
                    st.subheader("Presets")
                    if 'weight_presets' not in st.session_state:
                        st.session_state.weight_presets = {}
                    preset_cols = st.columns([2, 2, 1])
                    with preset_cols[0]:
                        preset_name = st.text_input("Preset name", value="My Strategy", key="bt_preset_name")
                        if st.button("💾 Save preset", key="bt_save_preset"):
                            if preset_name.strip():
                                st.session_state.weight_presets[preset_name.strip()] = dict(bt_weights)
                                st.success(f"Saved preset '{preset_name.strip()}'")
                            else:
                                st.warning("Enter a preset name before saving.")
                    with preset_cols[1]:
                        load_choice = st.selectbox(
                            "Load preset",
                            ["-- Select --"] + list(st.session_state.weight_presets.keys()),
                            key="bt_load_preset"
                        )
                        if load_choice and load_choice != "-- Select --":
                            bt_weights = dict(st.session_state.weight_presets[load_choice])
                            st.info(f"Loaded preset '{load_choice}'")
                    with preset_cols[2]:
                        if st.button("🗑️ Delete", key="bt_delete_preset"):
                            if load_choice and load_choice != "-- Select --":
                                del st.session_state.weight_presets[load_choice]
                                st.success(f"Deleted preset '{load_choice}'")
                            else:
                                st.warning("Select a preset to delete.")

                    # Prepare data for backtest
                    if isinstance(bt_dates, (list, tuple)) and len(bt_dates) == 2:
                        bt_start = pd.to_datetime(bt_dates[0])
                        bt_end = pd.to_datetime(bt_dates[1])
                    else:
                        bt_start, bt_end = bt_min, bt_max

                    nav_bt = filtered_nav_data[(filtered_nav_data.index >= bt_start) & (filtered_nav_data.index <= bt_end)]
                    bench_bt = benchmark_series[(benchmark_series.index >= bt_start) & (benchmark_series.index <= bt_end)]

                    if nav_bt.empty or bench_bt.empty:
                        st.warning("No data available for the selected backtest period.")
                    else:
                        # Multi-strategy comparison
                        # Build available strategies map
                        strategies_map = {
                            "Current Config": dict(bt_weights)
                        }
                        # Built-ins
                        strategies_map.update({
                            "Composite": {'sharpe_ratio': 0.25, 'sortino_ratio': 0.25, 'information_ratio': 0.2, 'max_drawdown': -0.1, 'annual_return': 0.2, 'annual_volatility': -0.1, 'tracking_error': -0.1},
                            "Momentum": {'annual_return': 0.4, 'sharpe_ratio': 0.3, 'beta': -0.2, 'max_drawdown': -0.1},
                            "Consistency": {'sortino_ratio': 0.4, 'information_ratio': 0.3, 'max_drawdown': -0.2, 'annual_volatility': -0.1},
                            "Risk-Adjusted": {'sharpe_ratio': 0.4, 'sortino_ratio': 0.4, 'max_drawdown': -0.2},
                        })
                        # Dashboard custom as an option
                        if st.session_state.get('custom_weights_active', False) and 'custom_weights' in st.session_state:
                            strategies_map["Dashboard Custom"] = dict(st.session_state.custom_weights)
                        # User presets
                        for pname, pweights in st.session_state.get('weight_presets', {}).items():
                            strategies_map[f"Preset: {pname}"] = dict(pweights)

                        compare_choices = st.multiselect(
                            "Compare strategies (select 2 or more to overlay)",
                            options=list(strategies_map.keys()),
                            default=[]
                        )

                        if compare_choices:
                            try:
                                # Map UI frequency to code
                                if rebalance == 'Monthly':
                                    rb = 'M'
                                elif rebalance == 'Quarterly':
                                    rb = 'Q'
                                else:
                                    rb = '6M'

                                combined = pd.DataFrame()
                                metrics_list = []
                                for name in compare_choices:
                                    wts = strategies_map[name]
                                    try:
                                        eq_i, m_i, _hold_i = run_backtest(
                                            nav_data=nav_bt,
                                            benchmark_series=bench_bt,
                                            metrics_weights=wts,
                                            risk_free_rate=risk_free_rate,
                                            rebalance_freq=rb,
                                            lookback_days=lookback_days,
                                            top_n=top_n
                                        )
                                    except ValueError:
                                        eq_i, m_i = run_backtest(
                                            nav_data=nav_bt,
                                            benchmark_series=bench_bt,
                                            metrics_weights=wts,
                                            risk_free_rate=risk_free_rate,
                                            rebalance_freq=rb,
                                            lookback_days=lookback_days,
                                            top_n=top_n
                                        )
                                    # Align on index
                                    if combined.empty:
                                        combined = eq_i[['Strategy']].rename(columns={'Strategy': name})
                                    else:
                                        combined = combined.join(eq_i[['Strategy']].rename(columns={'Strategy': name}), how='outer')
                                    # Collect metrics (Strategy row only)
                                    row = m_i.loc['Strategy']
                                    row.name = name
                                    metrics_list.append(row)

                                # Plot combined equity curves
                                st.subheader("Strategy Comparison")
                                comb_df = combined.sort_index().copy()
                                comb_plot_df = comb_df.reset_index()
                                if 'index' in comb_plot_df.columns:
                                    comb_plot_df = comb_plot_df.rename(columns={'index': 'date'})
                                elif comb_plot_df.columns[0] != 'date':
                                    comb_plot_df = comb_plot_df.rename(columns={comb_plot_df.columns[0]: 'date'})
                                comp_fig = px.line(comb_plot_df, x='date', y=compare_choices, title='Equity Curves: Multi-Strategy')
                                st.plotly_chart(comp_fig, use_container_width=True)

                                # Metrics table
                                if metrics_list:
                                    comp_metrics = pd.DataFrame(metrics_list)
                                    # Format select columns
                                    for col in ["CAGR", "Volatility", "Total Return", "Max Drawdown"]:
                                        if col in comp_metrics.columns:
                                            comp_metrics[col] = comp_metrics[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
                                    if 'Sharpe' in comp_metrics.columns:
                                        comp_metrics['Sharpe'] = comp_metrics['Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                                    st.dataframe(comp_metrics, use_container_width=True)
                                # Skip single-run section when comparison is shown
                                st.stop()
                            except Exception as e:
                                st.error(f"Comparison failed: {e}")

                        try:
                            if rebalance == 'Monthly':
                                rb = 'M'
                            elif rebalance == 'Quarterly':
                                rb = 'Q'
                            else:
                                rb = '6M'
                            # Support both new (3-return) and old (2-return) versions of run_backtest
                            try:
                                equity_df, metrics_df, holdings_df = run_backtest(
                                    nav_data=nav_bt,
                                    benchmark_series=bench_bt,
                                    metrics_weights=bt_weights,
                                    risk_free_rate=risk_free_rate,
                                    rebalance_freq=rb,
                                    lookback_days=lookback_days,
                                    top_n=top_n
                                )
                            except ValueError:
                                equity_df, metrics_df = run_backtest(
                                    nav_data=nav_bt,
                                    benchmark_series=bench_bt,
                                    metrics_weights=bt_weights,
                                    risk_free_rate=risk_free_rate,
                                    rebalance_freq=rb,
                                    lookback_days=lookback_days,
                                    top_n=top_n
                                )
                                import pandas as _pd
                                holdings_df = _pd.DataFrame(columns=['date','funds'])

                            # Plot equity curves
                            st.subheader("Equity Curve")
                            eq_df = equity_df.reset_index()
                            # Ensure x column is named 'date'
                            if 'index' in eq_df.columns:
                                eq_df = eq_df.rename(columns={'index': 'date'})
                            elif eq_df.columns[0] != 'date':
                                # Fallback: force first column to 'date'
                                eq_df = eq_df.rename(columns={eq_df.columns[0]: 'date'})
                            
                            eq_plot = px.line(
                                eq_df,
                                x='date',
                                y=['Strategy', 'Benchmark'],
                                labels={'date': 'Date', 'value': 'Equity'},
                                title='Strategy vs Benchmark'
                            )
                            st.plotly_chart(eq_plot, use_container_width=True)

                            # Show metrics
                            st.subheader("Backtest Metrics")
                            display_metrics = metrics_df.copy()
                            for col in ["CAGR", "Volatility", "Total Return", "Max Drawdown", "Alpha"]:
                                if col in display_metrics.columns:
                                    display_metrics[col] = display_metrics[col].apply(
                                        lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
                                    )
                            for col in ['Sharpe', 'Beta']:
                                if col in display_metrics.columns:
                                    display_metrics[col] = display_metrics[col].apply(
                                        lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
                                    )
                            st.dataframe(display_metrics, use_container_width=True)
                            
                            # Highlight key metrics
                            if 'Alpha' in metrics_df.columns and 'Beta' in metrics_df.columns:
                                st.markdown("### 📊 Strategy Performance vs Benchmark")
                                metric_cols = st.columns(4)
                                with metric_cols[0]:
                                    alpha_val = metrics_df.loc['Strategy', 'Alpha']
                                    st.metric("Alpha (Annual)", 
                                             f"{alpha_val*100:.2f}%" if pd.notnull(alpha_val) else "N/A",
                                             delta="Excess return" if alpha_val > 0 else "Underperformance")
                                with metric_cols[1]:
                                    beta_val = metrics_df.loc['Strategy', 'Beta']
                                    st.metric("Beta", 
                                             f"{beta_val:.3f}" if pd.notnull(beta_val) else "N/A",
                                             delta="Higher volatility" if beta_val > 1 else "Lower volatility")
                                with metric_cols[2]:
                                    sharpe_val = metrics_df.loc['Strategy', 'Sharpe']
                                    st.metric("Sharpe Ratio", 
                                             f"{sharpe_val:.3f}" if pd.notnull(sharpe_val) else "N/A")
                                with metric_cols[3]:
                                    cagr_val = metrics_df.loc['Strategy', 'CAGR']
                                    st.metric("CAGR", 
                                             f"{cagr_val*100:.2f}%" if pd.notnull(cagr_val) else "N/A")

                            # Downloads
                            equity_csv = equity_df.to_csv(index=True)
                            metrics_csv = metrics_df.to_csv(index=True)
                            st.download_button(
                                label="📥 Download Equity Curve (CSV)",
                                data=equity_csv,
                                file_name="backtest_equity.csv",
                                mime="text/csv",
                                key="dl_bt_equity"
                            )
                            st.download_button(
                                label="📥 Download Metrics (CSV)",
                                data=metrics_csv,
                                file_name="backtest_metrics.csv",
                                mime="text/csv",
                                key="dl_bt_metrics"
                            )

                            # Holdings per rebalance and first/last 6M lists
                            if 'holdings_df' in locals() and not holdings_df.empty:
                                st.subheader("Holdings Per Rebalance")
                                hd = holdings_df.copy()
                                hd['date'] = pd.to_datetime(hd['date']).dt.strftime('%Y-%m-%d')
                                st.dataframe(hd, use_container_width=True)

                                # First and last period holdings (6M label when applicable)
                                first_row = holdings_df.iloc[0]
                                last_row = holdings_df.iloc[-1]
                                first_label = "First 6 Months Holdings" if rb == '6M' else "First Period Holdings"
                                last_label = "Last 6 Months Holdings" if rb == '6M' else "Last Period Holdings"

                                def bullet_list(funds_str: str) -> str:
                                    items = [f.strip() for f in str(funds_str).split(',') if str(f).strip()]
                                    return "\n".join([f"- {it}" for it in items]) if items else "- (none)"

                                st.markdown(f"### {first_label} ({pd.to_datetime(first_row['date']).date()})")
                                st.markdown(bullet_list(first_row['funds']))

                                st.markdown(f"### {last_label} ({pd.to_datetime(last_row['date']).date()})")
                                st.markdown(bullet_list(last_row['funds']))

                                st.download_button(
                                    label="📥 Download Holdings Log (CSV)",
                                    data=holdings_df.to_csv(index=False),
                                    file_name="backtest_holdings.csv",
                                    mime="text/csv",
                                    key="dl_bt_holdings"
                                )
                        except Exception as e:
                            st.error(f"Backtest failed: {e}")

               
                # Documentation Tab
                with tab5:
                    st.header("📚 Documentation (Standard Definitions)")
                    st.markdown("Standard, widely used definitions and formulas that match this app's calculations.")

                    st.subheader("Glossary of Symbols")
                    st.markdown("- **r_t**: periodic return at time t (daily).  - **r_{b,t}**: benchmark return at time t.  - **r_f**: annual risk‑free rate.  - **P**: periods/year (252).  - **T**: number of periods.  - **V_t**: portfolio value at t.")

                    st.subheader("Annualized Return (CAGR)")
                    st.markdown("Constant annual growth rate implied by the observed returns.")
                    st.latex(r"\text{Annualized Return} = \left(\prod_{t=1}^{T} (1 + r_t)\right)^{\tfrac{P}{T}} - 1")

                    st.subheader("Annual Volatility")
                    st.markdown("Standard deviation of periodic returns, scaled to annual frequency.")
                    st.latex(r"\sigma_{\text{annual}} = \operatorname{stdev}(r)\,\sqrt{P}")

                    st.subheader("Sharpe Ratio")
                    st.markdown("Excess return per unit of total volatility, annualized.")
                    st.latex(r"\text{Sharpe} = \frac{\mathbb{E}(r - r_f/P)}{\operatorname{stdev}(r)}\,\sqrt{P}")

                    st.subheader("Sortino Ratio")
                    st.markdown("Excess return per unit of downside volatility (only negative returns), annualized.")
                    st.latex(r"\text{Sortino} = \frac{\mathbb{E}(r - r_f/P)}{\operatorname{stdev}(r\,|\, r<0)}\,\sqrt{P}")

                    st.subheader("Maximum Drawdown (MDD)")
                    st.markdown("Largest peak‑to‑trough decline in value over the period.")
                    st.latex(r"\text{MDD} = \min_t\left( \frac{V_t}{\max_{s\le t} V_s} - 1 \right)")

                    st.subheader("Information Ratio (IR)")
                    st.markdown("Average active return relative to its volatility (tracking error), annualized.")
                    st.latex(r"\text{IR} = \frac{\mathbb{E}(r - r_b)}{\operatorname{stdev}(r - r_b)}\,\sqrt{P}")

                    st.subheader("Tracking Error (TE)")
                    st.markdown("Standard deviation of active returns versus the benchmark, annualized.")
                    st.latex(r"\text{TE} = \operatorname{stdev}(r - r_b)\,\sqrt{P}")

                    st.subheader("Beta and Alpha (annualized)")
                    st.markdown("Beta: sensitivity to benchmark excess returns. Alpha: average excess return not explained by beta.")
                    st.latex(r"\beta = \frac{\operatorname{Cov}(r - r_f/P,\ r_b - r_f/P)}{\operatorname{Var}(r_b - r_f/P)}")
                    st.latex(r"\alpha_{\text{annual}} = \big(\mathbb{E}(r - r_f/P) - \beta\,\mathbb{E}(r_b - r_f/P)\big)\,P")

                    st.subheader("Composite Ranking")
                    st.markdown("Min–max normalize metrics to [0,1]; invert those where lower is better (volatility, drawdown, tracking error).")
                    st.latex(r"x^{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min} + 10^{-10}}\quad,\quad x^{\text{norm}}_{\text{lower better}} = 1 - x^{\text{norm}}")
                    st.markdown("Weighted sum yields the composite score used for ranking:")
                    st.latex(r"\text{Composite} = \sum_i |w_i|\, m_i^{\text{norm}}")

                    st.subheader("Preset Strategies (Conceptual)")
                    st.markdown("- Momentum: emphasize higher returns and Sharpe; penalize beta and drawdown.")
                    st.markdown("- Consistency: emphasize Sortino and Information Ratio; penalize drawdown and volatility.")
                    st.markdown("- Risk‑Adjusted: emphasize Sharpe and Sortino with drawdown control.")

                    st.subheader("Backtesting Overview")
                    st.markdown("Rebalance on a schedule, compute metrics over a lookback, rank by composite, hold Top‑N equally to next rebalance, compare equity curve vs benchmark, report CAGR/Vol/Sharpe/MDD/Total Return.")

                    st.subheader("Forecasting Overview")
                    st.markdown("- **Mean**: forecast monthly return as historical mean.")
                    st.markdown("- **AR(1)**: next return depends on last return (recursive multi‑step).")
                    st.latex(r"r_t = c + \phi\, r_{t-1} \;\Rightarrow\; \widehat{r}_{t+h} = c + \phi\, \widehat{r}_{t+h-1}")
                    st.markdown("- **Gradient Boosting**: learn nonlinear patterns from engineered features (lag1, rolling means/stds, drawdowns).")
                    st.markdown("- **LSTM (Long Short-Term Memory)**: Deep learning model with 2-layer architecture (64→32 LSTM units + Dense layers).")
                    st.markdown("  - Hyperparameter optimization: Early stopping (patience=10), learning rate reduction (ReduceLROnPlateau)")
                    st.markdown("  - Data normalization: Z-score standardization")
                    st.markdown("  - Lookback window: 12 months")
                    st.markdown("  - Train/validation split: 80/20")
                    st.markdown("  - Dropout layers (0.2, 0.2, 0.1) for regularization")
                    st.markdown("  - Optimizer: Adam (lr=0.001)")
                    st.markdown("  - Loss: Mean Squared Error (MSE)")

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
