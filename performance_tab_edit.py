with tab1:
    st.header("📊 Performance Metrics")
    ranking_criteria = st.selectbox(
        "Ranking Criteria:", 
        ["Composite Score", "Momentum", "Consistency", "Risk-Adjusted Returns"],
        key="performance_ranking"
    )
