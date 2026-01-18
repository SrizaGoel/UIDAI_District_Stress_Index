import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import traceback
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("Starting Aadhaar Dashboard...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print("=" * 80)

try:
    from src.data_loader import load_data
    from src.preprocess import preprocess
    from src.metrics import calculate_dsi
    from src.analysis import classify_stress
    from src.advanced_metrics import add_relative_stress, add_persistent_stress, calculate_performance_metrics
    from src.recommendations import recommend_action
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    st.error(f"Module import error: {e}")
    st.stop()

# PAGE CONFIG 
st.set_page_config(
    page_title="Aadhaar Operational Intelligence",
    layout="wide",
    page_icon="🆔",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    /* Clean professional background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        min-height: 100vh;
    }
    
    /* Main content */
    .main .block-container {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.8rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        color: #1e40af;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    
    /* Sub-section headers */
    .sub-section-header {
        font-size: 1.3rem;
        color: #374151;
        margin: 1.5rem 0 1rem 0;
        font-weight: 500;
        padding-left: 0.5rem;
        border-left: 4px solid #10b981;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Charts container */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        padding: 0.5rem;
        background: #f8fafc;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background: white;
        color: #4b5563;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-color: #3b82f6;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.2);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Filter info box */
    .filter-info {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    /* Warning message */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    /* Danger message */
    .danger-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(59, 130, 246, 0.3);
        border-radius: 50%;
        border-top-color: #3b82f6;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# header
st.markdown('<h1 class="main-header"> Aadhaar Operational Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Analytics for Efficient Aadhaar Service Delivery Across India</p>', unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_and_process_data():
    """Load and process data with caching"""
    try:
        with st.spinner("Loading and processing Aadhaar data. This may take a moment..."):
            print("\n" + "="*60)
            print("LOADING AND PROCESSING DATA")
            print("="*60)
            
            # raw data
            enrol, bio, demo = load_data()
            
            # Preprocess
            enrol, bio, demo = preprocess(enrol, bio, demo)
            
            
            df = calculate_dsi(enrol, bio, demo)
            
            # Analysis
            df = classify_stress(df)
            df = add_relative_stress(df)
            df = add_persistent_stress(df)
            df = calculate_performance_metrics(df)
            
            
            df["recommended_action"] = df.apply(recommend_action, axis=1)
            df["last_updated"] = datetime.now()
            
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            print(f"\n✓ Data processing complete: {len(df):,} records")
            print("="*60)
            
            return df, enrol, bio, demo
            
    except Exception as e:
        print(f"\n✗ Error in data processing: {str(e)}")
        print(traceback.format_exc())
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


df, enrol, bio, demo = load_and_process_data()


if df.empty:
    st.error("""
     No data loaded. Please check:
    1. CSV files are in the `data/` folder
    2. Files contain 'enrol', 'bio', or 'demo' in names
    3. Files have required columns: state, district, date
    """)
    st.stop()

# sidebar
st.sidebar.markdown("##  Navigation")

# Navigation
page = st.sidebar.radio(
    "Select Dashboard View",
    [" Overview", "Priority Alerts", "Action Center", "Analytics", "State View", "Data Info"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Data Filters")


all_states = sorted(df["state"].unique().tolist())
all_months = sorted(df["month"].unique().tolist())
all_stress_levels = sorted(df["stress_level"].unique().tolist())
all_priority_levels = sorted(df["priority_level"].unique().tolist())

# State filter 
st.sidebar.markdown("### Select States")
selected_states = st.sidebar.multiselect(
    "Choose states (select all or specific)",
    options=["All States"] + all_states,
    default=["All States"],
    key="state_filter",
    label_visibility="collapsed"
)

# Month filter
st.sidebar.markdown("### Select Time Period")
selected_months = st.sidebar.multiselect(
    "Choose months",
    options=["All Months"] + all_months,
    default=all_months[-3:] if len(all_months) >= 3 else ["All Months"],
    key="month_filter",
    label_visibility="collapsed"
)

# Stress level filter
st.sidebar.markdown("### Stress Levels")
selected_stress = st.sidebar.multiselect(
    "Filter by stress level",
    options=["All Levels"] + all_stress_levels,
    default=["All Levels"],
    key="stress_filter",
    label_visibility="collapsed"
)

# Priority level filter
st.sidebar.markdown("### Priority Levels")
selected_priority = st.sidebar.multiselect(
    "Filter by priority",
    options=["All Priorities"] + all_priority_levels,
    default=["All Priorities"],
    key="priority_filter",
    label_visibility="collapsed"
)

# Persistent stress filter
st.sidebar.markdown("### Persistent Stress")
show_persistent = st.sidebar.checkbox("Show only districts with persistent stress", False)


st.sidebar.markdown("---")
if st.sidebar.button(" Reset All Filters", use_container_width=True):
    st.rerun()


filtered_df = df.copy()


if "All States" not in selected_states and selected_states:
    filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]


if "All Months" not in selected_months and selected_months:
    filtered_df = filtered_df[filtered_df["month"].isin(selected_months)]


if "All Levels" not in selected_stress and selected_stress:
    filtered_df = filtered_df[filtered_df["stress_level"].isin(selected_stress)]


if "All Priorities" not in selected_priority and selected_priority:
    filtered_df = filtered_df[filtered_df["priority_level"].isin(selected_priority)]


if show_persistent:
    filtered_df = filtered_df[filtered_df["persistent_stress"] == True]

st.sidebar.markdown("---")


st.sidebar.markdown("Current View")
st.sidebar.metric("Total Records", f"{len(filtered_df):,}")
st.sidebar.metric("States", filtered_df["state"].nunique())
st.sidebar.metric("Districts", filtered_df["district"].nunique())


if len(filtered_df) < len(df):
    st.sidebar.success(f" Filters applied: Showing {len(filtered_df):,} of {len(df):,} records")
else:
    st.sidebar.info("Showing all data (no filters applied)")

current_df = filtered_df

# Dashboard Pages 
if page == "Overview":
    
    
    if len(current_df) < len(df):
        st.markdown(f"""
        <div class="filter-info">
        <strong> Filtered View:</strong> Showing {len(current_df):,} of {len(df):,} records | 
        States: {current_df['state'].nunique()} | Districts: {current_df['district'].nunique()}
        </div>
        """, unsafe_allow_html=True)
    
    st.header(" Executive Dashboard")
    
    if not current_df.empty:
        # Key Metrics
        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
        
        # row 1
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_districts = current_df["district"].nunique()
            st.metric("Total Districts", f"{total_districts:,}", 
                     help="Number of unique districts in current view")
        
        with col2:
            high_stress = len(current_df[current_df["stress_level"] == "HIGH"])
            high_stress_percent = (high_stress / len(current_df)) * 100 if len(current_df) > 0 else 0
            st.metric("High Stress Instances", f"{high_stress:,}", 
                     delta=f"{high_stress_percent:.1f}%",
                     help="District-month combinations with HIGH stress level")
        
        with col3:
            avg_stress = current_df["district_stress_index"].mean()
            st.metric("Avg Stress Index", f"{avg_stress:.3f}",
                     help="Average District Stress Index (0-5 scale)")
        
        with col4:
            critical = len(current_df[current_df["priority_level"] == "CRITICAL"])
            st.metric("Critical Priority", f"{critical:,}",
                     help="District-month combinations marked as CRITICAL priority")
        
        # row 2
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            total_demand = current_df["total_service_demand"].sum()
            st.metric("Total Service Demand", f"{total_demand:,.0f}",
                     help="Sum of biometric + demographic service requests")
        
        with col6:
            total_enrol = current_df["total_enrolment"].sum()
            st.metric("Total Enrollment", f"{total_enrol:,.0f}",
                     help="Total Aadhaar enrollment across districts")
        
        with col7:
            persistent = len(current_df[current_df["persistent_stress"] == True])
            st.metric("Persistent Stress", f"{persistent:,}",
                     help="Districts with 3+ consecutive months of HIGH stress")
        
        with col8:
            service_efficiency = current_df["service_efficiency"].mean() if "service_efficiency" in current_df.columns else 0
            st.metric("Service Efficiency", f"{service_efficiency:.2f}",
                     help="Average service efficiency score")
        
        
        st.markdown('<div class="section-header">Visual Analytics</div>', unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="sub-section-header">Stress Level Distribution</div>', unsafe_allow_html=True)
            stress_dist = current_df["stress_level"].value_counts()
            if not stress_dist.empty:
                fig = px.pie(
                    values=stress_dist.values,
                    names=stress_dist.index,
                    color=stress_dist.index,
                    color_discrete_map={
                        'HIGH': '#EF4444',
                        'MEDIUM': '#F59E0B', 
                        'LOW': '#10B981'
                    },
                    hole=0.4,
                    labels={'value': 'Count', 'names': 'Stress Level'}
                )
                fig.update_layout(
                    height=350,
                    showlegend=True,
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="sub-section-header">Monthly Stress Trend</div>', unsafe_allow_html=True)
            if "month" in current_df.columns and len(current_df["month"].unique()) > 1:
                monthly_trend = current_df.groupby("month")["district_stress_index"].agg(['mean', 'std']).reset_index()
                fig = px.line(
                    monthly_trend,
                    x="month",
                    y="mean",
                    error_y="std",
                    markers=True,
                    line_shape="spline",
                    labels={'mean': 'Avg Stress Index', 'month': 'Month'}
                )
                fig.update_layout(
                    height=350,
                    xaxis_title="Month",
                    yaxis_title="Average Stress Index",
                    hovermode="x unified",
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for trend analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        
        st.markdown('<div class="section-header">District Performance Analysis</div>', unsafe_allow_html=True)
        
        col_top1, col_top2 = st.columns(2)
        
        with col_top1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="sub-section-header">Top 10 High-Stress Districts</div>', unsafe_allow_html=True)
            if len(current_df) >= 10:
                top_districts = current_df.nlargest(10, "district_stress_index")[
                    ["state", "district", "month", "district_stress_index", 
                     "stress_level", "priority_level", "high_stress_months"]
                ].copy()
                top_districts["district_stress_index"] = top_districts["district_stress_index"].round(3)
                
                st.dataframe(
                    top_districts.style.background_gradient(
                        subset=['district_stress_index'], 
                        cmap='Reds'
                    ),
                    use_container_width=True,
                    height=350
                )
            else:
                st.info(f"Only {len(current_df)} records available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_top2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="sub-section-header">State-wise Performance</div>', unsafe_allow_html=True)
            if len(current_df) > 0:
                state_avg = current_df.groupby("state")["district_stress_index"].mean().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=state_avg.values,
                    y=state_avg.index,
                    orientation='h',
                    color=state_avg.values,
                    color_continuous_scale='RdYlGn_r',
                    labels={'x': 'Average Stress Index', 'y': 'State'}
                )
                fig.update_layout(
                    height=350,
                    xaxis_title="Average Stress Index",
                    yaxis_title="State",
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        
        st.markdown('<div class="section-header">Quick Insights</div>', unsafe_allow_html=True)
        
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        
        with col_insight1:
            if len(current_df) > 0:
                max_stress_district = current_df.loc[current_df["district_stress_index"].idxmax()]
                st.markdown(f"""
                <div class="warning-box">
                <strong> Highest Stress District:</strong><br>
                {max_stress_district['district']}, {max_stress_district['state']}<br>
                <small>Stress Index: {max_stress_district['district_stress_index']:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col_insight2:
            if len(current_df[current_df["emergency_flag"] == True]) > 0:
                emergency_count = len(current_df[current_df["emergency_flag"] == True])
                st.markdown(f"""
                <div class="danger-box">
                <strong> Emergency Districts:</strong><br>
                {emergency_count} districts need<br>immediate attention
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                <strong> No Emergencies:</strong><br>
                All districts are operating<br>within normal parameters
                </div>
                """, unsafe_allow_html=True)
        
        with col_insight3:
            if "trend" in current_df.columns:
                increasing = len(current_df[current_df["trend"] == "INCREASING"])
                if increasing > 0:
                    st.markdown(f"""
                    <div class="warning-box">
                    <strong> Increasing Stress:</strong><br>
                    {increasing} districts showing<br>worsening stress trends
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                    <strong> Stable Trends:</strong><br>
                    No districts showing<br>increasing stress trends
                    </div>
                    """, unsafe_allow_html=True)

elif page == " Priority Alerts":
    
    st.header(" Priority Alerts Dashboard")
    
    if not current_df.empty:
    
        st.markdown('<div class="section-header">Emergency Alerts</div>', unsafe_allow_html=True)
        
        emergency_df = current_df[current_df["emergency_flag"] == True]
        if not emergency_df.empty:
            st.markdown(f"""
            <div class="danger-box">
            <strong> EMERGENCY ALERT:</strong> {len(emergency_df)} districts require IMMEDIATE attention!
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Emergency Districts Details", expanded=True):
                emergency_details = emergency_df[
                    ["state", "district", "month", "district_stress_index",
                     "stress_level", "high_stress_months", "consecutive_high"]
                ].sort_values("district_stress_index", ascending=False)
                
                st.dataframe(
                    emergency_details.style.background_gradient(
                        subset=['district_stress_index'], 
                        cmap='Reds'
                    ),
                    use_container_width=True,
                    height=300
                )
                
                csv = emergency_details.to_csv(index=False)
                st.download_button(
                    label="Download Emergency Districts List",
                    data=csv,
                    file_name="emergency_districts.csv",
                    mime="text/csv"
                )
        else:
            st.markdown(f"""
            <div class="success-box">
             <strong>No Emergency Alerts</strong> - All districts are operating within normal parameters.
            </div>
            """, unsafe_allow_html=True)
        
        # Critical Priority Districts
        st.markdown('<div class="section-header">Critical Priority Districts</div>', unsafe_allow_html=True)
        
        critical_df = current_df[current_df["priority_level"] == "CRITICAL"]
        
        if not critical_df.empty:
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Critical Districts", len(critical_df))
            with col2:
                st.metric("Avg Stress Index", f"{critical_df['district_stress_index'].mean():.3f}")
            with col3:
                st.metric("Avg High Months", f"{critical_df['high_stress_months'].mean():.1f}")
            
            
            with st.expander(" View All Critical Districts", expanded=True):
                critical_details = critical_df[
                    ["state", "district", "month", "district_stress_index",
                     "stress_level", "high_stress_months", "consecutive_high",
                     "recommended_action"]
                ].sort_values("district_stress_index", ascending=False)
                
                st.dataframe(
                    critical_details,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "recommended_action": st.column_config.TextColumn(
                            "Action Plan",
                            width="large"
                        )
                    }
                )
        else:
            st.markdown(f"""
            <div class="success-box">
             <strong>No Critical Priority Districts</strong> in the current view.
            </div>
            """, unsafe_allow_html=True)
        
        #  Persistent Stress Analysis
        st.markdown('<div class="section-header">Persistent Stress Analysis</div>', unsafe_allow_html=True)
        
        persistent_df = current_df[current_df["persistent_stress"] == True]
        
        if not persistent_df.empty:
            col_pers1, col_pers2 = st.columns(2)
            
            with col_pers1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">Persistent Stress Districts</div>', unsafe_allow_html=True)
                

                unique_persistent = persistent_df.groupby(["state", "district"]).agg({
                    "high_stress_months": "max",
                    "consecutive_high": "max"
                }).reset_index().sort_values("high_stress_months", ascending=False)
                
                st.dataframe(
                    unique_persistent,
                    use_container_width=True,
                    height=300
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_pers2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">State-wise Distribution</div>', unsafe_allow_html=True)
                
                state_counts = persistent_df.groupby("state").size().sort_values(ascending=False).head(10)
                fig = px.bar(
                    x=state_counts.values,
                    y=state_counts.index,
                    orientation='h',
                    color=state_counts.values,
                    color_continuous_scale='Reds',
                    labels={'x': 'Number of Instances', 'y': 'State'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
             <strong>No districts with persistent stress</strong> (3+ consecutive high stress months)
            </div>
            """, unsafe_allow_html=True)

elif page == " Action Center":
    
    st.header(" Action Center & Recommendations")
    
    if not current_df.empty:
    
        st.markdown('<div class="section-header">District-Specific Analysis</div>', unsafe_allow_html=True)
        
        col_sel1, col_sel2 = st.columns(2)
        
        with col_sel1:
            selected_state = st.selectbox(
                "Select State",
                options=sorted(current_df["state"].unique()),
                key="action_state"
            )
        
        state_df = current_df[current_df["state"] == selected_state]
        
        with col_sel2:
            selected_district = st.selectbox(
                "Select District", 
                options=sorted(state_df["district"].unique()),
                key="action_district"
            )
        
        
        district_data = state_df[state_df["district"] == selected_district]
        
        if not district_data.empty:
            # Get latest data for the district
            latest_data = district_data.sort_values("month").iloc[-1]
            
            st.markdown('<div class="section-header">District Performance Metrics</div>', unsafe_allow_html=True)
            
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Stress Index", f"{latest_data['district_stress_index']:.3f}")
            metrics_cols[1].metric("Stress Level", latest_data['stress_level'])
            metrics_cols[2].metric("Priority Level", latest_data['priority_level'])
            metrics_cols[3].metric("High Months", int(latest_data['high_stress_months']))
            
        
            if "consecutive_high" in latest_data:
                st.info(f"Consecutive high stress months: {int(latest_data['consecutive_high'])}")
            
            #  Recommendations
            st.markdown('<div class="section-header">Recommended Actions</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("**Action Plan for Immediate Implementation**")
            st.markdown(latest_data['recommended_action'], unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Historical Trend
            st.markdown('<div class="section-header">Historical Performance Trend</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            trend_df = district_data.sort_values("month")
            
            fig = go.Figure()
            
            # Add stress index line
            fig.add_trace(go.Scatter(
                x=trend_df["month"],
                y=trend_df["district_stress_index"],
                mode='lines+markers',
                name='Stress Index',
                line=dict(color='#EF4444', width=3),
                marker=dict(size=8)
            ))
            
        
            fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                         annotation_text="High Stress Threshold", 
                         annotation_position="bottom right")
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Stress Threshold", 
                         annotation_position="bottom right")
            
            fig.update_layout(
                title=f"{selected_district}, {selected_state} - Stress Index Trend",
                xaxis_title="Month",
                yaxis_title="District Stress Index",
                hovermode="x unified",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">All Districts - Action Summary</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
    
        summary_df = current_df[
            ["state", "district", "month", "stress_level", 
             "priority_level", "district_stress_index", "recommended_action"]
        ].sort_values(["state", "district", "month"])
        
        # Group by district to show latest status
        latest_summary = summary_df.sort_values("month").groupby(["state", "district"]).last().reset_index()
        
        st.dataframe(
            latest_summary,
            use_container_width=True,
            height=500,
            column_config={
                "recommended_action": st.column_config.TextColumn(
                    "Action Plan",
                    width="large"
                )
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Analytics":
    
    st.header("Advanced Analytics")
    
    if not current_df.empty and len(current_df) > 1:
        
        st.markdown('<div class="section-header">Metric Analysis</div>', unsafe_allow_html=True)
        
        metric_options = [
            "district_stress_index",
            "total_service_demand", 
            "total_enrolment",
            "service_efficiency",
            "biometric_penetration",
            "demographic_coverage"
        ]
        
        available_metrics = [m for m in metric_options if m in current_df.columns]
        
        if available_metrics:
            selected_metric = st.selectbox(
                "Select Metric to Analyze",
                options=available_metrics,
                index=0,
                help="Choose a metric to analyze trends and distributions"
            )
            
        
            st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
            
            col_ts1, col_ts2 = st.columns(2)
            
            with col_ts1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">Monthly Trend</div>', unsafe_allow_html=True)
                
                time_series = current_df.groupby("month")[selected_metric].agg(['mean', 'std', 'count']).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_series["month"],
                    y=time_series["mean"],
                    mode='lines+markers',
                    name='Mean',
                    line=dict(color='#3B82F6', width=3)
                ))
                
                
                fig.add_trace(go.Scatter(
                    x=time_series["month"].tolist() + time_series["month"].tolist()[::-1],
                    y=(time_series["mean"] + time_series["std"]).tolist() + 
                      (time_series["mean"] - time_series["std"]).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Standard Deviation',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f"Monthly Trend of {selected_metric.replace('_', ' ').title()}",
                    xaxis_title="Month",
                    yaxis_title=selected_metric.replace('_', ' ').title(),
                    hovermode="x unified",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_ts2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">Distribution Analysis</div>', unsafe_allow_html=True)
                
                fig_hist = px.histogram(
                    current_df,
                    x=selected_metric,
                    nbins=30,
                    marginal="box",
                    title=f"Distribution of {selected_metric.replace('_', ' ').title()}"
                )
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            
            st.markdown('<div class="section-header">State-wise Comparison</div>', unsafe_allow_html=True)
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">Top 15 States</div>', unsafe_allow_html=True)
                
                state_comparison = current_df.groupby("state")[selected_metric].agg(['mean', 'std']).reset_index()
                state_comparison = state_comparison.sort_values("mean", ascending=False).head(15)
                
                fig = px.bar(
                    state_comparison,
                    x="mean",
                    y="state",
                    orientation='h',
                    error_x="std",
                    color="mean",
                    color_continuous_scale="RdYlGn_r",
                    labels={'mean': f'Average {selected_metric.replace("_", " ").title()}', 'state': 'State'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_comp2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">Box Plot by Stress Level</div>', unsafe_allow_html=True)
                
                if "stress_level" in current_df.columns:
                    fig_box = px.box(
                        current_df,
                        x="stress_level",
                        y=selected_metric,
                        color="stress_level",
                        color_discrete_map={
                            'HIGH': '#EF4444',
                            'MEDIUM': '#F59E0B',
                            'LOW': '#10B981'
                        },
                        title=f"{selected_metric.replace('_', ' ').title()} by Stress Level"
                    )
                    fig_box.update_layout(height=400)
                    st.plotly_chart(fig_box, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Correlation Analysis
            st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            numeric_cols = ['district_stress_index', 'total_service_demand', 'total_enrolment', 'service_efficiency']
            numeric_cols = [col for col in numeric_cols if col in current_df.columns]
            
            if len(numeric_cols) > 1:
                corr_matrix = current_df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title="Correlation Matrix"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient numeric columns for correlation analysis")
            st.markdown('</div>', unsafe_allow_html=True)

elif page == " State View":
    
    st.header("State-wise Analysis")
    
    if not current_df.empty:
    
        st.markdown('<div class="section-header">Select State for Detailed Analysis</div>', unsafe_allow_html=True)
        
        selected_state_detail = st.selectbox(
            "Choose a state",
            options=sorted(current_df["state"].unique()),
            key="state_detail"
        )
        
        state_detail_df = current_df[current_df["state"] == selected_state_detail]
        
        if not state_detail_df.empty:
        
            st.markdown(f'<div class="section-header">{selected_state_detail} - State Overview</div>', unsafe_allow_html=True)
            
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                districts_count = state_detail_df["district"].nunique()
                st.metric("Total Districts", districts_count)
            
            with col_s2:
                avg_stress_state = state_detail_df["district_stress_index"].mean()
                st.metric("Avg Stress Index", f"{avg_stress_state:.3f}")
            
            with col_s3:
                high_stress_pct = (state_detail_df["stress_level"] == "HIGH").mean() * 100
                st.metric("High Stress %", f"{high_stress_pct:.1f}%")
            
            with col_s4:
                critical_count = (state_detail_df["priority_level"] == "CRITICAL").sum()
                st.metric("Critical Instances", critical_count)
            
            # Top Districts In State
            st.markdown(f'<div class="section-header">Top Districts in {selected_state_detail}</div>', unsafe_allow_html=True)
            
            col_top1, col_top2 = st.columns(2)
            
            with col_top1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">Top 5 High-Stress Districts</div>', unsafe_allow_html=True)
                
                top_state_districts = state_detail_df.nlargest(
                    5, 
                    "district_stress_index"
                )[["district", "month", "district_stress_index", "stress_level", "priority_level"]]
                
                st.dataframe(
                    top_state_districts.style.background_gradient(
                        subset=['district_stress_index'],
                        cmap='Reds'
                    ),
                    use_container_width=True,
                    height=250
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_top2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="sub-section-header">Stress Level Distribution</div>', unsafe_allow_html=True)
                
                stress_dist_state = state_detail_df["stress_level"].value_counts()
                
                fig = px.pie(
                    values=stress_dist_state.values,
                    names=stress_dist_state.index,
                    color=stress_dist_state.index,
                    color_discrete_map={'HIGH': '#EF4444', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'},
                    hole=0.3
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Trend Analysis
            st.markdown(f'<div class="section-header">Trend Analysis for {selected_state_detail}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            state_monthly = state_detail_df.groupby("month").agg({
                "district_stress_index": "mean",
                "total_service_demand": "sum",
                "total_enrolment": "sum"
            }).reset_index()
            
            fig_state = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Average Stress Index", "Service Demand vs Enrollment"),
                vertical_spacing=0.15
            )
            
            # Stress Index
            fig_state.add_trace(
                go.Scatter(
                    x=state_monthly["month"],
                    y=state_monthly["district_stress_index"],
                    mode='lines+markers',
                    name='Stress Index',
                    line=dict(color='#EF4444', width=2)
                ),
                row=1, col=1
            )
            
            # Service Demand and Enrollment 
            fig_state.add_trace(
                go.Bar(
                    x=state_monthly["month"],
                    y=state_monthly["total_service_demand"],
                    name='Service Demand',
                    marker_color='#3B82F6',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig_state.add_trace(
                go.Scatter(
                    x=state_monthly["month"],
                    y=state_monthly["total_enrolment"],
                    mode='lines+markers',
                    name='Enrollment',
                    line=dict(color='#10B981', width=2),
                    yaxis='y2'
                ),
                row=2, col=1
            )
            
            fig_state.update_layout(
                height=600,
                showlegend=True,
                hovermode="x unified"
            )
            
            fig_state.update_yaxes(title_text="Service Demand", row=2, col=1)
            fig_state.update_yaxes(title_text="Enrollment", row=2, col=1, secondary_y=True)
            
            st.plotly_chart(fig_state, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        
        st.markdown('<div class="section-header">All States Summary</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        state_summary = current_df.groupby("state").agg({
            "district": "nunique",
            "district_stress_index": "mean",
            "total_service_demand": "sum",
            "total_enrolment": "sum"
        }).reset_index()
        
        state_summary = state_summary.round({
            "district_stress_index": 3
        })
        
        st.dataframe(
            state_summary.sort_values("district_stress_index", ascending=False),
            use_container_width=True,
            height=400
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Data Info":
    
    st.header("Data Information & Configuration")
    

    st.markdown('<div class="section-header">Data Source Information</div>', unsafe_allow_html=True)
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("Total Records", f"{len(current_df):,}")
        st.metric("Unique States", current_df["state"].nunique())
        st.metric("Unique Districts", current_df["district"].nunique())
    
    with col_info2:
        st.metric("Time Period", f"{current_df['month'].min()} to {current_df['month'].max()}")
        st.metric("Data Categories", "3")
        st.metric("Last Updated", current_df["last_updated"].iloc[0].strftime("%Y-%m-%d %H:%M"))
    
    with col_info3:
        high_stress_pct = (current_df["stress_level"] == "HIGH").mean() * 100
        st.metric("High Stress %", f"{high_stress_pct:.1f}%")
        critical_pct = (current_df["priority_level"] == "CRITICAL").mean() * 100
        st.metric("Critical %", f"{critical_pct:.1f}%")
        persistent_pct = (current_df["persistent_stress"] == True).mean() * 100
        st.metric("Persistent %", f"{persistent_pct:.1f}%")
    
    
    st.markdown('<div class="section-header">Data Quality Metrics</div>', unsafe_allow_html=True)
    
    col_quality1, col_quality2, col_quality3, col_quality4 = st.columns(4)
    
    with col_quality1:
        missing_values = current_df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    
    with col_quality2:
        duplicate_rows = current_df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_rows)
    
    with col_quality3:
        memory_usage = current_df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    with col_quality4:
        column_count = len(current_df.columns)
        st.metric("Columns", column_count)
    
    
    st.markdown('<div class="section-header">Data Export</div>', unsafe_allow_html=True)
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        csv_data = current_df.to_csv(index=False)
        st.download_button(
            label=" Download Current View",
            data=csv_data,
            file_name="aadhaar_current_view.csv",
            mime="text/csv",
            help="Download all data in current view"
        )
    
    with col_export2:
        if len(current_df[current_df["priority_level"] == "CRITICAL"]) > 0:
            critical_data = current_df[current_df["priority_level"] == "CRITICAL"].to_csv(index=False)
            st.download_button(
                label="Download Critical Districts",
                data=critical_data,
                file_name="critical_districts.csv",
                mime="text/csv",
                help="Download only critical priority districts"
            )
    
    with col_export3:
        summary_data = current_df.groupby(["state", "district"]).agg({
            "district_stress_index": "mean",
            "stress_level": lambda x: (x == "HIGH").mean(),
            "high_stress_months": "max"
        }).reset_index().to_csv(index=False)
        
        st.download_button(
            label="Download Summary",
            data=summary_data,
            file_name="district_summary.csv",
            mime="text/csv",
            help="Download district-level summary statistics"
        )
    
    
    st.markdown('<div class="section-header">System Information</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    sys_info = f"""
    **Python Version:** {sys.version.split()[0]}
    
    **Pandas Version:** {pd.__version__}
    
    **Streamlit Version:** {st.__version__}
    
    **Current Filters Applied:** {len(current_df):,} of {len(df):,} records
    
    **Data Processing Time:** Cached (updates every 10 minutes)
    
    **Dashboard Last Refreshed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    st.code(sys_info)
    
    
    if st.button("Refresh Dashboard & Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4b5563; padding: 1.5rem; background: #f8fafc; border-radius: 8px;'>
    <p style='font-size: 1rem; margin-bottom: 0.5rem; font-weight: 500;'>© 2024 Aadhaar Operational Intelligence Dashboard</p>
    <p style='font-size: 0.9rem; opacity: 0.8;'>Empowering Efficient Aadhaar Service Delivery Across India</p>
    <p style='font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem;'>Data Source: Aadhaar Enrollment, Biometric & Demographic Records</p>
</div>
""", unsafe_allow_html=True)