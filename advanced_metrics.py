import pandas as pd
import numpy as np

def add_relative_stress(df):
    """Enhanced relative stress analysis"""
    
    # Calculate percentiles - handle NaN values
    df["relative_stress_percentile"] = df["district_stress_index"].rank(pct=True, na_option='keep')
    
    # Calculate state percentiles
    df["state_percentile"] = df.groupby("state")["district_stress_index"].rank(pct=True, na_option='keep')
    
    # Priority classification
    df["priority_level"] = "NORMAL"
    df.loc[df["relative_stress_percentile"] >= 0.9, "priority_level"] = "CRITICAL"
    df.loc[
        (df["relative_stress_percentile"] >= 0.6) &
        (df["relative_stress_percentile"] < 0.9),
        "priority_level"
    ] = "WATCH"
    
    # Emergency flag
    df["emergency_flag"] = (df["priority_level"] == "CRITICAL") & (df["stress_level"] == "HIGH")
    
    return df

def add_persistent_stress(df):
    """Enhanced persistent stress analysis"""
    
    # Count high stress months
    counts = (
        df[df["stress_level"] == "HIGH"]
        .groupby(["state", "district"])
        .size()
        .reset_index(name="high_stress_months")
    )
    
    # Merge with main dataframe
    df = df.merge(counts, on=["state", "district"], how="left")
    df["high_stress_months"] = df["high_stress_months"].fillna(0).astype(int)
    
    # Persistent stress classification
    df["persistent_stress"] = df["high_stress_months"] >= 3
    df["chronic_stress"] = df["high_stress_months"] >= 6
    
    # Calculate consecutive high stress months - FIXED VERSION
    df['is_high_stress'] = (df['stress_level'] == 'HIGH').astype(int)
    
    # Sort by state, district, and month for proper grouping
    df = df.sort_values(['state', 'district', 'month'])
    
    # Calculate consecutive high stress months
    def calculate_consecutive(group):
        group = group.copy()
        group['consecutive_high'] = (group['is_high_stress']
                                     .groupby((group['is_high_stress'] == 0).cumsum())
                                     .cumsum() * group['is_high_stress'])
        return group['consecutive_high']
    
    # Apply the calculation
    df['consecutive_high'] = df.groupby(['state', 'district']).apply(calculate_consecutive).reset_index(level=[0,1], drop=True)
    
    return df

def calculate_performance_metrics(df):
    """Calculate performance and efficiency metrics"""
    
    # Service efficiency - handle division by zero
    df["service_efficiency"] = np.where(
        df["total_enrolment"] > 0,
        df["total_service_demand"] / df["total_enrolment"],
        0
    )
    
    # Resource utilization score
    df["resource_utilization"] = df["district_stress_index"] * df.get("biometric_penetration", 0)
    
    # Capacity gap analysis
    df["capacity_gap"] = df["total_service_demand"] - df["total_enrolment"] * 0.3
    
    return df