import pandas as pd
import numpy as np

def calculate_dsi(enrol, bio, demo):
    """Calculate District Stress Index with enhanced metrics"""
    
    print("Calculating District Stress Index...")
    
    # Aggregate data efficiently
    print("Aggregating enrolment data...")
    e = enrol.groupby(["state", "district", "month"], as_index=False)["total_enrolment"].sum()
    
    print("Aggregating biometric data...")
    b = bio.groupby(["state", "district", "month"], as_index=False)["total_biometric"].sum()
    
    print("Aggregating demographic data...")
    d = demo.groupby(["state", "district", "month"], as_index=False)["total_demographic"].sum()
    
    # Merge all data
    print("Merging data...")
    df = e.merge(b, on=["state", "district", "month"], how="left")
    df = df.merge(d, on=["state", "district", "month"], how="left")
    
    # Fill NaN values
    fill_cols = ["total_biometric", "total_demographic"]
    df[fill_cols] = df[fill_cols].fillna(0).astype(int)
    
    # Calculate primary stress index
    print("Calculating stress index...")
    df["district_stress_index"] = np.where(
        df["total_enrolment"] > 0,
        (df["total_biometric"] + df["total_demographic"]) / df["total_enrolment"],
        0
    )
    
    # Clip extreme values
    df["district_stress_index"] = df["district_stress_index"].clip(upper=5)
    
    # Calculate additional metrics
    print("Calculating additional metrics...")
    df["biometric_penetration"] = np.where(
        df["total_enrolment"] > 0,
        df["total_biometric"] / df["total_enrolment"],
        0
    )
    
    df["demographic_coverage"] = np.where(
        df["total_enrolment"] > 0,
        df["total_demographic"] / df["total_enrolment"],
        0
    )
    
    df["total_service_demand"] = df["total_biometric"] + df["total_demographic"]
    
    # Calculate month-over-month change
    print("Calculating trends...")
    df = df.sort_values(["state", "district", "month"])
    
    df["stress_change"] = df.groupby(["state", "district"])["district_stress_index"].pct_change() * 100
    df["stress_change"] = df["stress_change"].fillna(0)
    
    # Categorize by population size
    print("Categorizing by population...")
    if len(df) > 0:
        try:
            df["population_category"] = pd.qcut(
                df["total_enrolment"], 
                q=min(4, len(df)),
                labels=["Very Small", "Small", "Medium", "Large"][:min(4, len(df))]
            )
        except:
            df["population_category"] = "Medium"
    else:
        df["population_category"] = "Medium"
    
    print(f"DSI calculation complete: {len(df):,} district-month combinations")
    return df