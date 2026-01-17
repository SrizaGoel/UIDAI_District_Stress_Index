import pandas as pd
import numpy as np

def classify_stress(df):
    """Enhanced stress classification with multiple dimensions"""
    
    # Primary stress classification
    df["stress_level"] = "LOW"
    df.loc[df["district_stress_index"] > 0.6, "stress_level"] = "HIGH"
    df.loc[
        (df["district_stress_index"] > 0.3) &
        (df["district_stress_index"] <= 0.6),
        "stress_level"
    ] = "MEDIUM"
    
    # Add severity score (0-100)
    df["severity_score"] = df["district_stress_index"] * 20
    
    # Trend classification
    df["trend"] = "STABLE"
    df.loc[df["stress_change"] > 10, "trend"] = "INCREASING"
    df.loc[df["stress_change"] < -10, "trend"] = "DECREASING"
    
    # Risk assessment
    conditions = [
        (df["stress_level"] == "HIGH") & (df["trend"] == "INCREASING"),
        (df["stress_level"] == "HIGH") & (df["trend"] == "STABLE"),
        (df["stress_level"] == "MEDIUM") & (df["trend"] == "INCREASING"),
        (df["stress_level"] == "LOW") & (df["trend"] == "INCREASING")
    ]
    choices = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk"]
    df["risk_level"] = np.select(conditions, choices, default="No Risk")
    
    return df