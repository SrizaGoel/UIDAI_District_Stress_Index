import pandas as pd
import numpy as np

def preprocess(enrol, bio, demo):
    """Enhanced preprocessing with additional metrics"""
    
    print("Starting preprocessing...")
    
    # Enrollment calculations
    print("Processing enrolment data...")
    if 'age_0_5' in enrol.columns and 'age_5_17' in enrol.columns and 'age_18_greater' in enrol.columns:
        enrol["total_enrolment"] = (
            enrol["age_0_5"] +
            enrol["age_5_17"] +
            enrol["age_18_greater"]
        ).astype(int)
        
        # Handle division by zero safely
        mask = enrol["total_enrolment"] > 0
        enrol["child_percentage"] = 0.0
        enrol["adult_percentage"] = 0.0
        
        enrol.loc[mask, "child_percentage"] = (
            (enrol.loc[mask, "age_0_5"] + enrol.loc[mask, "age_5_17"]) / 
            enrol.loc[mask, "total_enrolment"]
        )
        
        enrol.loc[mask, "adult_percentage"] = (
            enrol.loc[mask, "age_18_greater"] / 
            enrol.loc[mask, "total_enrolment"]
        )
    else:
        print("Warning: Missing age columns in enrolment data")
        enrol["total_enrolment"] = 0
    
    # Biometric calculations
    print("Processing biometric data...")
    if 'bio_age_5_17' in bio.columns and 'bio_age_17_' in bio.columns:
        bio["total_biometric"] = (
            bio["bio_age_5_17"] +
            bio["bio_age_17_"]
        ).astype(int)
        
        # Fix: Handle division by zero for biometric_coverage
        total_biometric_sum = bio["total_biometric"].sum()
        if total_biometric_sum > 0:
            bio["biometric_coverage"] = bio["total_biometric"] / total_biometric_sum
        else:
            bio["biometric_coverage"] = 0.0
    else:
        print("Warning: Missing bio age columns")
        bio["total_biometric"] = 0
        bio["biometric_coverage"] = 0.0
    
    # Demographic calculations
    print("Processing demographic data...")
    if 'demo_age_5_17' in demo.columns and 'demo_age_17_' in demo.columns:
        demo["total_demographic"] = (
            demo["demo_age_5_17"] +
            demo["demo_age_17_"]
        ).astype(int)
    else:
        print("Warning: Missing demo age columns")
        demo["total_demographic"] = 0
    
    # Add month and year columns
    print("Adding temporal columns...")
    for df in [enrol, bio, demo]:
        if 'date' in df.columns:
            df["month"] = df["date"].dt.to_period("M").astype(str)
            df["year"] = df["date"].dt.year
            df["quarter"] = df["date"].dt.quarter
            df["month_year"] = df["date"].dt.strftime("%b %Y")
        
        if 'state' in df.columns:
            df["state"] = df["state"].str.strip().str.title()
        if 'district' in df.columns:
            df["district"] = df["district"].str.strip().str.title()
    
    # Calculate population density proxy
    print("Calculating population density scores...")
    max_enrolment = enrol["total_enrolment"].max()
    if max_enrolment > 0:
        enrol["population_density_score"] = enrol["total_enrolment"] / max_enrolment
    else:
        enrol["population_density_score"] = 0.0
    
    print(f"Preprocessing complete:")
    print(f"  - Enrolment: {len(enrol):,} rows")
    print(f"  - Biometric: {len(bio):,} rows")
    print(f"  - Demographic: {len(demo):,} rows")
    
    return enrol, bio, demo