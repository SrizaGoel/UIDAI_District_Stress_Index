import pandas as pd
import os
import glob
from typing import List, Tuple

def find_all_files(data_dir: str, keyword: str) -> List[str]:
    """Find all CSV files containing keyword in name"""
    pattern = os.path.join(data_dir, f"*{keyword}*.csv")
    files = glob.glob(pattern, recursive=False)
    
    if not files:
        # Try alternative keywords
        alt_keywords = {
            'enrol': ['enrollment', 'enrolment', 'enrol'],
            'bio': ['biometric', 'bio'],
            'demo': ['demographic', 'demo']
        }
        
        for alt_key in alt_keywords.get(keyword, [keyword]):
            pattern = os.path.join(data_dir, f"*{alt_key}*.csv")
            files = glob.glob(pattern, recursive=False)
            if files:
                break
    
    if not files:
        raise FileNotFoundError(f"No CSV files found for keyword: {keyword} in {data_dir}")
    
    print(f"Found {len(files)} files for keyword '{keyword}': {files}")
    return sorted(files)  # Sort for consistent ordering

def load_multiple_files(file_paths: List[str]) -> pd.DataFrame:
    """Load and combine multiple CSV files"""
    if not file_paths:
        raise ValueError("No file paths provided")
    
    print(f"Loading {len(file_paths)} files...")
    dataframes = []
    
    for i, file_path in enumerate(file_paths):
        print(f"  Loading file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"    Successfully loaded {len(df)} rows")
        except Exception as e:
            print(f"    Error loading {file_path}: {str(e)}")
            raise
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined total: {len(combined_df)} rows")
    return combined_df

def normalize_state_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize state names to handle duplicates like Odisha/Orissa"""
    state_mapping = {
        'orissa': 'odisha',
        'uttaranchal': 'uttarakhand',
        'madras': 'tamil nadu',
        'bombay': 'maharashtra',
        'mysore': 'karnataka',
        'calcutta': 'west bengal',
        'bangalore': 'karnataka',
        'hyderabad': 'telangana',
        'puducherry': 'pondicherry',
        'pondicherry': 'puducherry'
    }
    
    if 'state' in df.columns:
        # Convert to string and handle NaN
        df['state'] = df['state'].astype(str).str.lower().str.strip()
        df['state'] = df['state'].replace(state_mapping)
        df['state'] = df['state'].str.title()
    
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.strip().str.title()
    
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize column names"""
    # Create a copy to avoid warnings
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Common column name mappings
    column_mappings = {
        'state_name': 'state',
        'district_name': 'district',
        'date_of_record': 'date',
        'record_date': 'date',
        'enrollment_date': 'date',
        'enrolment_date': 'date',
        'age_0_5': 'age_0_5',
        'age_5_17': 'age_5_17',
        'age_18+': 'age_18_greater',
        'age_18_greater': 'age_18_greater',
        'age_18_above': 'age_18_greater',
        'bio_age_5_17': 'bio_age_5_17',
        'bio_age_17_': 'bio_age_17_',
        'bio_age_17+': 'bio_age_17_',
        'demo_age_5_17': 'demo_age_5_17',
        'demo_age_17_': 'demo_age_17_',
        'demo_age_17+': 'demo_age_17_'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    return df

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize all data files from multiple CSVs"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "data")
    
    # Debug: List files in data directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Data directory: {data_dir}")
    all_files = os.listdir(data_dir)
    print(f"All files in data directory ({len(all_files)}):")
    for f in sorted(all_files):
        print(f"  - {f}")
    
    try:
        # Load enrolment data from multiple files
        print("\n=== Loading Enrolment Data ===")
        enrol_files = find_all_files(data_dir, "enrol")
        enrol = load_multiple_files(enrol_files)
        enrol = clean_column_names(enrol)
        
        # Load biometric data from multiple files
        print("\n=== Loading Biometric Data ===")
        bio_files = find_all_files(data_dir, "bio")
        bio = load_multiple_files(bio_files)
        bio = clean_column_names(bio)
        
        # Load demographic data from multiple files
        print("\n=== Loading Demographic Data ===")
        demo_files = find_all_files(data_dir, "demo")
        demo = load_multiple_files(demo_files)
        demo = clean_column_names(demo)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        # List available CSV files
        csv_files = [f for f in all_files if f.lower().endswith('.csv')]
        print(f"\nAvailable CSV files ({len(csv_files)}):")
        for csv_file in sorted(csv_files):
            print(f"  - {csv_file}")
        raise
    
    # Convert date columns
    for df in [enrol, bio, demo]:
        if 'date' in df.columns:
            print(f"Converting date column for dataframe with {len(df)} rows")
            # Try multiple date formats
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            
            # Check for successful conversions
            null_dates = df['date'].isnull().sum()
            if null_dates > 0:
                print(f"  Warning: {null_dates} rows have invalid dates")
    
    # Normalize state names
    print("\n=== Normalizing State Names ===")
    enrol = normalize_state_names(enrol)
    bio = normalize_state_names(bio)
    demo = normalize_state_names(demo)
    
    # Summary
    print("\n=== Data Loading Summary ===")
    print(f"Enrolment data: {len(enrol):,} rows, columns: {list(enrol.columns)}")
    print(f"Biometric data: {len(bio):,} rows, columns: {list(bio.columns)}")
    print(f"Demographic data: {len(demo):,} rows, columns: {list(demo.columns)}")
    
    return enrol, bio, demo