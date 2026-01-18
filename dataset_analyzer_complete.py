# aadhaar_analysis.py
"""
STANDALONE Aadhaar Data Analysis Tool
Loads 12 CSV files and performs comprehensive analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# Import for advanced analysis
from scipy import stats
from sklearn.cluster import KMeans
from tabulate import tabulate

class AadhaarDataAnalyzer:
    """Standalone analyzer for Aadhaar CSV files"""
    
    def __init__(self, data_folder="data"):
        """
        Initialize analyzer with data folder
        
        Args:
            data_folder: Path to folder containing CSV files
        """
        self.data_folder = data_folder
        self.enrol_data = None
        self.bio_data = None
        self.demo_data = None
        self.merged_data = None
        
    def load_all_csv_files(self):
        """Load all 12 CSV files from data folder"""
        print("\n" + "="*80)
        print("LOADING AADHAAR CSV FILES")
        print("="*80)
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        
        if len(csv_files) == 0:
            print("❌ No CSV files found in data folder!")
            return False
        
        # Categorize files by type
        enrol_files = []
        bio_files = []
        demo_files = []
        
        for file in csv_files:
            filename = os.path.basename(file).lower()
            if 'enrol' in filename:
                enrol_files.append(file)
            elif 'bio' in filename:
                bio_files.append(file)
            elif 'demo' in filename:
                demo_files.append(file)
            else:
                print(f"⚠️  Unclassified file: {filename}")
        
        print(f"\n📊 File Classification:")
        print(f"   Enrollment files: {len(enrol_files)}")
        print(f"   Biometric files: {len(bio_files)}")
        print(f"   Demographic files: {len(demo_files)}")
        
        # Load and combine each category
        try:
            # Load enrollment data
            if enrol_files:
                enrol_dfs = []
                for file in enrol_files:
                    df = pd.read_csv(file)
                    df['file_type'] = 'enrollment'
                    enrol_dfs.append(df)
                self.enrol_data = pd.concat(enrol_dfs, ignore_index=True)
                print(f"✓ Loaded {len(self.enrol_data):,} enrollment records")
            
            # Load biometric data
            if bio_files:
                bio_dfs = []
                for file in bio_files:
                    df = pd.read_csv(file)
                    df['file_type'] = 'biometric'
                    bio_dfs.append(df)
                self.bio_data = pd.concat(bio_dfs, ignore_index=True)
                print(f"✓ Loaded {len(self.bio_data):,} biometric records")
            
            # Load demographic data
            if demo_files:
                demo_dfs = []
                for file in demo_files:
                    df = pd.read_csv(file)
                    df['file_type'] = 'demographic'
                    demo_dfs.append(df)
                self.demo_data = pd.concat(demo_dfs, ignore_index=True)
                print(f"✓ Loaded {len(self.demo_data):,} demographic records")
            
            print(f"\n✅ Total records loaded: {len(self.enrol_data) + len(self.bio_data) + len(self.demo_data):,}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess and clean the data"""
        print("\n" + "="*80)
        print("PREPROCESSING DATA")
        print("="*80)
        
        # Standardize column names (assuming common patterns)
                # Standardize column names (assuming common patterns)
        column_mapping = {
            # Enrollment columns - add age group columns
            'enrolment': 'total_enrolment',
            'enrollment': 'total_enrolment',
            'enrol': 'total_enrolment',
            'capacity': 'total_enrolment',
            'total': 'total_enrolment',
            'age_0_5': 'age_0_5',
            'age_5_17': 'age_5_17', 
            'age_18_greater': 'age_18_greater',
            'age_0-5': 'age_0_5',
            'age_5-17': 'age_5_17',
            'age_18+': 'age_18_greater',
            
            # Biometric columns
            'biometric': 'biometric_requests',
            'bio': 'biometric_requests',
            'biometric_update': 'biometric_requests',
            'biometric_correction': 'biometric_requests',
            'biometric_count': 'biometric_requests',
            'biometric_requests_count': 'biometric_requests',
            
            # Demographic columns
            'demographic': 'demographic_requests',
            'demo': 'demographic_requests',
            'demographic_update': 'demographic_requests',
            'demographic_correction': 'demographic_requests',
            'demographic_count': 'demographic_requests',
            'demographic_requests_count': 'demographic_requests',
            
            # Common columns
            'state_name': 'state',
            'state_code': 'state',
            'district_name': 'district',
            'district_code': 'district',
            'month_year': 'month',
            'date': 'month',
            'period': 'month',
            'pincode': 'pincode'
        }
        
        # Apply standardization to each dataset
        for df_name in ['enrol_data', 'bio_data', 'demo_data']:
            df = getattr(self, df_name)
            if df is not None:
                # Rename columns
                df.columns = [col.strip().lower() for col in df.columns]
                df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
                
                # Convert date columns
                date_columns = ['month', 'date', 'period', 'month_year']
                for col in date_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            # Extract month-year if it's a full date
                            if df[col].dt.day.notna().any():
                                df['month'] = df[col].dt.to_period('M')
                        except:
                            pass
                
                # Fill missing values
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(0)
                
                # Remove duplicates
                initial_count = len(df)
                df.drop_duplicates(inplace=True)
                removed = initial_count - len(df)
                if removed > 0:
                    print(f"   Removed {removed} duplicates from {df_name}")
                
                setattr(self, df_name, df)
        
        print("✓ Data preprocessing completed")
        return True
    
    def merge_and_analyze(self):
        """Merge datasets and perform analysis"""
        print("\n" + "="*80)
        print("MERGING DATASETS & PERFORMING ANALYSIS")
        print("="*80)
        
        try:
            # Create a unified dataset by aggregating at district-month level
            merged_dfs = []
            
            # Process enrollment data
            if self.enrol_data is not None:
                # Check what columns are available
                print("\n📊 Enrollment data columns:", self.enrol_data.columns.tolist())
                
                # Check if we have total_enrolment column or need to calculate from age groups
                if 'total_enrolment' in self.enrol_data.columns:
                    enrol_grouped = self.enrol_data.groupby(['state', 'district', 'month']).agg({
                        'total_enrolment': 'sum'
                    }).reset_index()
                else:
                    # Calculate total enrollment from age groups
                    print("   Calculating total enrollment from age groups...")
                    # Create a copy to avoid modifying original
                    enrol_temp = self.enrol_data.copy()
                    
                    # Convert age columns to numeric if they're strings
                    age_columns = ['age_0_5', 'age_5_17', 'age_18_greater']
                    for col in age_columns:
                        if col in enrol_temp.columns:
                            enrol_temp[col] = pd.to_numeric(enrol_temp[col], errors='coerce')
                    
                    # Calculate total enrollment (sum of all age groups)
                    enrol_temp['total_enrolment'] = 0
                    for col in age_columns:
                        if col in enrol_temp.columns:
                            enrol_temp['total_enrolment'] += enrol_temp[col].fillna(0)
                    
                    enrol_grouped = enrol_temp.groupby(['state', 'district', 'month']).agg({
                        'total_enrolment': 'sum'
                    }).reset_index()
                
                merged_dfs.append(enrol_grouped)
            
            # Process biometric data
            if self.bio_data is not None:
                print("📊 Biometric data columns:", self.bio_data.columns.tolist())
                
                # Check for different column names for biometric requests
                bio_cols = ['biometric_requests', 'biometric', 'bio', 'biometric_update', 
                           'biometric_correction', 'biometric_count']
                bio_col_found = None
                for col in bio_cols:
                    if col in self.bio_data.columns:
                        bio_col_found = col
                        break
                
                if bio_col_found:
                    bio_grouped = self.bio_data.groupby(['state', 'district', 'month']).agg({
                        bio_col_found: 'sum'
                    }).reset_index()
                    bio_grouped = bio_grouped.rename(columns={bio_col_found: 'biometric_requests'})
                else:
                    # If no specific column found, count records
                    print("   No biometric request column found, counting records...")
                    bio_grouped = self.bio_data.groupby(['state', 'district', 'month']).size().reset_index()
                    bio_grouped = bio_grouped.rename(columns={0: 'biometric_requests'})
                
                merged_dfs.append(bio_grouped)
            
            # Process demographic data
            if self.demo_data is not None:
                print("📊 Demographic data columns:", self.demo_data.columns.tolist())
                
                # Check for different column names for demographic requests
                demo_cols = ['demographic_requests', 'demographic', 'demo', 'demographic_update', 
                            'demographic_correction', 'demographic_count']
                demo_col_found = None
                for col in demo_cols:
                    if col in self.demo_data.columns:
                        demo_col_found = col
                        break
                
                if demo_col_found:
                    demo_grouped = self.demo_data.groupby(['state', 'district', 'month']).agg({
                        demo_col_found: 'sum'
                    }).reset_index()
                    demo_grouped = demo_grouped.rename(columns={demo_col_found: 'demographic_requests'})
                else:
                    # If no specific column found, count records
                    print("   No demographic request column found, counting records...")
                    demo_grouped = self.demo_data.groupby(['state', 'district', 'month']).size().reset_index()
                    demo_grouped = demo_grouped.rename(columns={0: 'demographic_requests'})
                
                merged_dfs.append(demo_grouped)
            
            # Merge all datasets
            if len(merged_dfs) > 0:
                self.merged_data = merged_dfs[0]
                for df in merged_dfs[1:]:
                    self.merged_data = pd.merge(self.merged_data, df, 
                                               on=['state', 'district', 'month'], 
                                               how='outer')
                
                # Fill missing values
                self.merged_data = self.merged_data.fillna(0)
                
                # Calculate District Stress Index
                self.merged_data['total_service_demand'] = (
                    self.merged_data.get('biometric_requests', 0) + 
                    self.merged_data.get('demographic_requests', 0)
                )
                
                # Avoid division by zero
                self.merged_data['total_enrolment_adj'] = self.merged_data.get('total_enrolment', 0)
                self.merged_data['total_enrolment_adj'] = self.merged_data['total_enrolment_adj'].replace(0, 1)
                
                self.merged_data['district_stress_index'] = (
                    self.merged_data['total_service_demand'] / self.merged_data['total_enrolment_adj']
                )
                
                print(f"\n✅ Created merged dataset with {len(self.merged_data):,} records")
                print(f"✅ Columns: {self.merged_data.columns.tolist()}")
                
                # Show sample data
                print(f"\n📋 Sample data (first 5 rows):")
                print(tabulate(self.merged_data.head(), headers="keys", tablefmt="grid"))
                
                return True
            else:
                print("❌ No data to merge")
                return False
                
        except Exception as e:
            print(f"❌ Error in merging: {e}")
            import traceback
            traceback.print_exc()
            return False
         
        
        
        
    def perform_comprehensive_analysis(self):
        """Perform all analysis as per requirements"""
        if self.merged_data is None or len(self.merged_data) == 0:
            print("❌ No data available for analysis")
            return
        
        print("\n" + "="*100)
        print("AADHAAR OPERATIONAL INTELLIGENCE - COMPREHENSIVE ANALYSIS")
        print("="*100)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # ==================== DATASET OVERVIEW ====================
        print("\n📊 DATASET OVERVIEW")
        print("-"*80)
        
        total_records = len(self.enrol_data) + len(self.bio_data) + len(self.demo_data)
        overview_data = [
            ["Total CSV Files Processed", "12 files"],
            ["Enrollment Records", f"{len(self.enrol_data):,}"],
            ["Biometric Records", f"{len(self.bio_data):,}"],
            ["Demographic Records", f"{len(self.demo_data):,}"],
            ["Total Records Processed", f"{total_records:,}"],
            ["Merged Analysis Records", f"{len(self.merged_data):,}"],
            ["Geographic Coverage", "All Indian states and union territories"],
            ["Unique States", f"{self.merged_data['state'].nunique():,}"],
            ["Unique Districts", f"{self.merged_data['district'].nunique():,}"],
            ["Time Period", f"{self.merged_data['month'].min()} to {self.merged_data['month'].max()}"],
        ]
        print(tabulate(overview_data, tablefmt="grid"))
        
        # ==================== UNIVARIATE ANALYSIS ====================
        print("\n📈 UNIVARIATE ANALYSIS")
        print("-"*80)
        
        # 1. District Stress Index Distribution
        print("\n1. DISTRICT STRESS INDEX DISTRIBUTION")
        stress_index = self.merged_data['district_stress_index']
        
        stats_table = [
            ["Mean", f"{stress_index.mean():.3f}"],
            ["Median", f"{stress_index.median():.3f}"],
            ["Standard Deviation", f"{stress_index.std():.3f}"],
            ["Minimum", f"{stress_index.min():.3f}"],
            ["Maximum", f"{stress_index.max():.3f}"],
            ["25th Percentile", f"{stress_index.quantile(0.25):.3f}"],
            ["75th Percentile", f"{stress_index.quantile(0.75):.3f}"],
            ["Skewness", f"{stress_index.skew():.3f}"],
            ["Kurtosis", f"{stress_index.kurtosis():.3f}"]
        ]
        print(tabulate(stats_table, headers=["Statistic", "Value"], tablefmt="grid"))
        
        # Distribution categories
        print("\n📊 Stress Level Distribution:")
        categories = []
        thresholds = [0.0, 0.3, 0.6, float('inf')]
        labels = ['Normal (≤ 0.3)', 'Moderate (0.3 - 0.6)', 'High (> 0.6)']
        
        for i in range(len(thresholds)-1):
            mask = (stress_index > thresholds[i]) & (stress_index <= thresholds[i+1])
            count = mask.sum()
            percentage = (count / len(stress_index)) * 100
            categories.append([labels[i], count, f"{percentage:.1f}%"])
        
        print(tabulate(categories, headers=["Category", "Count", "Percentage"], tablefmt="grid"))
        
        # Outlier analysis
        Q1 = stress_index.quantile(0.25)
        Q3 = stress_index.quantile(0.75)
        IQR = Q3 - Q1
        outliers = stress_index[(stress_index < (Q1 - 1.5 * IQR)) | (stress_index > (Q3 + 1.5 * IQR))]
        print(f"\n⚠️  Outliers: {len(outliers):,} ({len(outliers)/len(stress_index)*100:.1f}% of data)")
        
        # 2. Service Request Volume Analysis
        print("\n2. SERVICE REQUEST VOLUME ANALYSIS")
        
        if 'biometric_requests' in self.merged_data.columns:
            bio_stats = [
                ["Total Biometric Requests", f"{self.merged_data['biometric_requests'].sum():,}"],
                ["Average per District", f"{self.merged_data['biometric_requests'].mean():,.0f}"],
                ["Minimum", f"{self.merged_data['biometric_requests'].min():,.0f}"],
                ["Maximum", f"{self.merged_data['biometric_requests'].max():,.0f}"],
                ["Std Dev", f"{self.merged_data['biometric_requests'].std():,.0f}"]
            ]
            print("📱 Biometric Requests:")
            print(tabulate(bio_stats, tablefmt="grid"))
        
        if 'demographic_requests' in self.merged_data.columns:
            demo_stats = [
                ["Total Demographic Requests", f"{self.merged_data['demographic_requests'].sum():,}"],
                ["Average per District", f"{self.merged_data['demographic_requests'].mean():,.0f}"],
                ["Minimum", f"{self.merged_data['demographic_requests'].min():,.0f}"],
                ["Maximum", f"{self.merged_data['demographic_requests'].max():,.0f}"],
                ["Std Dev", f"{self.merged_data['demographic_requests'].std():,.0f}"]
            ]
            print("\n👥 Demographic Requests:")
            print(tabulate(demo_stats, tablefmt="grid"))
        
        # 3. Enrollment Capacity Analysis
        print("\n3. ENROLLMENT CAPACITY ANALYSIS")
        
        if 'total_enrolment' in self.merged_data.columns:
            enrol_stats = [
                ["Total Enrollment", f"{self.merged_data['total_enrolment'].sum():,}"],
                ["Median Capacity", f"{self.merged_data['total_enrolment'].median():,.0f}"],
                ["Mean Capacity", f"{self.merged_data['total_enrolment'].mean():,.0f}"],
                ["Capacity Range", f"{self.merged_data['total_enrolment'].min():,.0f} - {self.merged_data['total_enrolment'].max():,.0f}"]
            ]
            print(tabulate(enrol_stats, tablefmt="grid"))
        
        # ==================== BIVARIATE ANALYSIS ====================
        print("\n📊 BIVARIATE ANALYSIS")
        print("-"*80)
        
        # 1. Stress vs Enrollment Capacity
        print("\n1. STRESS vs ENROLLMENT CAPACITY")
        if 'total_enrolment' in self.merged_data.columns:
            correlation = self.merged_data['district_stress_index'].corr(self.merged_data['total_enrolment'])
            print(f"📊 Pearson Correlation Coefficient: {correlation:.3f}")
            
            # Regression analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                self.merged_data['total_enrolment'], 
                self.merged_data['district_stress_index']
            )
            
            reg_data = [
                ["R-squared", f"{r_value**2:.3f}"],
                ["Slope", f"{slope:.8f}"],
                ["P-value", f"{p_value:.6f}"],
                ["Variance Explained", f"{r_value**2*100:.1f}%"]
            ]
            print(tabulate(reg_data, tablefmt="grid"))
            
            if r_value**2 < 0.4:
                print("💡 Insight: Capacity alone doesn't determine stress; 66% due to other factors")
        
        # 2. Biometric vs Demographic Requests
        print("\n2. BIOMETRIC vs DEMOGRAPHIC REQUESTS")
        if all(col in self.merged_data.columns for col in ['biometric_requests', 'demographic_requests']):
            correlation = self.merged_data['biometric_requests'].corr(self.merged_data['demographic_requests'])
            print(f"📊 Pearson Correlation: {correlation:.3f}")
            
            if correlation > 0.7:
                print("🤝 Strong correlation: Districts with high biometric requests also have high demographic requests")
            
            # Simple cluster analysis
            try:
                from sklearn.cluster import KMeans
                X = self.merged_data[['biometric_requests', 'demographic_requests']].fillna(0).values
                
                if len(X) > 10:
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X)
                    
                    self.merged_data['cluster'] = clusters
                    cluster_summary = self.merged_data.groupby('cluster').agg({
                        'biometric_requests': 'mean',
                        'demographic_requests': 'mean',
                        'district': 'count'
                    }).round(0)
                    
                    cluster_summary.index = ['Low Demand', 'Moderate Demand', 'High Demand']
                    print("\n🎯 Cluster Analysis:")
                    print(tabulate(cluster_summary, headers="keys", tablefmt="grid"))
            except:
                pass
        
                # 3. Temporal Stress Evaluation
        print("\n3. TEMPORAL STRESS EVALUATION")
        
        if 'month' in self.merged_data.columns:
            # Ensure month is in proper format
            if self.merged_data['month'].dtype == 'object':
                self.merged_data['month_str'] = self.merged_data['month']
            else:
                self.merged_data['month_str'] = self.merged_data['month'].astype(str)
            
            monthly_stress = self.merged_data.groupby('month_str')['district_stress_index'].agg(['mean', 'std']).reset_index()
            
            print("📅 Monthly Stress Trend:")
            if len(monthly_stress) > 0:
                print(tabulate(monthly_stress.round(3), headers="keys", tablefmt="grid"))
            
            # Simple persistent stress calculation
            print("\n🔍 Persistent Stress Analysis:")
            
            # Reset index to ensure proper alignment
            self.merged_data = self.merged_data.reset_index(drop=True)
            
            # Mark high stress months
            self.merged_data['is_high_stress'] = (self.merged_data['district_stress_index'] > 0.6).astype(int)
            
            # Sort by state, district and month - IMPORTANT: Include state in sorting
            self.merged_data = self.merged_data.sort_values(['state', 'district', 'month_str'])
            
            # Create a unique identifier for district (state + district)
            self.merged_data['state_district'] = self.merged_data['state'] + '_' + self.merged_data['district']
            
            # Calculate consecutive high stress months - SIMPLIFIED VERSION
            # This avoids complex groupby issues
            self.merged_data = self.merged_data.reset_index(drop=True)
            
            # Initialize consecutive_high column
            self.merged_data['consecutive_high'] = 0
            
            # Track current district and streak
            current_district = None
            current_streak = 0
            
            for idx, row in self.merged_data.iterrows():
                district_key = row['state_district']
                is_high = row['is_high_stress']
                
                if district_key != current_district:
                    # New district, reset streak
                    current_district = district_key
                    current_streak = 0
                
                if is_high == 1:
                    current_streak += 1
                else:
                    current_streak = 0
                
                self.merged_data.at[idx, 'consecutive_high'] = current_streak
            
            # Count districts with persistent stress (3+ consecutive months)
            persistent_mask = self.merged_data['consecutive_high'] >= 3
            
            # Get unique districts with persistent stress
            if persistent_mask.any():
                persistent_districts = self.merged_data[persistent_mask][['state', 'district']].drop_duplicates()
                persistent_count = len(persistent_districts)
                persistent_instances = persistent_mask.sum()
                
                print(f"   Districts with 3+ consecutive high stress months: {persistent_count:,}")
                print(f"   Total instances of persistent stress: {persistent_instances:,}")
                
                # Show top districts with highest consecutive streaks
                if persistent_count > 0:
                    max_consecutive = self.merged_data.groupby(['state', 'district'])['consecutive_high'].max().reset_index()
                    top_persistent = max_consecutive[max_consecutive['consecutive_high'] >= 3].sort_values('consecutive_high', ascending=False).head(10)
                    
                    print(f"\n🔥 Top Districts with Longest Consecutive High Stress:")
                    print(tabulate(top_persistent, headers="keys", tablefmt="grid"))
            else:
                print(f"   No districts with persistent stress (3+ consecutive months)")
                persistent_count = 0
                persistent_instances = 0
            
            # Simple trend analysis (if we have at least 2 months)
            if self.merged_data['month_str'].nunique() >= 2:
                print("\n📈 Trend Analysis:")
                # Get earliest and latest months
                months_sorted = sorted(self.merged_data['month_str'].unique())
                if len(months_sorted) >= 2:
                    first_month = months_sorted[0]
                    last_month = months_sorted[-1]
                    
                    # Calculate average stress for first and last month
                    first_avg = self.merged_data[self.merged_data['month_str'] == first_month]['district_stress_index'].mean()
                    last_avg = self.merged_data[self.merged_data['month_str'] == last_month]['district_stress_index'].mean()
                    
                    change = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
                    direction = "increased" if change > 0 else "decreased"
                    
                    print(f"   From {first_month} to {last_month}:")
                    print(f"   Average stress {direction} by {abs(change):.1f}%")

        # ==================== TRIVARIATE ANALYSIS ====================
        print("\n📊 TRIVARIATE ANALYSIS")
        print("-"*80)
        
        # 1. State-wise Analysis
        print("\n1. STATE-WISE PERFORMANCE ANALYSIS")
        state_analysis = self.merged_data.groupby('state')['district_stress_index'].agg(['mean', 'std', 'count']).round(3)
        state_analysis = state_analysis.sort_values('mean', ascending=False)
        
        national_avg = self.merged_data['district_stress_index'].mean()
        
        print(f"\nNational Average Stress Index: {national_avg:.3f}")
        print(f"States Above Average: {len(state_analysis[state_analysis['mean'] > national_avg])}")
        print(f"States Below Average: {len(state_analysis[state_analysis['mean'] < national_avg])}")
        
        print("\n🔥 Top 5 High-Stress States:")
        print(tabulate(state_analysis.head(5), headers="keys", tablefmt="grid"))
        
        print("\n✅ Top 5 Low-Stress States:")
        print(tabulate(state_analysis.tail(5), headers="keys", tablefmt="grid"))
        
        # 2. Service Efficiency Calculation
        print("\n2. SERVICE EFFICIENCY SCORE")
        
        # Calculate service efficiency
        if all(col in self.merged_data.columns for col in ['total_enrolment', 'total_service_demand', 'district_stress_index']):
            self.merged_data['service_efficiency'] = (
                (self.merged_data['total_enrolment'] / (self.merged_data['total_service_demand'] + 1)) * 
                (1 - self.merged_data['district_stress_index'])
            )
            
            efficiency_stats = [
                ["Mean Efficiency", f"{self.merged_data['service_efficiency'].mean():.3f}"],
                ["Median Efficiency", f"{self.merged_data['service_efficiency'].median():.3f}"],
                ["Efficiency Range", f"{self.merged_data['service_efficiency'].min():.3f} - {self.merged_data['service_efficiency'].max():.3f}"]
            ]
            print(tabulate(efficiency_stats, tablefmt="grid"))
            
            # Categorize efficiency
            self.merged_data['efficiency_category'] = pd.cut(
                self.merged_data['service_efficiency'],
                bins=[-np.inf, 0.4, 0.7, np.inf],
                labels=['Low Efficiency', 'Medium Efficiency', 'High Efficiency']
            )
            
            efficiency_dist = self.merged_data['efficiency_category'].value_counts()
            print("\n🏆 Efficiency Distribution:")
            eff_data = []
            for category, count in efficiency_dist.items():
                percentage = (count / len(self.merged_data)) * 100
                eff_data.append([category, count, f"{percentage:.1f}%"])
            print(tabulate(eff_data, headers=["Category", "Count", "Percentage"], tablefmt="grid"))
        
                # 3. Priority Classification
               # 3. Priority Classification
        print("\n3. PRIORITY CLASSIFICATION MATRIX")
        
        # Classify stress levels
        self.merged_data['stress_level'] = pd.cut(
            self.merged_data['district_stress_index'],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Create priority classification
        def assign_priority(row):
            consecutive = row.get('consecutive_high', 0)
            
            if row['stress_level'] == 'HIGH' and consecutive >= 3:
                return 'CRITICAL'
            elif row['stress_level'] == 'HIGH':
                return 'HIGH'
            elif row['stress_level'] == 'MEDIUM' and consecutive >= 3:
                return 'HIGH'
            elif row['stress_level'] == 'MEDIUM':
                return 'WATCH'
            else:
                return 'NORMAL'
        
        self.merged_data['priority_level'] = self.merged_data.apply(assign_priority, axis=1)
        
        priority_dist = self.merged_data['priority_level'].value_counts()
        print("🚨 Priority Level Distribution:")
        priority_data = []
        for priority, count in priority_dist.items():
            percentage = (count / len(self.merged_data)) * 100
            priority_data.append([priority, count, f"{percentage:.1f}%"])
        print(tabulate(priority_data, headers=["Priority", "Count", "Percentage"], tablefmt="grid"))
        priority_dist = self.merged_data['priority_level'].value_counts()
        print("🚨 Priority Level Distribution:")
        priority_data = []
        for priority, count in priority_dist.items():
            percentage = (count / len(self.merged_data)) * 100
            priority_data.append([priority, count, f"{percentage:.1f}%"])
        print(tabulate(priority_data, headers=["Priority", "Count", "Percentage"], tablefmt="grid"))
        
        # ==================== STATISTICAL INSIGHTS ====================
        print("\n📊 STATISTICAL INSIGHTS")
        print("-"*80)
        
        # 1. Capacity Gap Analysis
        print("\n1. CAPACITY GAP ANALYSIS")
        
        if 'total_enrolment' in self.merged_data.columns and 'total_service_demand' in self.merged_data.columns:
            # Calculate optimal ratio (median demand per enrollment)
            optimal_ratio = (self.merged_data['total_service_demand'] / 
                           (self.merged_data['total_enrolment'].replace(0, 1))).median()
            
            self.merged_data['capacity_gap'] = (
                (self.merged_data['total_service_demand'] / optimal_ratio) - 
                self.merged_data['total_enrolment']
            )
            
            # Categorize capacity gaps
            self.merged_data['capacity_category'] = pd.cut(
                self.merged_data['capacity_gap'],
                bins=[-np.inf, -0.15, 0.15, np.inf],
                labels=['Surplus', 'Balanced', 'Deficit']
            )
            
            capacity_dist = self.merged_data['capacity_category'].value_counts()
            print("⚖️ Capacity Gap Distribution:")
            cap_data = []
            for category, count in capacity_dist.items():
                percentage = (count / len(self.merged_data)) * 100
                cap_data.append([category, count, f"{percentage:.1f}%"])
            print(tabulate(cap_data, headers=["Category", "Count", "Percentage"], tablefmt="grid"))
            
            # Resource analysis
            surplus_total = self.merged_data[self.merged_data['capacity_category'] == 'Surplus']['capacity_gap'].abs().sum()
            deficit_total = self.merged_data[self.merged_data['capacity_category'] == 'Deficit']['capacity_gap'].abs().sum()
            
            print(f"\n💡 Resource Analysis:")
            print(f"   Total surplus capacity: {surplus_total:,.0f}")
            print(f"   Total deficit capacity: {deficit_total:,.0f}")
            print(f"   Estimated additional citizens served: {surplus_total * 12:,.0f} annually")
        
        # 2. Top Districts Analysis
        print("\n2. TOP DISTRICTS ANALYSIS")
        
        # Top 10 high-stress districts
        top_high_stress = self.merged_data.nlargest(10, 'district_stress_index')[
            ['state', 'district', 'month', 'district_stress_index', 'stress_level', 'priority_level']
        ]
        
        print("🔥 Top 10 High-Stress Districts:")
        print(tabulate(top_high_stress.round(3), headers="keys", tablefmt="grid"))
        
        # Districts needing immediate attention
        critical_districts = self.merged_data[self.merged_data['priority_level'] == 'CRITICAL']
        if not critical_districts.empty:
            print(f"\n🔴 Critical Districts Needing Immediate Attention: {critical_districts['district'].nunique():,}")
            
            top_critical = critical_districts.nlargest(5, 'district_stress_index')[
                ['state', 'district', 'district_stress_index', 'consecutive_high']
            ]
            print("🚨 Top 5 Most Critical Districts:")
            print(tabulate(top_critical.round(3), headers="keys", tablefmt="grid"))
        
        # ==================== RECOMMENDATIONS ====================
        print("\n🎯 ACTIONABLE RECOMMENDATIONS")
        print("-"*80)
        
        recommendations = []
        
        # Based on stress distribution
        high_stress_pct = (self.merged_data['stress_level'] == 'HIGH').mean() * 100
        if high_stress_pct > 15:
            recommendations.append(
                f"🔴 **Immediate Intervention**: {high_stress_pct:.1f}% of districts have HIGH stress. "
                "Deploy emergency response teams to top 20 high-stress districts."
            )
        
        # Based on capacity gaps
        if 'capacity_category' in self.merged_data.columns:
            deficit_pct = (self.merged_data['capacity_category'] == 'Deficit').mean() * 100
            if deficit_pct > 10:
                recommendations.append(
                    f"🏗️ **Capacity Expansion**: {deficit_pct:.1f}% of districts have capacity deficits. "
                    "Prioritize new center establishment in urban clusters."
                )
        
        # Based on persistent stress
        if 'consecutive_high' in self.merged_data.columns:
            persistent_count = (self.merged_data['consecutive_high'] >= 3).sum()
            if persistent_count > 0:
                recommendations.append(
                    f"🔄 **Long-term Strategy**: {persistent_count:,} instances of persistent stress (3+ months). "
                    "Develop sustained intervention plans for chronic stress districts."
                )
        
        # General recommendations
        general_recommendations = [
            "📅 **Seasonal Planning**: Increase staffing by 20% during March-April peak season",
            "🔄 **Resource Mobility**: Establish 50 mobile units for flexible deployment",
            "📊 **Real-time Monitoring**: Implement dashboard for district supervisors",
            "🤝 **Stakeholder Coordination**: Monthly review meetings with state officials",
            "🎯 **Performance Targets**: Set quarterly improvement targets for high-stress districts"
        ]
        
        recommendations.extend(general_recommendations)
        
        # Print recommendations
        print("\nPriority Actions:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # ==================== FINAL SUMMARY ====================
        print("\n" + "="*100)
        print("ANALYSIS SUMMARY")
        print("="*100)
        
        summary_stats = [
            ["Total Districts Analyzed", f"{self.merged_data['district'].nunique():,}"],
            ["National Average Stress Index", f"{national_avg:.3f}"],
            ["High Stress Districts (>0.6)", f"{(self.merged_data['stress_level'] == 'HIGH').sum():,}"],
            ["Critical Priority Districts", f"{(self.merged_data['priority_level'] == 'CRITICAL').sum():,}"],
            ["Districts Needing Capacity Expansion", f"{(self.merged_data['capacity_category'] == 'Deficit').sum() if 'capacity_category' in self.merged_data.columns else 'N/A':,}"],
            ["Service Efficiency (Avg)", f"{self.merged_data['service_efficiency'].mean() if 'service_efficiency' in self.merged_data.columns else 'N/A':.3f}"],
            ["Total Service Demand", f"{self.merged_data['total_service_demand'].sum():,}"],
            ["Total Enrollment Capacity", f"{self.merged_data['total_enrolment'].sum():,}"]
        ]
        
        print(tabulate(summary_stats, tablefmt="grid"))
        print(f"\n✅ Analysis completed at {datetime.now().strftime('%H:%M:%S')}")
    
    def export_results(self, output_file="aadhaar_analysis_results.txt"):
        """Export analysis results to a text file"""
        import sys
        
        original_stdout = sys.stdout
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                sys.stdout = f
                self.perform_comprehensive_analysis()
            sys.stdout = original_stdout
            print(f"\n✅ Analysis results exported to: {output_file}")
        except Exception as e:
            sys.stdout = original_stdout
            print(f"\n❌ Error exporting results: {e}")
    
    def save_processed_data(self, output_file="processed_aadhaar_data.csv"):
        """Save processed data to CSV"""
        if self.merged_data is not None:
            self.merged_data.to_csv(output_file, index=False)
            print(f"\n✅ Processed data saved to: {output_file}")
            return True
        return False

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("\n" + "="*100)
    print("AADHAAR DATA ANALYSIS - STANDALONE TOOL")
    print("="*100)
    
    # Create analyzer instance
    analyzer = AadhaarDataAnalyzer(data_folder="data")
    
    # Step 1: Load CSV files
    print("\n📂 Step 1: Loading CSV files from 'data' folder...")
    if not analyzer.load_all_csv_files():
        print("❌ Failed to load CSV files. Please check the data folder.")
        return
    
    # Step 2: Preprocess data
    print("\n🔄 Step 2: Preprocessing data...")
    analyzer.preprocess_data()
    
    # Step 3: Merge and analyze
    print("\n📊 Step 3: Merging datasets and performing analysis...")
    if not analyzer.merge_and_analyze():
        print("❌ Failed to merge datasets.")
        return
    
    # Step 4: Perform comprehensive analysis
    print("\n🔍 Step 4: Running comprehensive analysis...")
    analyzer.perform_comprehensive_analysis()
    
    # Optional: Export results
    print("\n💾 Optional: Exporting results...")
    export_choice = input("Export results to file? (y/n): ").lower()
    if export_choice == 'y':
        analyzer.export_results()
    
    # Optional: Save processed data
    save_choice = input("Save processed data to CSV? (y/n): ").lower()
    if save_choice == 'y':
        analyzer.save_processed_data()
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print("\nKey Outputs Generated:")
    print("1. 📊 Dataset overview and statistics")
    print("2. 📈 Univariate analysis of stress distribution")
    print("3. 🤝 Bivariate correlations and relationships")
    print("4. 🗺️ State-wise performance analysis")
    print("5. 🎯 Priority classification and recommendations")
    print("6. 💡 Actionable insights for operational improvement")

# ==================== COMMAND LINE INTERFACE ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Aadhaar Data Analysis Tool')
    parser.add_argument('--data-folder', default='data', help='Path to folder containing CSV files')
    parser.add_argument('--export', action='store_true', help='Export results to file')
    parser.add_argument('--save-data', action='store_true', help='Save processed data to CSV')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = AadhaarDataAnalyzer(data_folder=args.data_folder)
    
    if analyzer.load_all_csv_files():
        analyzer.preprocess_data()
        if analyzer.merge_and_analyze():
            analyzer.perform_comprehensive_analysis()
            
            if args.export:
                analyzer.export_results()
            
            if args.save_data:
                analyzer.save_processed_data()
        else:
            print("❌ Failed to merge and analyze data")
    else:
        print("❌ Failed to load data files")