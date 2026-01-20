# state_district_analysis.py
"""
State and District Analysis - Directly from CSV files
Reads actual CSV data and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# Set style for beautiful graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StateDistrictAnalyzer:
    """Analyze state and district data from CSV files"""
    
    def __init__(self, data_folder="data"):
        """Initialize with data folder path"""
        self.data_folder = data_folder
        self.merged_data = None
        
    def load_and_merge_data(self):
        """Load and merge data from all CSV files"""
        print("📊 Loading data from CSV files...")
        
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        
        if len(csv_files) == 0:
            print("❌ No CSV files found!")
            return False
        
        # Load all files
        all_data = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = os.path.basename(file)
                all_data.append(df)
                print(f"  ✓ Loaded {os.path.basename(file)}: {len(df):,} rows")
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
        
        if not all_data:
            print("❌ Failed to load any data")
            return False
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Combined data: {len(combined_data):,} total rows")
        
        # Standardize column names
        combined_data.columns = [col.strip().lower() for col in combined_data.columns]
        
        # Check available columns
        print("\n📋 Available columns:")
        for col in combined_data.columns:
            print(f"  • {col}")
        
        # Try to identify key columns
        state_cols = [col for col in combined_data.columns if 'state' in col]
        district_cols = [col for col in combined_data.columns if 'district' in col]
        month_cols = [col for col in combined_data.columns if 'month' in col or 'date' in col]
        
        print(f"\n🔍 Identified columns:")
        print(f"  State columns: {state_cols}")
        print(f"  District columns: {district_cols}")
        print(f"  Time columns: {month_cols}")
        
        # Use the first available column for each category
        state_col = state_cols[0] if state_cols else None
        district_col = district_cols[0] if district_cols else None
        month_col = month_cols[0] if month_cols else None
        
        if not state_col or not district_col:
            print("❌ Could not find state/district columns!")
            return False
        
        # Create a clean dataframe with key columns
        self.merged_data = combined_data[[state_col, district_col]].copy()
        self.merged_data.columns = ['state', 'district']
        
        # Find numeric columns for analysis
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  Numeric columns: {numeric_cols[:5]}...")  # Show first 5
        
        if numeric_cols:
            # Use the first numeric column as stress indicator
            stress_col = numeric_cols[0]
            self.merged_data['stress_index'] = combined_data[stress_col]
            print(f"  Using '{stress_col}' as stress indicator")
        
        # Fill missing values
        self.merged_data = self.merged_data.dropna(subset=['state', 'district'])
        
        # Clean state and district names
        self.merged_data['state'] = self.merged_data['state'].str.strip().str.title()
        self.merged_data['district'] = self.merged_data['district'].str.strip().str.title()
        
        # Remove "state" or similar suffixes
        self.merged_data['state'] = self.merged_data['state'].str.replace(' State$', '', regex=True)
        self.merged_data['state'] = self.merged_data['state'].str.replace(' STATE$', '', regex=True)
        
        print(f"\n✅ Final dataset: {len(self.merged_data):,} rows")
        print(f"   Unique states: {self.merged_data['state'].nunique()}")
        print(f"   Unique districts: {self.merged_data['district'].nunique()}")
        
        return True
    
    def plot_top_districts_within_states(self, num_states=5, districts_per_state=10):
        """Plot top stressed districts within states"""
        if self.merged_data is None or len(self.merged_data) == 0:
            print("❌ No data available for visualization")
            return
        
        print(f"\n📈 Creating Top Districts Within States visualization...")
        
        # Group by state and calculate average stress
        state_avg_stress = self.merged_data.groupby('state')['stress_index'].agg(['mean', 'count']).reset_index()
        state_avg_stress = state_avg_stress.sort_values('mean', ascending=False)
        
        # Get top N states
        top_states = state_avg_stress.head(num_states)['state'].tolist()
        
        # Create figure
        fig, axes = plt.subplots(num_states, 1, figsize=(14, 5*num_states))
        fig.suptitle(f'Top {districts_per_state} Stressed Districts within Top {num_states} States', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # If only one state, axes is not a list
        if num_states == 1:
            axes = [axes]
        
        for idx, state in enumerate(top_states):
            ax = axes[idx]
            
            # Get districts for this state
            state_data = self.merged_data[self.merged_data['state'] == state]
            
            # Calculate district averages
            district_avg = state_data.groupby('district')['stress_index'].agg(['mean', 'count']).reset_index()
            district_avg = district_avg.sort_values('mean', ascending=False).head(districts_per_state)
            
            # Color bars based on stress level
            colors = []
            for stress in district_avg['mean']:
                if stress > district_avg['mean'].quantile(0.8):
                    colors.append('#FF6B6B')  # Red for high stress
                elif stress > district_avg['mean'].quantile(0.5):
                    colors.append('#FFD166')  # Yellow for medium-high
                elif stress > district_avg['mean'].quantile(0.3):
                    colors.append('#4ECDC4')  # Teal for medium
                else:
                    colors.append('#06D6A0')  # Green for low
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(district_avg)), district_avg['mean'], 
                          color=colors, edgecolor='black', alpha=0.8)
            
            # Add value labels
            for i, (bar, row) in enumerate(zip(bars, district_avg.itertuples())):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{row.mean:.3f} ({row.count})', 
                       va='center', fontsize=9, fontweight='bold')
            
            # Set y-axis labels
            ax.set_yticks(range(len(district_avg)))
            ax.set_yticklabels(district_avg['district'])
            ax.invert_yaxis()  # Highest stress at top
            
            # Add state average line
            state_mean = state_data['stress_index'].mean()
            ax.axvline(x=state_mean, color='blue', linestyle='--', linewidth=2,
                      label=f'State Avg: {state_mean:.3f}')
            
            # Add national average line (if we have it)
            national_mean = self.merged_data['stress_index'].mean()
            ax.axvline(x=national_mean, color='red', linestyle=':', linewidth=2,
                      label=f'National Avg: {national_mean:.3f}')
            
            ax.set_xlabel('Average Stress Index')
            ax.set_title(f'{state} - Top {districts_per_state} Districts', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Set x-limit based on data
            x_max = district_avg['mean'].max() * 1.2
            ax.set_xlim(0, x_max)
        
        plt.tight_layout()
        plt.savefig('top_districts_within_states.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also create a summary table
        print(f"\n📋 Summary of Top Districts by State:")
        print("-" * 70)
        for state in top_states:
            state_data = self.merged_data[self.merged_data['state'] == state]
            top_district = state_data.loc[state_data['stress_index'].idxmax()]
            state_mean = state_data['stress_index'].mean()
            
            print(f"{state:25} | Avg: {state_mean:.3f} | Top District: {top_district['district']} ({top_district['stress_index']:.3f})")
        print("-" * 70)
    
    def plot_interstate_comparison(self, top_n=15):
        """Create interstate comparison visualizations"""
        if self.merged_data is None or len(self.merged_data) == 0:
            print("❌ No data available for visualization")
            return
        
        print(f"\n📊 Creating Interstate Comparison visualizations...")
        
        # Calculate state-level statistics
        state_stats = self.merged_data.groupby('state').agg({
            'stress_index': ['mean', 'std', 'count', 'min', 'max'],
            'district': 'nunique'
        }).round(3)
        
        # Flatten column names
        state_stats.columns = ['_'.join(col).strip() for col in state_stats.columns.values]
        state_stats = state_stats.reset_index()
        
        # Rename columns for clarity
        state_stats = state_stats.rename(columns={
            'stress_index_mean': 'avg_stress',
            'stress_index_std': 'std_stress',
            'stress_index_count': 'total_records',
            'stress_index_min': 'min_stress',
            'stress_index_max': 'max_stress',
            'district_nunique': 'num_districts'
        })
        
        # Sort by average stress
        state_stats = state_stats.sort_values('avg_stress', ascending=False)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Interstate Comparison Analysis', fontsize=18, fontweight='bold', y=1.02)
        
        # 1. Top N States by Average Stress (Bar Chart)
        ax1 = axes[0, 0]
        top_states = state_stats.head(top_n)
        
        # Color based on stress level
        colors_bar = []
        for stress in top_states['avg_stress']:
            if stress > state_stats['avg_stress'].quantile(0.8):
                colors_bar.append('#FF0000')  # Red for very high
            elif stress > state_stats['avg_stress'].quantile(0.6):
                colors_bar.append('#FF6B6B')  # Light red
            elif stress > state_stats['avg_stress'].quantile(0.4):
                colors_bar.append('#FFD166')  # Yellow
            elif stress > state_stats['avg_stress'].quantile(0.2):
                colors_bar.append('#4ECDC4')  # Teal
            else:
                colors_bar.append('#06D6A0')  # Green
        
        bars = ax1.barh(range(len(top_states)), top_states['avg_stress'],
                       color=colors_bar, edgecolor='black', alpha=0.8)
        
        # Add error bars for standard deviation
        ax1.errorbar(top_states['avg_stress'], range(len(top_states)),
                    xerr=top_states['std_stress'], fmt='none',
                    ecolor='black', capsize=3, alpha=0.7)
        
        # Add value labels
        for i, (bar, row) in enumerate(zip(bars, top_states.itertuples())):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{row.avg_stress:.3f} (±{row.std_stress:.3f})',
                    va='center', fontsize=9, fontweight='bold')
        
        ax1.set_yticks(range(len(top_states)))
        ax1.set_yticklabels(top_states['state'])
        ax1.invert_yaxis()
        ax1.axvline(x=state_stats['avg_stress'].mean(), color='blue', 
                   linestyle='--', linewidth=2, label='National Average')
        ax1.set_xlabel('Average Stress Index (± Std Dev)')
        ax1.set_title(f'Top {top_n} States by Stress Index', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Stress vs District Count (Scatter Plot)
        ax2 = axes[0, 1]
        
        scatter = ax2.scatter(state_stats['num_districts'], state_stats['avg_stress'],
                             s=state_stats['total_records']/100,  # Size by total records
                             c=state_stats['avg_stress'],
                             cmap='RdYlGn_r',
                             edgecolors='black',
                             alpha=0.7)
        
        # Add state labels for outliers
        for i, row in state_stats.iterrows():
            # Label states with very high or very low stress
            if (row['avg_stress'] > state_stats['avg_stress'].quantile(0.9) or 
                row['avg_stress'] < state_stats['avg_stress'].quantile(0.1)):
                ax2.annotate(row['state'],
                            xy=(row['num_districts'], row['avg_stress']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(state_stats['num_districts'], state_stats['avg_stress'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(state_stats['num_districts'].min(), 
                           state_stats['num_districts'].max(), 100)
        ax2.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend Line')
        
        # Calculate correlation
        correlation = np.corrcoef(state_stats['num_districts'], state_stats['avg_stress'])[0,1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Number of Districts')
        ax2.set_ylabel('Average Stress Index')
        ax2.set_title('Stress Index vs District Count', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Stress Index')
        
        # 3. State Performance Distribution (Box Plot)
        ax3 = axes[1, 0]
        
        # Prepare data for box plot
        box_data = []
        box_labels = []
        
        # Group states into categories based on stress level
        state_stats['stress_category'] = pd.qcut(state_stats['avg_stress'], 
                                                q=4, 
                                                labels=['Very Low', 'Low', 'High', 'Very High'])
        
        for category in ['Very Low', 'Low', 'High', 'Very High']:
            cat_data = state_stats[state_stats['stress_category'] == category]['avg_stress']
            if len(cat_data) > 0:
                box_data.append(cat_data)
                box_labels.append(category)
        
        box_colors = ['#06D6A0', '#4ECDC4', '#FFD166', '#FF6B6B']
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add swarm plot points
        for i, category in enumerate(box_labels):
            cat_data = box_data[i]
            # Add jitter to x positions
            x_pos = np.random.normal(i+1, 0.04, size=len(cat_data))
            ax3.scatter(x_pos, cat_data, alpha=0.6, color='black', s=30, edgecolors='white')
        
        ax3.set_ylabel('Average Stress Index')
        ax3.set_title('State Performance Distribution by Stress Category', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Heatmap: Top States vs Performance Metrics
        ax4 = axes[1, 1]
        
        # Select top 10 states for heatmap
        heatmap_data = state_stats.head(10)[['state', 'avg_stress', 'num_districts', 
                                           'max_stress', 'std_stress']].copy()
        
        # Normalize the data for better visualization
        for col in ['avg_stress', 'num_districts', 'max_stress', 'std_stress']:
            if heatmap_data[col].std() > 0:
                heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].mean()) / heatmap_data[col].std()
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.set_index('state').T
        
        im = ax4.imshow(heatmap_pivot.values, aspect='auto', cmap='RdYlGn_r', 
                       interpolation='nearest')
        
        # Set labels
        ax4.set_xticks(range(len(heatmap_pivot.columns)))
        ax4.set_xticklabels(heatmap_pivot.columns, rotation=45, ha='right')
        ax4.set_yticks(range(len(heatmap_pivot.index)))
        ax4.set_yticklabels(heatmap_pivot.index)
        
        # Add value annotations
        for i in range(len(heatmap_pivot.index)):
            for j in range(len(heatmap_pivot.columns)):
                value = heatmap_pivot.iloc[i, j]
                text_color = 'white' if abs(value) > 0.5 else 'black'
                ax4.text(j, i, f'{value:.1f}', ha='center', va='center', 
                        color=text_color, fontsize=8, fontweight='bold')
        
        ax4.set_title('State Performance Heatmap (Normalized)', 
                     fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, label='Normalized Value')
        
        plt.tight_layout()
        plt.savefig('interstate_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Interstate Comparison Summary:")
        print("-" * 70)
        print(f"Total States Analyzed: {len(state_stats)}")
        print(f"National Average Stress: {state_stats['avg_stress'].mean():.3f}")
        print(f"Highest Stress State: {state_stats.iloc[0]['state']} ({state_stats.iloc[0]['avg_stress']:.3f})")
        print(f"Lowest Stress State: {state_stats.iloc[-1]['state']} ({state_stats.iloc[-1]['avg_stress']:.3f})")
        print(f"Standard Deviation Across States: {state_stats['avg_stress'].std():.3f}")
        print("-" * 70)
        
        # Print top 5 states
        print(f"\n🏆 Top 5 Highest Stress States:")
        for i, row in state_stats.head(5).iterrows():
            print(f"  {i+1}. {row['state']:25} Avg Stress: {row['avg_stress']:.3f} "
                  f"(Districts: {row['num_districts']})")
        
        print(f"\n✅ Top 5 Lowest Stress States:")
        for i, row in state_stats.tail(5).iterrows():
            print(f"  {i+1}. {row['state']:25} Avg Stress: {row['avg_stress']:.3f} "
                  f"(Districts: {row['num_districts']})")
    
    def plot_interactive_state_selection(self):
        """Create visualization for any selected state"""
        if self.merged_data is None or len(self.merged_data) == 0:
            print("❌ No data available for visualization")
            return
        
        print(f"\n🎯 Creating State-Specific Analysis...")
        
        # Get unique states
        states = sorted(self.merged_data['state'].unique())
        
        # Let user select a state
        print(f"\nAvailable States ({len(states)} total):")
        for i, state in enumerate(states[:20]):  # Show first 20
            print(f"  {i+1:2}. {state}")
        
        if len(states) > 20:
            print(f"  ... and {len(states)-20} more")
        
        # Select a state (you can modify this to accept user input)
        selected_state = states[0]  # Default to first state
        print(f"\nSelected State for Analysis: {selected_state}")
        
        # Create visualization for selected state
        state_data = self.merged_data[self.merged_data['state'] == selected_state]
        
        if len(state_data) == 0:
            print(f"❌ No data available for {selected_state}")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Detailed Analysis: {selected_state}', 
                    fontsize=16, fontweight='bold', y=1.05)
        
        # 1. District Ranking within State
        ax1 = axes[0]
        
        # Calculate district statistics
        district_stats = state_data.groupby('district')['stress_index'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        district_stats = district_stats.sort_values('mean', ascending=False)
        
        # Take top 15 districts
        top_districts = district_stats.head(15)
        
        # Color bars
        colors = []
        state_mean = state_data['stress_index'].mean()
        for stress in top_districts['mean']:
            if stress > state_mean * 1.2:
                colors.append('#FF6B6B')  # Red - well above state avg
            elif stress > state_mean:
                colors.append('#FFD166')  # Yellow - above state avg
            elif stress > state_mean * 0.8:
                colors.append('#4ECDC4')  # Teal - near state avg
            else:
                colors.append('#06D6A0')  # Green - below state avg
        
        bars = ax1.barh(range(len(top_districts)), top_districts['mean'],
                       color=colors, edgecolor='black', alpha=0.8)
        
        # Add state average line
        ax1.axvline(x=state_mean, color='blue', linestyle='--', linewidth=2,
                   label=f'State Average: {state_mean:.3f}')
        
        # Add national average line
        national_mean = self.merged_data['stress_index'].mean()
        ax1.axvline(x=national_mean, color='red', linestyle=':', linewidth=2,
                   label=f'National Average: {national_mean:.3f}')
        
        # Add value labels
        for i, (bar, row) in enumerate(zip(bars, top_districts.itertuples())):
            deviation = (row.mean - state_mean) / state_mean * 100
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{row.mean:.3f} ({deviation:+.1f}%)', 
                    va='center', fontsize=9, fontweight='bold')
        
        ax1.set_yticks(range(len(top_districts)))
        ax1.set_yticklabels(top_districts['district'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Stress Index')
        ax1.set_title(f'Top 15 Districts in {selected_state}', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Comparison with Other States
        ax2 = axes[1]
        
        # Calculate state averages for comparison
        all_state_avg = self.merged_data.groupby('state')['stress_index'].mean().sort_values(ascending=False)
        
        # Find position of selected state
        selected_rank = list(all_state_avg.index).index(selected_state) + 1
        total_states = len(all_state_avg)
        
        # Create comparison bar chart
        compare_states = []
        
        # Add selected state
        compare_states.append(selected_state)
        
        # Add states above and below
        rank_index = selected_rank - 1
        for i in range(-2, 3):
            if i != 0 and 0 <= rank_index + i < total_states:
                compare_states.append(all_state_avg.index[rank_index + i])
        
        # Get data for comparison states
        compare_data = all_state_avg[compare_states]
        
        # Color bars
        compare_colors = []
        for state in compare_data.index:
            if state == selected_state:
                compare_colors.append('#118AB2')  # Blue for selected
            elif compare_data[state] > state_mean:
                compare_colors.append('#FF6B6B')  # Red for higher stress
            else:
                compare_colors.append('#06D6A0')  # Green for lower stress
        
        bars_compare = ax2.bar(range(len(compare_data)), compare_data.values,
                              color=compare_colors, edgecolor='black', alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars_compare, compare_data.values)):
            rank_pos = list(all_state_avg.index).index(compare_data.index[i]) + 1
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'#{rank_pos}\n{val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xticks(range(len(compare_data)))
        ax2.set_xticklabels(compare_data.index, rotation=45, ha='right')
        ax2.set_ylabel('Average Stress Index')
        ax2.set_title(f'{selected_state} vs Comparable States (Rank: {selected_rank}/{total_states})',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add rank annotation
        percentile = (total_states - selected_rank + 1) / total_states * 100
        ax2.text(0.95, 0.95, f'Rank: {selected_rank}/{total_states}\nTop {percentile:.1f}%',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'state_analysis_{selected_state.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed statistics
        print(f"\n📊 {selected_state} - Detailed Statistics:")
        print("-" * 60)
        print(f"Total Districts: {state_data['district'].nunique()}")
        print(f"Average Stress Index: {state_mean:.3f}")
        print(f"Highest District: {top_districts.iloc[0]['district']} ({top_districts.iloc[0]['mean']:.3f})")
        print(f"Lowest District: {district_stats.iloc[-1]['district']} ({district_stats.iloc[-1]['mean']:.3f})")
        print(f"State Rank: {selected_rank} out of {total_states} states")
        print(f"Performance: {'Above' if state_mean > national_mean else 'Below'} National Average")
        print("-" * 60)
    
    def run_all_analyses(self):
        """Run all visualization analyses"""
        print("="*70)
        print("STATE & DISTRICT ANALYSIS FROM CSV DATA")
        print("="*70)
        
        # Load data
        if not self.load_and_merge_data():
            return
        
        # Create visualizations
        self.plot_top_districts_within_states(num_states=5, districts_per_state=10)
        self.plot_interstate_comparison(top_n=15)
        self.plot_interactive_state_selection()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\n📁 Output files created:")
        print("  • top_districts_within_states.png - Top districts in top 5 states")
        print("  • interstate_comparison.png - State comparison analysis")
        print("  • state_analysis_[StateName].png - Detailed state analysis")

# Main execution
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = StateDistrictAnalyzer(data_folder="data")  # Change folder if needed
    
    # Run all analyses
    analyzer.run_all_analyses()