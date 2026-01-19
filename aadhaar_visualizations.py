# aadhaar_simple_visuals.py
"""
Simple Aadhaar Data Visualizations
Creates colorful graphs for Composite Metric and State-wise Stress Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_sample_data():
    """
    Create sample data based on your analysis results
    You can replace this with your actual data loading code
    """
    
    # Sample data for Composite Metric (Efficiency Distribution)
    np.random.seed(42)
    n_districts = 1000
    
    # Create sample efficiency scores (72.2% low, 22.8% medium, 5% high)
    low_eff = np.random.normal(0.25, 0.1, int(n_districts * 0.722))
    medium_eff = np.random.normal(0.55, 0.1, int(n_districts * 0.228))
    high_eff = np.random.normal(0.85, 0.1, int(n_districts * 0.05))
    
    efficiency_scores = np.concatenate([low_eff, medium_eff, high_eff])
    efficiency_scores = np.clip(efficiency_scores, 0, 1)  # Clip to 0-1 range
    
    # Create categories
    efficiency_categories = []
    for score in efficiency_scores:
        if score < 0.4:
            efficiency_categories.append('Low Efficiency')
        elif score <= 0.7:
            efficiency_categories.append('Medium Efficiency')
        else:
            efficiency_categories.append('High Efficiency')
    
    # Sample data for State-wise Stress Analysis
    states = ['Daman and Diu', 'Andaman & Nicobar Islands', 'Lakshadweep', 'Kerala',
              'Meghalaya', 'Uttar Pradesh', 'West Bengal', 'Maharashtra', 'Tamil Nadu',
              'Karnataka', 'Delhi', 'Gujarat', 'Rajasthan', 'Punjab', 'Haryana']
    
    # Stress values based on your analysis
    stress_values = [15.261, 10.455, 9.33, 8.389,
                     0.798, 0.879, 1.0, 2.5, 3.2,
                     2.8, 4.1, 1.8, 1.2, 1.5, 1.7]
    
    # Monthly data for temporal analysis
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    national_avg = [2.5, 2.7, 3.2, 3.5, 3.0, 2.8]  # National average over months
    high_stress_avg = [10.5, 11.2, 12.8, 13.5, 12.0, 11.5]  # High stress states average
    low_stress_avg = [0.9, 0.95, 1.1, 1.2, 1.0, 0.95]  # Low stress states average
    
    return {
        'efficiency_scores': efficiency_scores,
        'efficiency_categories': efficiency_categories,
        'states': states,
        'stress_values': stress_values,
        'months': months,
        'national_avg': national_avg,
        'high_stress_avg': high_stress_avg,
        'low_stress_avg': low_stress_avg
    }

def plot_composite_metric(data):
    """Plot Composite Metric Analysis - Service Efficiency Distribution"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Composite Metric Analysis: Service Efficiency Distribution', fontsize=16, fontweight='bold')
    
    # 1. Distribution Histogram with KDE
    ax1 = axes[0, 0]
    sns.histplot(data['efficiency_scores'], kde=True, bins=30, ax=ax1, color='skyblue')
    ax1.axvline(x=0.4, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Low/Medium Threshold')
    ax1.axvline(x=0.7, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Medium/High Threshold')
    ax1.set_xlabel('Service Efficiency Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Service Efficiency Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Category Distribution Pie Chart
    ax2 = axes[0, 1]
    categories = pd.Series(data['efficiency_categories']).value_counts()
    
    colors = ['#FF6B6B', '#FFD166', '#06D6A0']  # Red, Yellow, Green
    wedges, texts, autotexts = ax2.pie(categories.values, labels=categories.index,
                                      autopct='%1.1f%%', colors=colors,
                                      startangle=90, explode=(0.05, 0.05, 0.05))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('Efficiency Categories Distribution', fontsize=14, fontweight='bold')
    ax2.legend(wedges, categories.index, title="Categories", loc="center left", 
               bbox_to_anchor=(1, 0, 0.5, 1))
    
    # 3. Box Plot by Category
    ax3 = axes[1, 0]
    df_box = pd.DataFrame({
        'Efficiency': data['efficiency_scores'],
        'Category': data['efficiency_categories']
    })
    
    sns.boxplot(x='Category', y='Efficiency', data=df_box, ax=ax3, palette=colors)
    sns.swarmplot(x='Category', y='Efficiency', data=df_box, ax=ax3, 
                  color='black', alpha=0.5, size=2)
    
    ax3.set_xlabel('Efficiency Category')
    ax3.set_ylabel('Service Efficiency Score')
    ax3.set_title('Service Efficiency by Category (Box Plot)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Violin Plot
    ax4 = axes[1, 1]
    sns.violinplot(x='Category', y='Efficiency', data=df_box, ax=ax4, palette=colors, inner='quartile')
    
    # Add text annotations
    for i, category in enumerate(['Low Efficiency', 'Medium Efficiency', 'High Efficiency']):
        cat_data = df_box[df_box['Category'] == category]['Efficiency']
        median_val = cat_data.median()
        ax4.text(i, median_val, f'Med: {median_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax4.set_xlabel('Efficiency Category')
    ax4.set_ylabel('Service Efficiency Score')
    ax4.set_title('Service Efficiency by Category (Violin Plot)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('composite_metric_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_trivariate_analysis(data):
    """Plot Trivariate Analysis: Stress × State × Time"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trivariate Analysis: Stress × State × Time', fontsize=16, fontweight='bold')
    
    # 1. State-wise Stress Bar Chart
    ax1 = axes[0, 0]
    
    # Sort states by stress value
    sorted_indices = np.argsort(data['stress_values'])
    sorted_states = [data['states'][i] for i in sorted_indices]
    sorted_stress = [data['stress_values'][i] for i in sorted_indices]
    
    # Color bars based on stress level
    colors = []
    for stress in sorted_stress:
        if stress > 5:
            colors.append('#FF6B6B')  # Red for high stress
        elif stress > 1.5:
            colors.append('#FFD166')  # Yellow for medium stress
        else:
            colors.append('#06D6A0')  # Green for low stress
    
    bars = ax1.barh(sorted_states, sorted_stress, color=colors, edgecolor='black')
    ax1.axvline(x=np.mean(data['stress_values']), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(data["stress_values"]):.2f}')
    
    # Add value labels
    for i, (bar, stress) in enumerate(zip(bars, sorted_stress)):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{stress:.2f}', va='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Average Stress Index')
    ax1.set_title('State-wise Average Stress Index', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Monthly Trends
    ax2 = axes[0, 1]
    
    # Plot multiple lines
    ax2.plot(data['months'], data['national_avg'], 'o-', linewidth=3, 
             markersize=8, label='National Average', color='blue')
    ax2.plot(data['months'], data['high_stress_avg'], 's--', linewidth=2, 
             markersize=6, label='High Stress States', color='red')
    ax2.plot(data['months'], data['low_stress_avg'], '^:', linewidth=2, 
             markersize=6, label='Low Stress States', color='green')
    
    # Fill between for high stress states (highlight tax season)
    if len(data['months']) >= 4:
        ax2.axvspan(data['months'][2], data['months'][3], alpha=0.2, color='orange',
                   label='Tax Season (Mar-Apr)')
    
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Stress Index')
    ax2.set_title('Monthly Stress Trends by State Category', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter Plot: State Performance
    ax3 = axes[1, 0]
    
    # Create some sample size data for bubbles
    state_sizes = np.random.randint(10, 100, len(data['states']))
    
    scatter = ax3.scatter(range(len(data['states'])), data['stress_values'],
                         s=state_sizes*20,  # Bubble size
                         c=data['stress_values'],
                         cmap='RdYlGn_r',  # Red-Yellow-Green reversed
                         edgecolors='black',
                         alpha=0.7)
    
    # Add state labels for top and bottom performers
    for i, state in enumerate(data['states']):
        if data['stress_values'][i] > 5 or data['stress_values'][i] < 1.2:
            ax3.text(i, data['stress_values'][i], state, fontsize=8,
                    ha='center', va='bottom' if data['stress_values'][i] > 5 else 'top',
                    fontweight='bold')
    
    ax3.set_xlabel('State Index')
    ax3.set_ylabel('Stress Index')
    ax3.set_title('State Performance Bubble Chart', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Stress Index')
    
    # 4. Heatmap-like grouped bar chart
    ax4 = axes[1, 1]
    
    # Create grouped data
    categories = ['Urban Dominant', 'Rural Dominant', 'Migration Heavy']
    months = data['months']
    
    # Sample data for different state types
    urban_data = [2.5, 2.8, 3.5, 4.0, 3.2, 2.9]
    rural_data = [1.5, 1.6, 1.7, 1.8, 1.6, 1.5]
    migration_data = [3.0, 3.2, 4.0, 4.5, 3.8, 3.3]
    
    x = np.arange(len(months))
    width = 0.25
    
    bars1 = ax4.bar(x - width, urban_data, width, label='Urban Dominant', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax4.bar(x, rural_data, width, label='Rural Dominant', 
                   color='#06D6A0', alpha=0.8)
    bars3 = ax4.bar(x + width, migration_data, width, label='Migration Heavy', 
                   color='#118AB2', alpha=0.8)
    
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Stress Index')
    ax4.set_title('Stress Patterns by State Type', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(months)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Highlight tax season
    if len(months) >= 4:
        ax4.axvspan(2.5, 3.5, alpha=0.1, color='orange')
        ax4.text(3, max(urban_data + rural_data + migration_data) * 0.9,
                'Tax Season\nPeak', ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('trivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_simple_dashboard():
    """Create a simple summary dashboard"""
    print("="*60)
    print("AADHAAR DATA VISUALIZATION DASHBOARD")
    print("="*60)
    
    # Load sample data
    print("\n📊 Loading data...")
    data = load_sample_data()
    
    print("\n📈 Creating Composite Metric Analysis...")
    plot_composite_metric(data)
    
    print("\n📊 Creating Trivariate Analysis...")
    plot_trivariate_analysis(data)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\n📁 Output files created:")
    print("   • composite_metric_analysis.png")
    print("   • trivariate_analysis.png")
    print("\n📊 Key Insights:")
    print("   1. Service Efficiency: 72.2% Low, 22.8% Medium, 5% High")
    print("   2. High Stress States: Daman and Diu (15.26), Andaman & Nicobar (10.46)")
    print("   3. Low Stress States: Meghalaya (0.80), Uttar Pradesh (0.88)")
    print("   4. Tax Season (Mar-Apr): 25-30% stress increase in urban states")
    print("="*60)

# Main execution
if __name__ == "__main__":
    create_simple_dashboard()