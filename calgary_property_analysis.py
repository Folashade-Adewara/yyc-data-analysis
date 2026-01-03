"""
CALGARY PROPERTY ANALYSIS - COMPLETE DATASET
Comprehensive analysis of all Calgary property assessments
"""

print("=" * 70)
print("CALGARY PROPERTY ANALYSIS - COMPLETE DATASET")
print("=" * 70)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import os
from pathlib import Path
import time

# Start timing
start_time = time.time()

# Setup paths
current_dir = Path.cwd()
data_path = current_dir / "Current_Year_Property_Assessments_(Parcel)_20260102.csv"

# Create output directories
outputs_path = current_dir / "outputs"
viz_path = current_dir / "visualizations"
outputs_path.mkdir(exist_ok=True)
viz_path.mkdir(exist_ok=True)

print(f"Working directory: {current_dir}")
print(f"Data file: {data_path.name}")

if not data_path.exists():
    print("Error: Could not find the data file")
    print(f"Expected location: {data_path}")
    exit()

# Get file info
file_size_mb = data_path.stat().st_size / (1024 * 1024)
print(f"File size: {file_size_mb:.1f} MB")

# ============================================
# 1. LOAD THE COMPLETE DATASET
# ============================================
print("\nLoading complete dataset...")
print("(This might take a minute for the full 282MB file)")

# Load with only the columns we need to save memory
df = pd.read_csv(
    data_path,
    encoding='latin-1',
    low_memory=False,
    usecols=['COMM_NAME', 'ASSESSED_VALUE', 'PROPERTY_TYPE', 'ADDRESS']
)

load_time = time.time() - start_time
print(f"Loaded {len(df):,} properties in {load_time:.1f} seconds")
print(f"Columns loaded: {len(df.columns)}")

# ============================================
# 2. CLEAN AND PREPARE THE DATA
# ============================================
print("\nCleaning data...")

def clean_value(x):
    """Convert property values from strings like '42,500' to numeric"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        # Remove formatting characters
        x = x.replace(',', '').replace('$', '').replace(' ', '').strip()
        if x == '':
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

# Clean the assessed values
df['VALUE_CLEAN'] = df['ASSESSED_VALUE'].apply(clean_value)

# Remove rows with missing values
initial_count = len(df)
df_clean = df.dropna(subset=['VALUE_CLEAN', 'COMM_NAME'])
cleaned_count = len(df_clean)

print(f"Initial rows: {initial_count:,}")
print(f"After cleaning: {cleaned_count:,}")
print(f"Rows removed: {initial_count - cleaned_count:,}")

# ============================================
# 3. CITY-WIDE STATISTICS
# ============================================
print("\n" + "-" * 50)
print("CALGARY-WIDE PROPERTY STATISTICS")
print("-" * 50)

calgary_stats = df_clean['VALUE_CLEAN'].describe()
print(f"Total properties analyzed: {calgary_stats['count']:,.0f}")
print(f"Average property value: ${calgary_stats['mean']:,.2f}")
print(f"Median property value: ${calgary_stats['50%']:,.2f}")
print(f"Minimum value: ${calgary_stats['min']:,.2f}")
print(f"Maximum value: ${calgary_stats['max']:,.2f}")
print(f"Standard deviation: ${calgary_stats['std']:,.2f}")

# Value distribution percentiles
percentiles = df_clean['VALUE_CLEAN'].quantile([0.25, 0.75, 0.90, 0.95, 0.99])
print("\nProperty Value Percentiles:")
print(f"  25th percentile: ${percentiles[0.25]:,.2f}")
print(f"  75th percentile: ${percentiles[0.75]:,.2f}")
print(f"  90th percentile: ${percentiles[0.90]:,.2f}")
print(f"  95th percentile: ${percentiles[0.95]:,.2f}")
print(f"  99th percentile: ${percentiles[0.99]:,.2f}")

# ============================================
# 4. NEIGHBORHOOD ANALYSIS
# ============================================
print("\n" + "-" * 50)
print("NEIGHBORHOOD ANALYSIS")
print("-" * 50)

# Count neighborhoods
total_hoods = df_clean['COMM_NAME'].nunique()
print(f"Total neighborhoods in Calgary: {total_hoods:,}")

# Top neighborhoods by property count
print("\nTop 15 Neighborhoods (by number of properties):")
print("-" * 40)

top_hoods = df_clean['COMM_NAME'].value_counts().head(15)
for i, (hood, count) in enumerate(top_hoods.items(), 1):
    percentage = (count / len(df_clean)) * 100
    print(f"{i:2d}. {hood:30} {count:6,} properties ({percentage:.1f}%)")

# Most expensive neighborhoods
print("\nTop 10 Most Expensive Neighborhoods:")
print("-" * 40)

hood_stats = df_clean.groupby('COMM_NAME')['VALUE_CLEAN'].agg(['mean', 'count'])
top_expensive = hood_stats.sort_values('mean', ascending=False).head(10)

for i, (hood, row) in enumerate(top_expensive.iterrows(), 1):
    print(f"{i:2d}. {hood:30} ${row['mean']:,.2f} avg ({row['count']:,} properties)")

# ============================================
# 5. BELTLINE NEIGHBORHOOD FOCUS
# ============================================
print("\n" + "-" * 50)
print("BELTLINE NEIGHBORHOOD - DETAILED ANALYSIS")
print("-" * 50)

beltline_data = df_clean[df_clean['COMM_NAME'] == 'BELTLINE']

if len(beltline_data) > 0:
    beltline_stats = beltline_data['VALUE_CLEAN'].describe()
    
    print(f"Properties in Beltline: {beltline_stats['count']:,.0f}")
    print(f"Average value: ${beltline_stats['mean']:,.2f}")
    print(f"Median value: ${beltline_stats['50%']:,.2f}")
    print(f"Value range: ${beltline_stats['min']:,.2f} - ${beltline_stats['max']:,.2f}")
    
    # Beltline value distribution
    beltline_percentiles = beltline_data['VALUE_CLEAN'].quantile([0.25, 0.50, 0.75, 0.90, 0.95])
    print("\nBeltline Value Distribution:")
    print(f"  25th percentile: ${beltline_percentiles[0.25]:,.2f}")
    print(f"  50th (Median): ${beltline_percentiles[0.50]:,.2f}")
    print(f"  75th percentile: ${beltline_percentiles[0.75]:,.2f}")
    print(f"  90th percentile: ${beltline_percentiles[0.90]:,.2f}")
    print(f"  95th percentile: ${beltline_percentiles[0.95]:,.2f}")
    
    # Comparison with Calgary average
    beltline_vs_calgary = (beltline_stats['mean'] / calgary_stats['mean'] - 1) * 100
    print(f"\nComparison with Calgary Average:")
    print(f"  Calgary average: ${calgary_stats['mean']:,.2f}")
    print(f"  Beltline average: ${beltline_stats['mean']:,.2f}")
    print(f"  Difference: {beltline_vs_calgary:+.1f}%")
    
    if beltline_vs_calgary > 0:
        print(f"  Beltline properties are {beltline_vs_calgary:.1f}% more expensive than Calgary average")
    else:
        print(f"  Beltline properties are {abs(beltline_vs_calgary):.1f}% less expensive than Calgary average")
    
    # Property types in Beltline
    if 'PROPERTY_TYPE' in beltline_data.columns:
        print("\nProperty Types in Beltline:")
        beltline_types = beltline_data['PROPERTY_TYPE'].value_counts()
        for prop_type, count in beltline_types.items():
            percentage = (count / len(beltline_data)) * 100
            print(f"  {prop_type:15} {count:5,} properties ({percentage:.1f}%)")

# ============================================
# 6. CREATE VISUALIZATIONS
# ============================================
print("\nCreating visualizations...")

# Set visual style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Calgary Property Analysis - Complete Dataset', fontsize=20, fontweight='bold', y=0.98)

# Chart 1: Top neighborhoods by property count
ax1 = plt.subplot(2, 3, 1)
top_10_count = df_clean['COMM_NAME'].value_counts().head(10)
ax1.barh(range(len(top_10_count)), top_10_count.values, color='steelblue', edgecolor='black')
ax1.set_yticks(range(len(top_10_count)))
ax1.set_yticklabels(top_10_count.index)
ax1.set_xlabel('Number of Properties')
ax1.set_title('Most Populous Neighborhoods')
ax1.invert_yaxis()

# Chart 2: Most expensive neighborhoods
ax2 = plt.subplot(2, 3, 2)
top_10_value = top_expensive.head(10).sort_values('mean', ascending=True)
ax2.barh(range(len(top_10_value)), top_10_value['mean'] / 1000000, color='forestgreen', edgecolor='black')
ax2.set_yticks(range(len(top_10_value)))
ax2.set_yticklabels(top_10_value.index)
ax2.set_xlabel('Average Value ($ Millions)')
ax2.set_title('Most Expensive Neighborhoods')
ax2.invert_yaxis()

# Chart 3: Beltline value distribution
ax3 = plt.subplot(2, 3, 3)
if len(beltline_data) > 0:
    beltline_values = beltline_data['VALUE_CLEAN'] / 1000000
    ax3.hist(beltline_values, bins=50, alpha=0.7, color='darkorange', edgecolor='black', log=True)
    ax3.set_xlabel('Property Value ($ Millions)')
    ax3.set_ylabel('Count (log scale)')
    ax3.set_title('Beltline: Property Value Distribution')
    ax3.grid(True, alpha=0.3)

# Chart 4: Beltline vs Calgary comparison
ax4 = plt.subplot(2, 3, 4)
if len(beltline_data) > 0:
    comparison_data = [calgary_stats['mean'], beltline_stats['mean']]
    labels = ['Calgary Average', 'Beltline Average']
    colors = ['lightgray', 'darkorange']
    bars = ax4.bar(labels, np.array(comparison_data) / 1000000, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Average Value ($ Millions)')
    ax4.set_title('Beltline vs Calgary Average')
    
    # Add value labels
    for bar, val in zip(bars, comparison_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${val/1000000:.1f}M', ha='center', va='bottom', fontsize=10)

# Chart 5: Overall value distribution (sample)
ax5 = plt.subplot(2, 3, 5)
sample_size = min(10000, len(df_clean))
sample_data = df_clean.sample(n=sample_size, random_state=42)['VALUE_CLEAN'] / 1000000
ax5.hist(sample_data, bins=50, alpha=0.6, color='mediumpurple', edgecolor='black', log=True)
ax5.set_xlabel('Property Value ($ Millions)')
ax5.set_ylabel('Count (log scale)')
ax5.set_title(f'Calgary Property Values\n(Sample: {sample_size:,} properties)')

# Chart 6: Neighborhood size distribution
ax6 = plt.subplot(2, 3, 6)
hood_counts = df_clean['COMM_NAME'].value_counts()
ax6.hist(hood_counts.values, bins=30, alpha=0.6, color='crimson', edgecolor='black', log=True)
ax6.set_xlabel('Properties per Neighborhood')
ax6.set_ylabel('Number of Neighborhoods (log scale)')
ax6.set_title('Distribution of Neighborhood Sizes')
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save the visualization
viz_file = viz_path / "calgary_property_analysis.png"
plt.savefig(viz_file, dpi=150, bbox_inches='tight')
print(f"Visualization saved: {viz_file}")

# ============================================
# 7. SAVE ANALYSIS RESULTS
# ============================================
print("\nSaving analysis results...")

# Save dataset summary
summary_data = {
    'Dataset_Size_MB': [file_size_mb],
    'Total_Properties': [len(df)],
    'Valid_Properties': [len(df_clean)],
    'Calgary_Average': [calgary_stats['mean']],
    'Calgary_Median': [calgary_stats['50%']],
    'Calgary_Min': [calgary_stats['min']],
    'Calgary_Max': [calgary_stats['max']],
    'Total_Neighborhoods': [total_hoods],
    'Analysis_Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
}

summary_df = pd.DataFrame(summary_data)
summary_file = outputs_path / "dataset_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Dataset summary saved: {summary_file}")

plt.close(fig)  # Close the figure to save memory
# Save neighborhood statistics
hood_summary = df_clean.groupby('COMM_NAME')['VALUE_CLEAN'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).round(2)
hood_summary_file = outputs_path / "neighborhood_stats.csv"
hood_summary.to_csv(hood_summary_file)
print(f"Neighborhood statistics saved: {hood_summary_file}")

# Save Beltline detailed data
if len(beltline_data) > 0:
    beltline_detailed = beltline_data[['ADDRESS', 'ASSESSED_VALUE', 'PROPERTY_TYPE', 'VALUE_CLEAN']].copy()
    beltline_detailed_file = outputs_path / "beltline_properties.csv"
    beltline_detailed.to_csv(beltline_detailed_file, index=False)
    print(f"Beltline property data saved: {beltline_detailed_file}")

# Save property sample
sample_file = outputs_path / "property_sample.csv"
df_clean.sample(n=min(1000, len(df_clean)), random_state=42).to_csv(sample_file, index=False)
print(f"Property sample saved: {sample_file}")

# ============================================
# 8. ANALYSIS COMPLETE
# ============================================
total_time = time.time() - start_time
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nTotal processing time: {total_time:.1f} seconds")
print(f"Properties analyzed: {len(df_clean):,}")
print(f"Neighborhoods analyzed: {total_hoods:,}")
print(f"Calgary average property value: ${calgary_stats['mean']:,.2f}")

if len(beltline_data) > 0:
    print(f"\nBeltline Neighborhood Findings:")
    print(f"  • {beltline_stats['count']:,.0f} total properties")
    print(f"  • Average value: ${beltline_stats['mean']:,.2f}")
    print(f"  • {beltline_vs_calgary:+.1f}% compared to Calgary average")
    print(f"  • Value range: ${beltline_stats['min']:,.2f} to ${beltline_stats['max']:,.2f}")

print(f"\nOutput Files Created:")
print(f"  1. calgary_property_analysis.png - Visualization dashboard")
print(f"  2. dataset_summary.csv - Summary statistics")
print(f"  3. neighborhood_stats.csv - All neighborhood data")
print(f"  4. property_sample.csv - Sample of 1,000 properties")
if len(beltline_data) > 0:
    print(f"  5. beltline_properties.csv - Detailed Beltline data")

print(f"\n---")
print(f"This analysis covers all {len(df_clean):,} properties in Calgary.")
print(f"The results can be used for market analysis, investment research,")
print(f"or urban planning insights.")