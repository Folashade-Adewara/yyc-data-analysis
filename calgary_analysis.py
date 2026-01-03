"""
CALGARY PROPERTY ANALYSIS - FIXED VERSION
Using correct column names from your data
"""

print("=" * 70)
print("🏠 CALGARY PROPERTY ANALYSIS - CORRECT COLUMNS")
print("=" * 70)

import pandas as pd
import os

print(f"✅ Pandas version: {pd.__version__}")

# Your data file
csv_file = "Current_Year_Property_Assessments_(Parcel)_20260102.csv"

if not os.path.exists(csv_file):
    print(f"❌ File not found: {csv_file}")
    exit()

print(f"✅ Found CSV: {csv_file}")
print(f"📊 File size: {os.path.getsize(csv_file)/(1024*1024):.1f} MB")

# Load more rows for better neighborhood coverage
print("\n📥 Loading first 50,000 rows...")
df = pd.read_csv(csv_file, low_memory=False)
print(f"✅ Loaded {len(df):,} rows")
print(f"📋 Columns: {len(df.columns)}")

# CORRECT COLUMNS FOR YOUR DATA:
COMMUNITY_COL = "COMM_NAME"           # Neighborhood column (found at index 11)
VALUE_COL = "ASSESSED_VALUE"          # Value column (found at index 4)

print(f"\n🎯 USING COLUMNS:")
print(f"🏘️  Neighborhood: '{COMMUNITY_COL}'")
print(f"💰 Value: '{VALUE_COL}'")

# Show sample neighborhoods
print(f"\n🏙️  NEIGHBORHOODS IN DATA (first 30):")
print("-" * 50)

# Clean and sort neighborhoods
hoods = df[COMMUNITY_COL].dropna().unique()
sorted_hoods = sorted([str(h).strip() for h in hoods])

for i, hood in enumerate(sorted_hoods[:30]):
    print(f"{i+1:2d}. {hood}")

print(f"\n📊 Total unique neighborhoods: {len(hoods)}")

# Count properties per neighborhood
print(f"\n📈 PROPERTIES PER NEIGHBORHOOD (top 10):")
hood_counts = df[COMMUNITY_COL].value_counts().head(10)
for hood, count in hood_counts.items():
    print(f"  {hood:30} - {count:,} properties")

# Ask for neighborhood
print("\n" + "=" * 60)
print("🎯 ENTER YOUR NEIGHBORHOOD")
print("=" * 60)
print("Examples from your data:")
print("  - COMMERCIAL CORE")
print("  - EAST FAIRVIEW INDUSTRIAL")
print("  - MISSION")
print("  - DOWNTOWN COMMERCIAL CORE")
print("  - BELTLINE")
print("-" * 60)

your_hood = input("\nYour neighborhood: ").strip().upper()

# Find matching neighborhood (case-insensitive)
matching_hoods = [h for h in hoods if str(h).strip().upper() == your_hood]

if not matching_hoods:
    print(f"\n❌ '{your_hood}' not found.")
    
    # Find similar
    print("\n🔍 Similar neighborhood names:")
    similar = [h for h in hoods if your_hood in str(h).upper()]
    for h in similar[:10]:
        print(f"  - {h}")
    
    # Show popular Calgary neighborhoods
    print("\n💡 Common Calgary neighborhoods in your data:")
    popular_in_data = ['BELTLINE', 'MISSION', 'COMMERCIAL', 'INDUSTRIAL', 'DOWNTOWN', 'RESIDENTIAL']
    found = []
    for h in hoods:
        h_upper = str(h).upper()
        for p in popular_in_data:
            if p in h_upper and h not in found:
                found.append(h)
                if len(found) >= 10:
                    break
        if len(found) >= 10:
            break
    
    for h in found:
        print(f"  - {h}")
    
    # Try again
    your_hood = input("\nChoose from above (type exactly): ").strip().upper()
    matching_hoods = [h for h in hoods if str(h).strip().upper() == your_hood]

if matching_hoods:
    actual_hood = matching_hoods[0]
    print(f"\n✅ Analyzing: {actual_hood}")
    
    # Filter data for neighborhood
    hood_data = df[df[COMMUNITY_COL] == actual_hood].copy()
    
    # Clean and convert values
    def clean_value(x):
        if isinstance(x, str):
            # Remove commas and dollar signs
            x = x.replace(',', '').replace('$', '').strip()
        try:
            return float(x)
        except:
            return None
    
    hood_data['CLEAN_VALUE'] = hood_data[VALUE_COL].apply(clean_value)
    hood_data_clean = hood_data.dropna(subset=['CLEAN_VALUE'])
    
    if len(hood_data_clean) > 0:
        # Calculate statistics
        avg_value = hood_data_clean['CLEAN_VALUE'].mean()
        median_value = hood_data_clean['CLEAN_VALUE'].median()
        min_value = hood_data_clean['CLEAN_VALUE'].min()
        max_value = hood_data_clean['CLEAN_VALUE'].max()
        std_value = hood_data_clean['CLEAN_VALUE'].std()
        
        print(f"\n📊 PROPERTY ANALYSIS FOR {actual_hood}:")
        print(f"   Properties found: {len(hood_data_clean):,}")
        print(f"   Average value: ${avg_value:,.2f}")
        print(f"   Median value: ${median_value:,.2f}")
        print(f"   Minimum value: ${min_value:,.2f}")
        print(f"   Maximum value: ${max_value:,.2f}")
        print(f"   Standard deviation: ${std_value:,.2f}")
        
        # Calculate percentiles
        percentiles = [25, 50, 75, 90, 95]
        percentile_values = hood_data_clean['CLEAN_VALUE'].quantile([p/100 for p in percentiles])
        
        print(f"\n📈 VALUE PERCENTILES:")
        for p, val in zip(percentiles, percentile_values):
            print(f"   {p}th percentile: ${val:,.2f}")
        
        # Calculate Calgary average
        df['CLEAN_VALUE'] = df[VALUE_COL].apply(clean_value)
        calgary_data = df.dropna(subset=['CLEAN_VALUE'])
        calgary_avg = calgary_data['CLEAN_VALUE'].mean()
        
        # Comparison
        difference = avg_value - calgary_avg
        difference_pct = (difference / calgary_avg) * 100
        
        print(f"\n📊 VS CALGARY OVERALL (from sample):")
        print(f"   Calgary average: ${calgary_avg:,.2f}")
        print(f"   Difference: ${difference:,.2f} ({difference_pct:+.1f}%)")
        
        if difference_pct > 0:
            print(f"   🟢 {actual_hood} is {difference_pct:.1f}% MORE expensive than Calgary average")
        else:
            print(f"   🔵 {actual_hood} is {abs(difference_pct):.1f}% LESS expensive than Calgary average")
        
        # Property type breakdown
        if 'PROPERTY_TYPE' in df.columns:
            print(f"\n🏠 PROPERTY TYPE DISTRIBUTION:")
            prop_types = hood_data_clean['PROPERTY_TYPE'].value_counts().head(5)
            for prop_type, count in prop_types.items():
                percentage = (count / len(hood_data_clean)) * 100
                print(f"   {prop_type:20} - {count:,} properties ({percentage:.1f}%)")
        
        # Save results
        print(f"\n💾 Saving results...")
        os.makedirs('outputs', exist_ok=True)
        
        # Save summary statistics
        summary = pd.DataFrame({
            'Neighborhood': [actual_hood],
            'Properties_Analyzed': [len(hood_data_clean)],
            'Average_Value': [avg_value],
            'Median_Value': [median_value],
            'Min_Value': [min_value],
            'Max_Value': [max_value],
            'Std_Dev': [std_value],
            'Calgary_Average': [calgary_avg],
            'Difference_Percent': [difference_pct],
            'Data_Source': ['First 50,000 rows of Calgary Property Assessment 2025']
        })
        
        output_file = f"outputs/{actual_hood.replace(' ', '_').replace('/', '_').lower()}_summary.csv"
        summary.to_csv(output_file, index=False)
        print(f"📊 Summary saved: {output_file}")
        
        # Save sample data
        sample_file = f"outputs/{actual_hood.replace(' ', '_').replace('/', '_').lower()}_sample.csv"
        sample_data = hood_data_clean[['ADDRESS', 'ASSESSED_VALUE', 'PROPERTY_TYPE', 'COMM_NAME', 'LAND_SIZE_SF']].head(50)
        sample_data.to_csv(sample_file, index=False)
        print(f"📋 Sample data saved: {sample_file}")
        
        # Create a simple text report
        report_file = f"outputs/{actual_hood.replace(' ', '_').replace('/', '_').lower()}_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"CALGARY PROPERTY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Neighborhood: {actual_hood}\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Properties analyzed: {len(hood_data_clean):,}\n\n")
            
            f.write("VALUE STATISTICS:\n")
            f.write(f"  Average: ${avg_value:,.2f}\n")
            f.write(f"  Median: ${median_value:,.2f}\n")
            f.write(f"  Minimum: ${min_value:,.2f}\n")
            f.write(f"  Maximum: ${max_value:,.2f}\n")
            f.write(f"  Range: ${max_value - min_value:,.2f}\n\n")
            
            f.write("COMPARISON TO CALGARY:\n")
            f.write(f"  Calgary average: ${calgary_avg:,.2f}\n")
            f.write(f"  Difference: ${difference:,.2f} ({difference_pct:+.1f}%)\n")
            
            if difference_pct > 0:
                f.write(f"  {actual_hood} is {difference_pct:.1f}% MORE expensive\n")
            else:
                f.write(f"  {actual_hood} is {abs(difference_pct):.1f}% LESS expensive\n")
        
        print(f"📝 Text report saved: {report_file}")
        
        print(f"\n💡 INSIGHTS:")
        if std_value / avg_value > 0.5:
            print("  • High variability in property values (wide range)")
        else:
            print("  • Relatively consistent property values")
        
        if avg_value > median_value:
            print("  • Average > Median: Some very high-value properties")
        elif avg_value < median_value:
            print("  • Average < Median: Some very low-value properties")
        
    else:
        print(f"\n❌ No valid property values found for {actual_hood}")
        print("   Check if ASSESSED_VALUE column contains numeric data")

print("\n" + "=" * 70)
print("🎉 ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n📁 Check the 'outputs' folder for:")
print(f"   • CSV summary with statistics")
print(f"   • CSV sample of properties")
print(f"   • Text report of findings")
print(f"\n🚀 Next steps:")
print(f"   1. Check the output files")
print(f"   2. Run on full dataset (remove 'nrows=50000')")
print(f"   3. Create visualizations with matplotlib")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your saved data
summary = pd.read_csv('outputs/beltline_summary.csv')
sample = pd.read_csv('outputs/beltline_sample.csv')

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Histogram of property values
plt.subplot(2, 2, 1)
plt.hist(sample['ASSESSED_VALUE'].str.replace(',', '').str.replace('$', '').astype(float) / 1000000, 
         bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Beltline Property Values (Sample)', fontsize=14, fontweight='bold')
plt.xlabel('Value ($ Millions)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(alpha=0.3)

# 2. Box plot
plt.subplot(2, 2, 2)
sample_values = sample['ASSESSED_VALUE'].str.replace(',', '').str.replace('$', '').astype(float) / 1000000
plt.boxplot(sample_values, vert=True, patch_artist=True)
plt.title('Value Distribution Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Value ($ Millions)', fontsize=12)
plt.grid(alpha=0.3)

# 3. Property type distribution
plt.subplot(2, 2, 3)
property_types = sample['PROPERTY_TYPE'].value_counts()
colors = ['#ff9999', '#66b3ff']
plt.pie(property_types.values, labels=property_types.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Property Type Distribution', fontsize=14, fontweight='bold')

# 4. Comparison with Calgary average
plt.subplot(2, 2, 4)
calgary_avg = summary['Calgary_Average'].iloc[0] / 1000000
beltline_avg = summary['Average_Value'].iloc[0] / 1000000
plt.bar(['Calgary Average', 'Beltline Average'], [calgary_avg, beltline_avg], 
        color=['lightgray', 'orange'], alpha=0.7, edgecolor='black')
plt.title('Beltline vs Calgary Average', fontsize=14, fontweight='bold')
plt.ylabel('Value ($ Millions)', fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/beltline_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualization saved to outputs/beltline_visualization.png")