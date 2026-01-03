# yyc-data-analysis
Calgary property assessment analysis
# Calgary Property Analysis Project

## What This Project Is About
I analyzed Calgary's property assessment data to understand how property values vary across different neighborhoods, with a special focus on the Beltline area. This project processes the entire city dataset to find patterns and insights.

## What I Was Trying to Figure Out
The goal was to answer several questions about Calgary's property market:
- Which neighborhoods are most expensive vs most affordable?
- How evenly distributed are property values within neighborhoods?
- What types of properties (commercial vs residential) dominate different areas?
- Are there any surprising patterns or outliers in the data?

## What I Found

### The Beltline Neighborhood Stands Out
The Beltline area has some striking characteristics:
- Properties there are **214% more expensive** than the Calgary average
- There's extreme variation in values - from $18,520 all the way up to $615.5 million
- **92.5% of properties** are classified as Light Industrial
- The average value ($9.68M) is much higher than the median ($2.56M), meaning a few very expensive properties skew the average

### Calgary-Wide Patterns
Looking at the whole city:
- I analyzed properties across 310 different neighborhoods
- Beltline, Tuscany, and Lake Bonavista have the most properties
- Downtown areas have heavy concentrations of commercial properties
- Property values vary dramatically across the city

## How I Did It

### My Process
1. **Got the data** from Calgary's Open Data portal (it's a 282MB CSV file)
2. **Cleaned it up** - fixed formatting issues, converted text to numbers, removed bad data
3. **Did the analysis** - calculated averages, medians, looked at distributions
4. **Made charts** to visualize what I found
5. **Saved everything** so others can see my work

### Tools I Used
- **Python** for all the analysis work
- **Pandas** to handle and process the data
- **Matplotlib and Seaborn** to create the charts
- **VS Code** to write and run everything

## Project Files
Here's what's in this project:
Data Analysis/
- README.md # This file - explains the project
- calgary_property_analysis.py # My main analysis script
- Current_Year_Property_Assessments_(Parcel)_20260102.csv # The raw data
- visualizations/ # Charts I created
  - calgary_property_analysis.png # The main dashboard chart
- outputs/ # Results from my analysis
- dataset_summary.csv # Overall statistics
- neighborhood_stats.csv # Data for all neighborhoods
- property_sample.csv # A sample of 1,000 properties
- beltline_properties.csv # Detailed Beltline data
