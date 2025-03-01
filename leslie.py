import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Load the data
file_path = 'weatherstats_winnipeg_daily (1).csv'
data = pd.read_csv(file_path, parse_dates=['date'])

# Sort by date (ascending)
data = data.sort_values('date')

# Filter data for December 2024 to February 2025
start_date = '2024-12-01'
end_date = '2025-02-28'
filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

print(f"Filtered data from {start_date} to {end_date}")
print(f"Total records in filtered dataset: {len(filtered_data)}")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 12))
fig.suptitle('Winnipeg Weather Data Analysis (Dec 2024 - Feb 2025)', fontsize=20, y=0.95)

# Plot 1: Temperature Trends
ax1 = plt.subplot(3, 1, 1)
ax1.plot(filtered_data['date'], filtered_data['max_temperature'], label='Max Temperature', color='red')
ax1.plot(filtered_data['date'], filtered_data['avg_temperature'], label='Avg Temperature', color='green')
ax1.plot(filtered_data['date'], filtered_data['min_temperature'], label='Min Temperature', color='blue')
ax1.set_ylabel('Temperature (°C)')
ax1.legend(loc='best')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)


# Add a fourth plot: Snow on ground if available
if 'snow_on_ground' in filtered_data.columns and not filtered_data['snow_on_ground'].isnull().all():
    # Create a new row for this plot
    fig.set_figheight(16)  # Increase figure height
    ax4 = plt.subplot(4, 1, 4)
    ax4.bar(filtered_data['date'], filtered_data['snow_on_ground'], color='lightblue', alpha=0.7,
           label='Snow on Ground (cm)')
    ax4.set_title('Snow on Ground (Winter 2024-2025)')
    ax4.set_ylabel('Snow Depth (cm)')
    ax4.legend(loc='best')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)

# Fix for layout warnings by setting equal number of subplots
# Check how many subplots we're actually using
n_plots = 4 if 'snow_on_ground' in filtered_data.columns and not filtered_data['snow_on_ground'].isnull().all() else 3

# Adjust layout
plt.subplots_adjust(hspace=0.3, top=0.92)

# Save the visualization
plt.savefig('winnipeg_winter_2024_2025.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Visualization complete! Saved as 'winnipeg_winter_2024_2025.png'")

# Additional Analysis: Temperature Statistics for the winter period
print("\nTemperature Statistics (Dec 2024 - Feb 2025):")
temp_stats = filtered_data[['max_temperature', 'avg_temperature', 'min_temperature']].describe()
print(temp_stats)

# Calculate and print the coldest and warmest days in the winter period
coldest_day = filtered_data.loc[filtered_data['min_temperature'].idxmin()]
warmest_day = filtered_data.loc[filtered_data['max_temperature'].idxmax()]

print(f"\nColdest day: {coldest_day['date'].strftime('%Y-%m-%d')} with minimum temperature of {coldest_day['min_temperature']}°C")
print(f"Warmest day: {warmest_day['date'].strftime('%Y-%m-%d')} with maximum temperature of {warmest_day['max_temperature']}°C")

# Calculate additional winter statistics
if 'snow_on_ground' in filtered_data.columns and not filtered_data['snow_on_ground'].isnull().all():
    max_snow_day = filtered_data.loc[filtered_data['snow_on_ground'].idxmax()]
    print(f"Maximum snow depth: {max_snow_day['snow_on_ground']} cm on {max_snow_day['date'].strftime('%Y-%m-%d')}")

# Calculate the number of extreme cold days (min temp below -20°C)
extreme_cold_days = filtered_data[filtered_data['min_temperature'] <= -20].shape[0]
print(f"Number of extreme cold days (min temp ≤ -20°C): {extreme_cold_days}")

# Calculate the average windchill if available
if 'min_windchill' in filtered_data.columns and not filtered_data['min_windchill'].isnull().all():
    avg_windchill = filtered_data['min_windchill'].mean()
    min_windchill_day = filtered_data.loc[filtered_data['min_windchill'].idxmin()]
    print(f"Average minimum windchill: {avg_windchill:.1f}°C")
    print(f"Lowest windchill: {min_windchill_day['min_windchill']}°C on {min_windchill_day['date'].strftime('%Y-%m-%d')}")
