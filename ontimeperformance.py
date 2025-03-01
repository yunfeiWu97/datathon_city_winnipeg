import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from shapely import wkt
import numpy as np

# Load the data
df = pd.read_csv('winnipeg_data_first_10000.csv')

# Convert scheduled_time to datetime
df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])

# Basic statistics on deviation
print(f"Average deviation: {df['deviation'].mean():.2f} seconds")
print(f"Minimum deviation: {df['deviation'].min()} seconds")
print(f"Maximum deviation: {df['deviation'].max()} seconds")

# Create a figure with multiple plots
plt.figure(figsize=(15, 12))

# Plot 1: Histogram of deviation times
plt.subplot(2, 2, 1)
sns.histplot(df['deviation'], bins=30, kde=True)
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Distribution of Bus Arrival Deviations')
plt.xlabel('Deviation (seconds)')
plt.ylabel('Count')

# Plot 2: Box plot of deviation by route_number (top 10 routes by frequency)
plt.subplot(2, 2, 2)
top_routes = df['route_number'].value_counts().head(10).index
route_data = df[df['route_number'].isin(top_routes)]
sns.boxplot(x='route_number', y='deviation', data=route_data)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Deviation by Route (Top 10 Routes)')
plt.xlabel('Route Number')
plt.ylabel('Deviation (seconds)')

# Plot 3: Deviation by time of day
plt.subplot(2, 2, 3)
df['hour'] = df['scheduled_time'].dt.hour
hourly_avg = df.groupby('hour')['deviation'].mean().reset_index()
sns.lineplot(x='hour', y='deviation', data=hourly_avg, marker='o')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Average Deviation by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Deviation (seconds)')
plt.xticks(range(0, 24))

# Plot 4: Deviation by day type
plt.subplot(2, 2, 4)
sns.boxplot(x='day_type', y='deviation', data=df)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Deviation by Day Type')
plt.xlabel('Day Type')
plt.ylabel('Deviation (seconds)')

plt.tight_layout()
plt.savefig('deviation_analysis.png')

# Create a map visualization
# Parse the POINT strings to extract latitude and longitude
def extract_coords(point_str):
    try:
        # Extract numbers from the POINT string
        coords = point_str.replace('POINT (', '').replace(')', '').split()
        return float(coords[0]), float(coords[1])
    except:
        return None, None

df['longitude'], df['latitude'] = zip(*df['location'].apply(extract_coords))

# Create a map centered on Winnipeg
m = folium.Map(location=[49.895, -97.138], zoom_start=12)

# Sample the data to avoid too many points (take 500 random points)
map_data = df.sample(min(500, len(df)))

# Add markers colored by deviation
for _, row in map_data.iterrows():
    # Color based on deviation
    if row['deviation'] < -60:  # More than 1 minute early
        color = 'blue'
    elif row['deviation'] < 60:  # On time (within 1 minute)
        color = 'green'
    elif row['deviation'] < 300:  # 1-5 minutes late
        color = 'orange'
    else:  # More than 5 minutes late
        color = 'red'
    
    # Add marker
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"Route: {row['route_number']} - {row['route_name']}<br>"
              f"Destination: {row['route_destination']}<br>"
              f"Deviation: {row['deviation']} seconds"
    ).add_to(m)

# Add a legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
            padding: 10px; border: 2px solid grey; border-radius: 5px;">
<p><b>Deviation Legend</b></p>
<p><i class="fa fa-circle" style="color:blue"></i> More than 1 minute early</p>
<p><i class="fa fa-circle" style="color:green"></i> On time (within 1 minute)</p>
<p><i class="fa fa-circle" style="color:orange"></i> 1-5 minutes late</p>
<p><i class="fa fa-circle" style="color:red"></i> More than 5 minutes late</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save the map
m.save('bus_deviation_map.html')

print("Analysis complete. Check deviation_analysis.png and bus_deviation_map.html for visualizations.")
