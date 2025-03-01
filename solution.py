import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import folium
from folium.plugins import HeatMap
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import random
import io
import base64
from PIL import Image

# Set styling
plt.style.use('dark_background')
sns.set_palette("viridis")

# Function to create a dashboard-style output
def create_traffic_prediction_dashboard():
    """
    Creates a mock AI traffic prediction dashboard for Winnipeg
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create current date and next 3 days
    today = datetime.now()
    prediction_dates = [today + timedelta(days=i) for i in range(1, 4)]
    date_strs = [d.strftime("%A, %B %d, %Y") for d in prediction_dates]
    
    # Major roads in Winnipeg
    major_roads = [
        "Portage Avenue", "Main Street", "Pembina Highway", "Henderson Highway",
        "St. Mary's Road", "Regent Avenue", "Kenaston Boulevard", "Bishop Grandin Boulevard",
        "Lagimodiere Boulevard", "St. Anne's Road", "Fermor Avenue", "Chief Peguis Trail"
    ]
    
    # Create time slots for each day (hour intervals)
    hours = list(range(6, 22))  # 6 AM to 9 PM
    time_slots = [f"{hour:02d}:00" for hour in hours]
    
    # Generate fake traffic prediction data
    # Higher values indicate heavier predicted traffic
    traffic_data = []
    
    # Generate predictions for each date
    for date_idx, pred_date in enumerate(prediction_dates):
        # Morning and evening rush hours will have higher values
        # Adjust baseline for each day (slight variation)
        day_factor = 0.8 + (date_idx * 0.1)  
        
        for road in major_roads:
            # Base traffic profile for the road (some roads are busier)
            road_factor = 0.5 + random.random() 
            
            for hour_idx, hour in enumerate(hours):
                # Traffic patterns based on time of day
                if 7 <= hour <= 9:  # Morning rush
                    time_factor = 0.8 + (random.random() * 0.4)
                elif 16 <= hour <= 18:  # Evening rush
                    time_factor = 0.9 + (random.random() * 0.3)
                elif 11 <= hour <= 13:  # Lunch hour
                    time_factor = 0.6 + (random.random() * 0.3)
                else:  # Normal hours
                    time_factor = 0.3 + (random.random() * 0.3)
                
                # Generate traffic score (0-100)
                # Add randomness for weekend vs weekday
                weekday_factor = 1.0
                if pred_date.weekday() >= 5:  # Weekend
                    weekday_factor = 0.7
                
                traffic_score = min(100, (road_factor * time_factor * day_factor * weekday_factor * 100) + random.randint(-10, 10))
                
                traffic_data.append({
                    'date': pred_date.strftime("%Y-%m-%d"),
                    'day': pred_date.strftime("%A"),
                    'road': road,
                    'hour': hour,
                    'time': f"{hour:02d}:00",
                    'traffic_score': traffic_score,
                    'congestion_level': get_congestion_level(traffic_score)
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(traffic_data)
    
    # Create a figure with multiple components
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    # Title and explanation
    plt.suptitle('TrafficSense AI: Winnipeg Traffic Predictions', fontsize=24, y=0.98)
    subtitle = "AI-powered prediction of traffic congestion for the next 3 days"
    plt.figtext(0.5, 0.94, subtitle, fontsize=16, ha='center')
    
    # Add model description
    model_text = (
        "MODEL: DeepTraffic GPT-7 Neural Network\n"
        "ACCURACY: 94.7% (based on historical validation)\n"
        "DATA SOURCES: Historical traffic patterns, weather forecasts, event schedules, road sensors"
    )
    plt.figtext(0.5, 0.90, model_text, fontsize=12, ha='center', 
                bbox=dict(facecolor='#333333', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # 1. Create heatmaps for each day (road vs time)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 2, 1])
    
    for day_idx, day in enumerate(sorted(df['date'].unique())):
        day_data = df[df['date'] == day]
        
        # Create pivot table manually to fix the error
        pivot_data = pd.pivot_table(
            day_data,
            values='traffic_score',
            index='road',
            columns='hour'
        )
        
        ax = plt.subplot(gs[day_idx, 0])
        sns.heatmap(pivot_data, cmap="RdYlGn_r", ax=ax, cbar=day_idx==0, 
                   vmin=0, vmax=100, annot=False)
        
        day_name = datetime.strptime(day, "%Y-%m-%d").strftime("%A")
        ax.set_title(f"{day_name}, {day}", fontsize=14)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Road", fontsize=12)
        
        # Emphasize peak traffic times
        day_peak = df[df['date'] == day].sort_values('traffic_score', ascending=False).iloc[0]
        ax.text(0.5, -0.15, 
                f"Peak Traffic: {day_peak['road']} at {day_peak['time']}", 
                transform=ax.transAxes, ha='center', fontsize=11,
                bbox=dict(facecolor='red', alpha=0.3))
    
    # 2. Line plots for selected roads over time
    top_roads = df.groupby('road')['traffic_score'].mean().sort_values(ascending=False).head(5).index.tolist()
    time_data = df[df['road'].isin(top_roads)]
    
    for day_idx, day in enumerate(sorted(df['date'].unique())):
        ax = plt.subplot(gs[day_idx, 1])
        day_time_data = time_data[time_data['date'] == day]
        
        for road in top_roads:
            road_data = day_time_data[day_time_data['road'] == road]
            ax.plot(road_data['hour'], road_data['traffic_score'], marker='o', label=road)
        
        ax.set_title(f"Traffic Predictions - {day}", fontsize=14)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Congestion Score (0-100)", fontsize=12)
        ax.set_xticks(range(6, 22, 2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 22, 2)])
        ax.grid(True, alpha=0.3)
        
        if day_idx == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # 3. Peak traffic predictions in a table
    ax = plt.subplot(gs[:, 2])
    ax.axis('off')
    
    peak_times_table = []
    for day in sorted(df['date'].unique()):
        day_data = df[df['date'] == day]
        morning_peak = day_data[(day_data['hour'] >= 7) & (day_data['hour'] <= 9)].sort_values('traffic_score', ascending=False).iloc[0]
        evening_peak = day_data[(day_data['hour'] >= 16) & (day_data['hour'] <= 18)].sort_values('traffic_score', ascending=False).iloc[0]
        
        day_name = datetime.strptime(day, "%Y-%m-%d").strftime("%a %b %d")
        peak_times_table.append([
            day_name,
            f"{morning_peak['time']}",
            f"{morning_peak['road']}",
            f"{morning_peak['traffic_score']:.1f}",
            f"{evening_peak['time']}",
            f"{evening_peak['road']}",
            f"{evening_peak['traffic_score']:.1f}"
        ])
    
    table_data = peak_times_table
    columns = ['Day', 'AM Time', 'AM Location', 'Score', 'PM Time', 'PM Location', 'Score']
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#444444'] * len(columns),
        colWidths=[0.13] + [0.1, 0.28, 0.07] * 2
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    ax.set_title("Predicted Peak Traffic Times", fontsize=16, pad=20)
    
    # Add prediction insights
    insights = [
        "• Traffic patterns suggest avoiding Portage Avenue during morning rush hours",
        "• Weather forecast indicates potential slow traffic on Pembina Highway Wed evening",
        "• Construction on Main Street will increase congestion by 35% on Thursday",
        "• Traffic sensors predict 25% higher volumes approaching downtown during rush hours",
        "• Suggested alternate route: Chief Peguis Trail to avoid North Main bottlenecks"
    ]
    
    insight_text = "AI-GENERATED INSIGHTS:\n" + "\n".join(insights)
    plt.figtext(0.5, 0.09, insight_text, fontsize=12, ha='center',
                bbox=dict(facecolor='#333366', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add footer with data update information
    update_text = f"Last Model Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Prediction Confidence: 92.3% | Data Sources: 17"
    plt.figtext(0.5, 0.01, update_text, fontsize=8, ha='center', alpha=0.7)
    
    # Save the figure
    plt.savefig('winnipeg_traffic_prediction.png', dpi=300, bbox_inches='tight', facecolor='#222222')
    plt.close()
    
    print("Traffic prediction visualization created as 'winnipeg_traffic_prediction.png'")
    
    # Create a map visualization with traffic hotspots
    create_traffic_map(df)

def get_congestion_level(score):
    """Determine congestion level from score"""
    if score < 30:
        return "Light"
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "Heavy"
    else:
        return "Severe"

def create_traffic_map(df):
    """Create a map visualization with predicted traffic hotspots"""
    # Winnipeg coordinates
    winnipeg_center = [49.8951, -97.1384]
    
    # Create a map centered on Winnipeg
    m = folium.Map(location=winnipeg_center, zoom_start=12, tiles="cartodbdark_matter")
    
    # Add title
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Predicted Traffic Hotspots - Next 3 Days</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Road coordinates (approximate)
    road_coords = {
        "Portage Avenue": [[49.8798, -97.2071], [49.8798, -97.1471]],
        "Main Street": [[49.9151, -97.1384], [49.8551, -97.1384]],
        "Pembina Highway": [[49.8551, -97.1584], [49.7751, -97.1384]],
        "Henderson Highway": [[49.9151, -97.1084], [49.9551, -97.0684]],
        "St. Mary's Road": [[49.8551, -97.1184], [49.7951, -97.0984]],
        "Regent Avenue": [[49.8951, -97.0684], [49.8951, -97.0084]],
        "Kenaston Boulevard": [[49.8351, -97.1984], [49.8951, -97.1984]],
        "Bishop Grandin Boulevard": [[49.8151, -97.2071], [49.8151, -97.1084]],
        "Lagimodiere Boulevard": [[49.9151, -97.0684], [49.8351, -97.0684]],
        "St. Anne's Road": [[49.8551, -97.1084], [49.7951, -97.0784]],
        "Fermor Avenue": [[49.8351, -97.1684], [49.8351, -97.0684]],
        "Chief Peguis Trail": [[49.9351, -97.1984], [49.9351, -97.0684]]
    }
    
    # Calculate average congestion by road
    avg_congestion = df.groupby('road')['traffic_score'].mean().reset_index()
    
    # Create lines for each road with color based on congestion
    for _, row in avg_congestion.iterrows():
        road = row['road']
        score = row['traffic_score']
        
        if road in road_coords:
            # Determine color based on score (red for high congestion)
            color = get_color_for_score(score)
            
            # Create a line with popup info
            coords = road_coords[road]
            popup_text = f"{road}<br>Average Congestion: {score:.1f}/100<br>{get_congestion_level(score)} Traffic"
            
            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.8,
                popup=popup_text
            ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 170px; height: 130px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
        ">
        <div style="font-weight: bold; margin-bottom: 5px;">Traffic Prediction</div>
        <span style="background-color: #00ff00; width: 15px; height: 15px; display: inline-block;"></span> Light (0-30)<br>
        <span style="background-color: #ffff00; width: 15px; height: 15px; display: inline-block;"></span> Moderate (30-60)<br>
        <span style="background-color: #ffa500; width: 15px; height: 15px; display: inline-block;"></span> Heavy (60-80)<br>
        <span style="background-color: #ff0000; width: 15px; height: 15px; display: inline-block;"></span> Severe (80-100)<br>
        <div style="font-size: 10px; margin-top: 5px;">Powered by TrafficSense AI</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    m.save('winnipeg_traffic_prediction_map.html')
    print("Traffic map visualization created as 'winnipeg_traffic_prediction_map.html'")

def get_color_for_score(score):
    """Return a color based on traffic score"""
    if score < 30:
        return 'green'
    elif score < 60:
        return 'yellow'
    elif score < 80:
        return 'orange'
    else:
        return 'red'

# Execute the dashboard creation
if __name__ == "__main__":
    print("Initializing TrafficSense AI...")
    print("Loading historical traffic patterns...")
    print("Integrating weather data and event schedules...")
    print("Running DeepTraffic GPT-7 predictive model...")
    print("Generating traffic predictions for next 3 days...")
    create_traffic_prediction_dashboard()
    print("Analysis complete. Predictions ready for review.")
    print("\nTIP: Open winnipeg_traffic_prediction.png to view the graphical dashboard")
    print("TIP: Open winnipeg_traffic_prediction_map.html in a web browser to view the interactive map")
    print("TIP: Open traffic_predict.html in a web browser for the interactive interface")
