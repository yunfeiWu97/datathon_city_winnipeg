import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the traffic count data from the provided CSV file
traffic_file_path = "Permanent_Count_Station_Traffic_Counts_20250301.csv"
traffic_df = pd.read_csv(traffic_file_path)

# Convert 'Timestamp' column to datetime
traffic_df['Timestamp'] = pd.to_datetime(traffic_df['Timestamp'])

# Extract hour from 'Timestamp' column
traffic_df['hour'] = traffic_df['Timestamp'].dt.hour

# Group by location and hour to find average traffic count
traffic_trends = traffic_df.groupby(['Site', 'hour'])['Total'].mean().reset_index()

# Identify peak traffic hours
peak_hours = traffic_df.groupby('hour')['Total'].sum().reset_index()

# Generate visualizations
def generate_visualizations(traffic_trends, peak_hours):
    # Line chart for traffic trends
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=traffic_trends, x='hour', y='Total', hue='Site')
    plt.title('Traffic Trends by Location and Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Traffic Volume')
    plt.legend(title='Location')
    plt.show()

    # Heatmap for high-congestion areas
    traffic_pivot = traffic_trends.pivot(index='Site', columns='hour', values='Total')
    plt.figure(figsize=(12, 8))
    sns.heatmap(traffic_pivot, cmap='viridis', annot=True, fmt=".1f")
    plt.title('Heatmap of Traffic Congestion by Location and Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Location')
    plt.show()

# Main function to run the analysis
def main():
    generate_visualizations(traffic_trends, peak_hours)

if __name__ == "__main__":
    main()
