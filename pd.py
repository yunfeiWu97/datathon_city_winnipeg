import pandas as pd

# URL of the dataset with limit parameter to get 10000 rows
url = "https://data.winnipeg.ca/resource/gp3k-am4u.csv?$limit=1000000"

# Load the data
df = pd.read_csv(url)

# Display the first few rows
print(df.head())

# Optionally, save to a local file
df.to_csv("winnipeg_data_first_10000.csv", index=False)
