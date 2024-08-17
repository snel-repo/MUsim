import os

import pandas as pd
import plotly.graph_objects as go

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("memory_usage.csv")

# Convert 'Timestamp' to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Convert 'Total Memory Usage (KB)' from KB to GB
df["Total Memory Usage (GB)"] = df["Total Memory Usage (KB)"] / 1000000

# Calculate the time difference in seconds from the first timestamp
df["Time (Minutes)"] = (
    df["Timestamp"] - df["Timestamp"].iloc[0]
).dt.total_seconds() / 60

# Create a Plotly line plot
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Time (Minutes)"],
        y=df["Total Memory Usage (GB)"],
        mode="lines",
        name="Memory Usage",
        line=dict(width=2, color="orange"),
    )
)

hostname = os.uname().nodename.title()
# Update layout for the plot
fig.update_layout(
    title=f"{hostname} Memory Usage Over Time",
    xaxis_title="Time (Minutes)",
    yaxis_title="Memory Usage (Gigabytes)",
    xaxis=dict(showline=True, showgrid=True, zeroline=False, title="Time (Minutes)"),
    yaxis=dict(
        showline=True, showgrid=True, zeroline=True, title="Memory Usage (Gigabytes)"
    ),
    template="plotly_dark",
)

# Show the plot
fig.show()
