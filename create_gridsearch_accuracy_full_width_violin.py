from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Get the current working directory
cwd = Path.cwd()
csv_folder = cwd / "plot4"
text_to_filter = "20241002-015642"

# Get CSV files containing the date string in the filename
csv_files = list(csv_folder.glob(f"*{text_to_filter}*.csv"))

# Combine the CSV files
combined_csv = pd.concat([pd.read_csv(f) for f in csv_files])

# Rename the first column as 'cluster'
combined_csv.rename(columns={"Unnamed: 0": "cluster"}, inplace=True)

# Sort values
combined_csv = combined_csv.sort_values(
    ["noise_level", "accuracy"], ascending=[True, False]
)

# Filter for Kilosort4 and EMUsort values
Kilosort4_vals = combined_csv[combined_csv["sort_type"] == "Kilosort4"]
EMUsort_vals = combined_csv[combined_csv["sort_type"] == "EMUsort"]

# Create a violin plot
fig = go.Figure()

# Add Kilosort4 violin plot
fig.add_trace(
    go.Violin(
        y=Kilosort4_vals["accuracy"],
        box_visible=True,
        points="all",
        line_color="firebrick",
        name="Kilosort4",
        # side="positive",
        # width=0.4,  # Adjusted width to bring closer together
    )
)

# Add EMUsort violin plot
fig.add_trace(
    go.Violin(
        y=EMUsort_vals["accuracy"],
        box_visible=True,
        points="all",
        line_color="black",
        name="EMUsort",
        # side="positive",
        # width=0.4,  # Adjusted width to bring closer together
    )
)

fig.update_layout(violingap=0)
# Update layout
fig.update_layout(
    template="plotly_white",
    font=dict(family="Open Sans", size=16, color="black", weight="bold"),
    title="Sort Accuracy Distributions (Overlapping Spikes)<br><sup>25 parameter combinations each</sup>",
    yaxis_title="Sort Accuracy",
)

fig.update_yaxes(range=[0.55, 1])

# now bring the sections much closer together

# Create the output directory if it doesn't exist
output_dir = cwd / "plot7"
output_dir.mkdir(exist_ok=True)

# Save the plot
fig.write_image(
    f"{output_dir}/violin_accuracy_EMUsort_vs_Kilosort4_{text_to_filter}.svg"
)
