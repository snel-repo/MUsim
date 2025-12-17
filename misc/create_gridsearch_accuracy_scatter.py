from pathlib import Path

import pandas as pd

# Get the current working directory
cwd = Path.cwd()
csv_folder = cwd / "plot4"
print(csv_folder)

# Get CSV files which contain the datestring in the filename
text_to_filter = "combined_performances_20241001-193"
# text_to_filter = "20241002-015642"
csv_files = list(csv_folder.glob(f"*{text_to_filter}*.csv"))
print(csv_files)

# Combine the CSV files
combined_csv = pd.concat([pd.read_csv(f) for f in csv_files])

# name first column as 'cluster'
combined_csv.rename(columns={"Unnamed: 0": "cluster"}, inplace=True)

# sort the 'noise_level' column in ascending order, then sort the 'accuracy' column in descending order
combined_csv = combined_csv.sort_values(
    ["noise_level", "accuracy"], ascending=[True, False]
)

Kilosort4_vals = combined_csv[combined_csv["sort_type"] == "Kilosort4"]
EMUsort_vals = combined_csv[combined_csv["sort_type"] == "EMUsort"]

# create a svg scatter plot in plotly of the accuracy values where sort_type column is filtered for "EMUsort" and "Kilosort4"
# Kilosort4 values will serve as the x-axis and EMUsort values will serve as the y-axis. Dots are black and have a size of 10.
# draw a dotted firebrick red line from (0,0) to (1,1) to represent the line of equality, line width 4
# set axis bounds to be from minimum value - 0.1 to maximum value + 0.1, bounded to 0 and 1.
# set title to "Paired Accuracy of Sorts with EMUsort vs Kilosort4", title font size 30, color black, bolded, Open Sans font
# use font Open Sans, size 24, color black, bolded
# set x-axis label to "Kilosort4 Sort Accuracy", bolded
# set y-axis label to "EMUsort Sort Accuracy", bolded
# use 'plotly_white' theme
# save the plot as a .svg file in a new plot6 folder with the filename "scatter_accuracy_EMUsort_vs_Kilosort4.svg"

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=Kilosort4_vals["accuracy"],
        y=EMUsort_vals["accuracy"],
        mode="markers",
        # give a hollow circle
        marker=dict(color="black", size=30, opacity=0.85),
        name="Accuracy",
    )
)

fig.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=1,
    y1=1,
    line=dict(color="firebrick", width=15, dash="dot"),
)

fig.update_xaxes(
    # range=[min(combined_csv["accuracy"]) - 0.05, 1],
    range=[0.54, 1],
)

fig.update_yaxes(
    # range=[min(combined_csv["accuracy"]) - 0.05, 1],
    range=[0.54, 1],
)


fig.update_layout(
    template="plotly_white",
    title="Paired Accuracy of Sorts (All Spikes)",
    font=dict(family="Open Sans", size=28, color="black", weight="bold"),
    title_font=dict(size=36, color="black", family="Open Sans", weight="bold"),
    # make axis ticks not bold
    # xaxis=dict(title_font=dict(size=24, color="black", weight="bold")),
    # yaxis=dict(title_font=dict(size=24, color="black", weight="bold")),
    xaxis_title="Kilosort4 Accuracy",
    yaxis_title="EMUsort Accuracy",
    autosize=False,
    width=1000,
    height=1000,
)

# make axes equal
fig.update_xaxes(matches="y")
fig.update_yaxes(matches="x")

if Path(cwd / "plot6").is_dir() is False:
    Path(cwd / "plot6").mkdir()

fig.write_image(
    cwd / "plot6" / f"scatter_accuracy_EMUsort_vs_Kilosort4_{text_to_filter}.svg"
)
