from pathlib import Path
from pdb import set_trace

import numpy as np
import pandas as pd

# Get the current working directory
cwd = Path.cwd()
csv_folder = cwd / "plot4"
print(csv_folder)

# Get CSV files which contain the datestring in the filename
# text_to_filter = "combined_performances_20241001-193"
# text_to_filter = "20250501-223507" # PAPER godz NEW OL # "20250430-005648" # PAPER godz NEW #"20250225-162650"  # "20250319-184844"  # "20250319-160232"  # "20250225-001122"  # "20250225-162650"  # "20250224-200121"  # "20250224-222435"  #  # "20250219-191941"  # "20250207-162455"  # "20250206-202428"  # "20241002-015642"
# text_to_filter = "20250513-204000"  # with MUedit combo ALL
text_to_filter = "20250513-214000"  # with MUedit combo OL
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
if "MUedit" in combined_csv["sort_type"].values:
    MUedit_vals = combined_csv[combined_csv["sort_type"] == "MUedit"]
else:
    MUedit_vals = None
    
# Group by 'sort_type' and calculate mean and std of 'accuracy'
grouped_means_std = (combined_csv.groupby('sort_type')['accuracy'].agg(['median','mean', 'std','skew'])*100).round(1)

# Print the grouped means and standard deviations
print(grouped_means_std)


# create a svg violin plot which shows the accuracy values as points, side by side for EMUsort and Kilosort in plotly
# Kilosort4 values will serve as the x-axis and EMUsort values will serve as the y-axis. Dots are black and have a size of 10.
# draw a dotted firebrick red line to represent the 25th and 75th percentiles, line width 4
# draw a solid firebrick red line to represent the median, line width 4
# use font Open Sans, size 24, color black, bolded
# set x-axis label to "Kilosort4 Sort Accuracy", bolded
# set y-axis label to "EMUsort Sort Accuracy", bolded
# use 'plotly_white' theme
# save the plot as a .svg file in a new plot6 folder with the filename "scatter_accuracy_EMUsort_vs_Kilosort4.svg"

import plotly.graph_objects as go

fig = go.Figure()


fig.add_trace(
    go.Violin(
        # x=np.ones(EMUsort_vals["accuracy"].shape)
        # + np.random.normal(0, 0.04, EMUsort_vals["accuracy"].shape),
        y=EMUsort_vals["accuracy"],
        box_visible=True,
        # meanline_visible=True,
        points="all",
        line_color="firebrick",
        name="EMUsort",
        pointpos=0.5,
        width=2,
        side="negative",
        alignmentgroup=0
    )
)

if MUedit_vals is not None:
    fig.add_trace(
    go.Violin(
        # x=np.ones(MUedit_vals["accuracy"].shape)
        # + np.random.normal(0, 0.04, MUedit_vals["accuracy"].shape),
        y=MUedit_vals["accuracy"],
        box_visible=True,
        # meanline_visible=True,
        points="all",
        line_color="lightskyblue",
        name="MUedit",
        pointpos=0.5,
        width=2,
        side="negative",
        alignmentgroup=0
    )
)
    
fig.add_trace(
    go.Violin(
        # x=np.ones(Kilosort4_vals["accuracy"].shape)
        # + np.random.normal(0, 0.04, Kilosort4_vals["accuracy"].shape),
        y=Kilosort4_vals["accuracy"],
        box_visible=True,
        # meanline_visible=True,
        points="all",
        line_color="black",
        name="Kilosort4",
        pointpos=-0.5,
        width=2,
        side="positive",
        alignmentgroup=1
    )
)


fig.update_layout(
    template="plotly_white",
    font=dict(family="Open Sans", size=24, color="black", weight="bold"),
    title_font=dict(size=36, color="black", family="Open Sans", weight="bold"),
    title="All Spikes Included",  # "All Spikes",  # "Overlapping Spikes Only",
    # title="All Spikes Evaluated",
    # title="Sort Accuracy Distributions (All Spikes Evaluated)<br><sup>25 parameter combinations each</sup>",
    # xaxis_title="Kilosort4 Sort Accuracy",
    yaxis_title="Sort Accuracy (Mean Across MUs)",
    violinmode="group",
    violingap=0,
    violingroupgap=0,
    showlegend=False,
)

# fig.update_yaxes(range=[0.55, 1])
fig.update_yaxes(range=[x - 0 for x in [0.0, 1.05]])

if Path(cwd / "plot7").is_dir() is False:
    Path(cwd / "plot7").mkdir()

# Save the plot
# fig.show()
fig.write_image(f"plot7/violin_accuracy_EMUsort_vs_Kilosort4_{text_to_filter}.svg")
