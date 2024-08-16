from pathlib import Path

import pandas as pd

# Get the current working directory
cwd = Path.cwd()
csv_folder = cwd / "plot4"
print(csv_folder)

# Get CSV files which contain the datestring in the filename
datestring_to_filter = "20240805"
csv_files = list(csv_folder.glob(f"*{datestring_to_filter}*.csv"))
print(csv_files)

# Combine the CSV files
combined_csv = pd.concat([pd.read_csv(f) for f in csv_files])

# name first column as 'cluster'
combined_csv.rename(columns={"Unnamed: 0": "cluster"}, inplace=True)

# sort the 'noise_level' column in ascending order, then sort the 'accuracy' column in descending order
combined_csv = combined_csv.sort_values(
    ["noise_level", "accuracy"], ascending=[True, False]
)

# Save the combined and sorted CSV file
combined_csv.to_csv(
    csv_folder / f"combined_performances_{datestring_to_filter}.csv", index=False
)

"""
This script will combine all CSV files in the `plot4` folder that contain the datestring `20240612` in their filename. The combined CSV file will be saved as `combined_20240612.csv` in the same folder.
You can run this script by saving it as `combine_csvs.py` and running it with Python. Make sure to have the necessary libraries installed (`pandas`) by running `pip install pandas` in your terminal.
"""
