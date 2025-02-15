# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:49:34 2025

@author: dell
"""
import json

# json_metadata_path = r"D:/Bureau/FI-2010/metadata.json"  # Use raw string (r"")
# with open(json_metadata_path, "r", encoding="utf-8") as f:
#     json_metadata = json.load(f)


# print(json_metadata.keys()) 

    
# import webbrowser
# webbrowser.open("https://research.csc.fi/ida")
# import os

# folder_path = r"D:/Bureau/FI-2010/"
# files = os.listdir(folder_path)

file_path = r"D:/Bureau/FI-2010/Auction/1.Auction_Zscore/Auction_Zscore_Training/Train_Dst_Auction_ZScore_CF_1.txt"
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    content = f.read().strip()  # Read as a single line

# Convert into a list of values
values = content.split()  # Split by spaces

# Reshape into rows of 144 columns
columns_per_row = 144  # Adjust if needed
rows = [values[i:i + columns_per_row] for i in range(0, len(values), columns_per_row)]

# Save the fixed version
# fixed_file = "fixed_dataset.txt"
# with open(fixed_file, "w", encoding="utf-8") as f:
#     for row in rows:
#         f.write(" ".join(row) + "\n")

# print(f"Fixed dataset saved as: {fixed_file}")

import pandas as pd
df = pd.read_csv(r'fixed_dataset.txt', delim_whitespace=True, header=None)

import numpy as np

print("Mean of each column:")
print(df.mean().round(3))

print("\nStandard deviation of each column:")
print(df.std().round(3))

corr_matrix = df.corr()
import matplotlib.pyplot as plt
import seaborn as sns
# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()


correlated_features = df.corr().abs().unstack().sort_values(ascending=False)

# Remove self-correlations (where row and column indices are the same)
correlated_features = correlated_features[correlated_features.index.get_level_values(0) != correlated_features.index.get_level_values(1)]

print(correlated_features.head(20))  # Show top 20 real correlations