# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:20:41 2025

@author: dell
"""

import pandas as pd
import numpy as np

# Features

basic_features = [
    f"{feature}_{i}"
    for i in range(1, 11)
    for feature in ["P_Ask", "V_Ask", "P_Bid", "V_Bid"]
]

time_insensitive_features = [
        f"{feature}_{i}"
        for i in range(1,11)
        for feature in ["Spread", "MidPrice"]
    ] + [
        "P_Diff_Ask", "P_Diff_Bid"
    ] + [
        f"{feature}_{i}"
        for i in range(1,10)
        for feature in ["P_AbsDiffRel_Ask", "P_AbsDiffRel_Bid"]
    ] + [
        "P_Mean_Ask",
        "P_Mean_Bid",
        "V_Mean_Ask",
        "V_Mean_Bid"
    ] + [
        "P_AccDiff",
        "V_AccDiff"
    ]

time_sensitive_features = [
        f"{feature}_{i}"
        for i in range(1,11)
        for feature in ["P_Deriv_Ask", "P_Deriv_Bid", "V_Deriv_Ask", "V_Deriv_Bid"]
    ] + [
        f"IntensityAverage_{i}"
        for i in range(1,7)
    ] + [
        f"IntensityRelComparison_{i}"
        for i in range(1,7)
    ] + [
        f"LimitActivityAcceleration_{i}"
        for i in range(1,7)
    ]

# file_path = r"D:/Bureau/FI-2010/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_9.txt"


# import numpy as np
# import pandas as pd

# # Define the training and testing days based on the experimental protocol
# train_days = 8
# test_day = 10  # Day 10 for testing

# # Choose the latest available training file
# train_file = f"D:/Bureau/FI-2010/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_9.txt"
# test_file = f"D:/Bureau/FI-2010/Auction/1.Auction_Zscore/Auction_Zscore_Training/Train_Dst_Auction_ZScore_CF_{test_day}.txt"

# # Read the latest training file
# with open(train_file, "r") as file:
#     raw_data = file.read().split()  # Split by whitespace
#     data_array = np.array(raw_data, dtype=float)

# # Ensure the data size is divisible by 144
# usable_elements = (len(data_array) // 144) * 144
# data_array = data_array[:usable_elements]  # Trim extra elements

# # Reshape into rows of 144 features
# reshaped_data = data_array.reshape(-1, 144)

# # Convert to DataFrame
# feature_names = basic_features + time_insensitive_features + time_sensitive_features
# df_train = pd.DataFrame(reshaped_data, columns=feature_names)

# # Print dataset info
# print(f"Training data shape: {df_train.shape}")
# print(df_train.head())

# # Repeat the process for the test file if needed


import numpy as np
import pandas as pd

# Paths to Auction and NoAuction files for CF_8
auction_file = "D:/Bureau/FI-2010/Auction/1.Auction_Zscore/Auction_Zscore_Training/Train_Dst_Auction_ZScore_CF_9.txt"
no_auction_file = "D:/Bureau/FI-2010/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_9.txt"

# Load Auction data
with open(auction_file, "r") as file:
    auction_data = np.array(file.read().split(), dtype=float)

# Reshape Auction data
usable_elements_auction = (len(auction_data) // 144) * 144
auction_data = auction_data[:usable_elements_auction]
reshaped_auction = auction_data.reshape(-1, 144)

# Load NoAuction data
with open(no_auction_file, "r") as file:
    no_auction_data = np.array(file.read().split(), dtype=float)

# Reshape NoAuction data
usable_elements_no_auction = (len(no_auction_data) // 144) * 144
no_auction_data = no_auction_data[:usable_elements_no_auction]
reshaped_no_auction = no_auction_data.reshape(-1, 144)

# Convert to DataFrames
df_auction = pd.DataFrame(reshaped_auction)
df_no_auction = pd.DataFrame(reshaped_no_auction)

# Compute row hashes for Auction and NoAuction data
df_auction["hash"] = df_auction.apply(lambda row: hash(tuple(row)), axis=1)
df_no_auction["hash"] = df_no_auction.apply(lambda row: hash(tuple(row)), axis=1)

# Find common hashes (overlapping rows)
overlapping_hashes = set(df_auction["hash"]).intersection(set(df_no_auction["hash"]))

# Count the number of overlapping rows
print(f"Number of overlapping rows: {len(overlapping_hashes)}")

