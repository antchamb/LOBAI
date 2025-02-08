# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:09:03 2025

@author: dell
"""

import pandas as pd
import numpy as np

path = r'D:/Bureau/LOBAI/FI-2010/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_2.txt'

with open(path, "r", encoding="utf-8") as f:
    raw_data = f.read()

# Split values by whitespace
values = np.array(raw_data.split(), dtype=float)  # Convert to numeric array

print("Number of values:", len(values)) 