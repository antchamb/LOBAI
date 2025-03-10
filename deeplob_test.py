# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:45:40 2025

@author: dell
"""

import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

train_fi = np.loadtxt(r'D:\Bureau\FI-2010\NoAuction\3.NoAuction_DecPre\NoAuction_DecPre_Training\Train_Dst_NoAuction_DecPre_CF_7.txt')

test_fi_1 = np.loadtxt(r'D:\Bureau\FI-2010\NoAuction\3.NoAuction_DecPre\NoAuction_DecPre_Testing\Test_Dst_NoAuction_DecPre_CF_6.txt')
test_fi_2 = np.loadtxt(r'D:\Bureau\FI-2010\NoAuction\3.NoAuction_DecPre\NoAuction_DecPre_Testing\Test_Dst_NoAuction_DecPre_CF_7.txt')
test_fi_3 = np.loadtxt(r'D:\Bureau\FI-2010\NoAuction\3.NoAuction_DecPre\NoAuction_DecPre_Testing\Test_Dst_NoAuction_DecPre_CF_8.txt')
test_fi_4 = np.loadtxt(r'D:\Bureau\FI-2010\NoAuction\3.NoAuction_DecPre\NoAuction_DecPre_Testing\Test_Dst_NoAuction_DecPre_CF_9.txt')

test_fi = np.hstack((test_fi_1, test_fi_2, test_fi_3, test_fi_4))


def extract_x_y_data(data, timestamp_per_sample):
    data_x = np.array(data[:40, :].T)
    data_y = np.array(data[-5:, :].T)
    [N, P_x] = data_x.shape
#     P_y = data_y.shape[1]
    
    x = np.zeros([(N-timestamp_per_sample+1), timestamp_per_sample, P_x])
    
    for i in tqdm(range(N-timestamp_per_sample+1)):
        x[i] = data_x[i:(i+timestamp_per_sample), :]
        
    x = x.reshape(x.shape + (1,))
        
    y = data_y[(timestamp_per_sample-1):]
    y = y[:,3] - 1
    y = to_categorical(y, 3)
    return x, y
# # columns = {"AskPrice_i", "AskVolume_i", "BidPrice_i", "BidVolume_i"} for i = 1 to 10
# data_x_train = np.array(train_fi[:40, :].T)

# data_y_train = np.array(train_fi[-5:, :].T)
# # test = np.cov(data_y_train, rowvar=False) # matrice variance covariance labels

# [N_train, P_x_train] = data_x_train.shape

# x_train = np.zeros([(N_train - 101), 100, P_x_train])


# for i in tqdm(range(N_train - 101)):
#     x_train[i] = data_x_train[i: (i + 100), :]


# x_train_test = x_train.reshape(x_train.shape + (1,)) # final x from extract function


# y_train = data_y_train[(100-1):]
# y_train = y_train[:, 3] - 1
# y_train = to_categorical(y_train, 3) # final y from extract function

train_fi_x, train_fi_y = extract_x_y_data(train_fi, timestamp_per_sample=100)
test_fi_x, test_fi_y = extract_x_y_data(test_fi, timestamp_per_sample=100)
