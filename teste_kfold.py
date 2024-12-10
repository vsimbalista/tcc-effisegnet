# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:21:14 2024
@author: vitor
"""
from sklearn.model_selection import KFold, train_test_split
import numpy as np

total_indices = np.zeros(414)
train_val_indices, test_indices = train_test_split(total_indices, test_size=0.1, random_state=42)

print(train_val_indices.shape, test_indices.shape)
print()

kf = KFold(n_splits=9, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
    print(train_idx.shape, val_idx.shape)
    
