# -*- coding: utf-8 -*-
"""
Split the clean dataset into train (2021-2023) and test (2024+).
Run once: python split_data.py
"""
import pandas as pd

df = pd.read_csv('data/train_data_clean.csv', index_col='timestamp', parse_dates=True)
df = df[~df.index.duplicated(keep='first')]

train = df.loc['2021':'2023']
test  = df.loc['2024':]

train.to_csv('data/train_2021_2023.csv')
test.to_csv('data/test_2024_onwards.csv')

print(f"Train: {len(train):,} rows  ({train.index[0].date()} → {train.index[-1].date()})")
print(f"Test:  {len(test):,} rows  ({test.index[0].date()} → {test.index[-1].date()})")