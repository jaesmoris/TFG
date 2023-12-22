import os
import numpy as np
import pandas as pd

csv_path = "/home/msiau/workspace/asdf/indices.csv"
dataset_path = "/home/msiau/data/tmp/jesmoris/dummy_dataset_csv"

df = pd.read_csv(csv_path, sep=",")
print(df.columns)
print(df.values[0:5])
for i in range(5):
    print(f"fold {i}")
    test_indices = df.values[df['KFOLD']==i]
    print(len(test_indices))
