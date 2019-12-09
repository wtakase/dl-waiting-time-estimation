#!/usr/bin/env python

import datetime
import glob
import os
import pandas as pd
import time

TRAIN_TEST_SIZE = 0.8

out_train_test_csv_file = "train_test_dataset.csv"
out_val_csv_file = "validation_dataset.csv"

in_csv_files = []
with open("userlist") as fd:
    for i, line in enumerate(fd):
        in_csv_file = "user_csv/accounting_%s.csv" % line.strip()
        if os.path.exists(in_csv_file):
            in_csv_files.append(in_csv_file)

dfs = []
start_time = time.time()
for in_csv_file in in_csv_files:
    dfs.append(
        pd.read_csv(in_csv_file, sep="\s*,\s*", engine="python"))
print("Done: read_csv, %f" % (time.time() - start_time), flush=True)

df = pd.concat(dfs)
df.index = range(len(df))
print("Done: pd.concat", flush=True)
total_rows = df.shape[0]
print("Total %d rows" % df.shape[0], flush=True)

df = df.sample(frac=1)
df.index = range(len(df))
pos = round(len(df) * TRAIN_TEST_SIZE)
df_train_test = df.loc[:pos, :]
df_val = df.loc[pos:, :]
df_train_test.to_csv(out_train_test_csv_file, sep=",", index=None)
df_val.to_csv(out_val_csv_file, sep=",", index=None)

print("Done")
