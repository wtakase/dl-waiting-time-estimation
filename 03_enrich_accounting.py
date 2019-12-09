#!/usr/bin/env python

import csv
import datetime
import numpy as np
import pandas as pd
import time
import sys
import os

users = []
with open("userlist") as fd:
    for line in fd:
        users.append(line.strip())

for i, user in enumerate(users):
    print("%04d/%04d: %s" % (i + 1, len(users), user), flush=True)
    raw_csv = "user_csv/accounting_raw_%s.csv" % user
    if not os.path.exists(raw_csv):
        continue
    user_df = pd.read_csv(
        raw_csv, sep="\s*,\s*", engine="python")
        #parse_dates=["submit_time", "start_time", "end_time"])
    user_df = user_df[user_df["pend_time"] >= 0]
    user_df.index = range(len(user_df))

    ext_cols = []
    for index, row in user_df.iterrows():
        running_df = user_df[
            (user_df["end_time"] > row["submit_time"]) & \
            (user_df["start_time"] < row["submit_time"])]
        using_cores = running_df["num_processors"].sum()

        spending_run_time = (
            row["submit_time"] - running_df["start_time"]).sum()

        pending_df = user_df[
            (user_df["submit_time"] < row["submit_time"]) & \
            (user_df["start_time"] > row["submit_time"])]
        pending_jobs = pending_df.shape[0]

        past_pend_time_df = user_df[
            (user_df["submit_time"] < row["submit_time"]) & \
            (user_df["start_time"] < row["submit_time"])]
        last_pend_time_df = past_pend_time_df.tail(1)
        last_pend_time = -1
        last_pend_time_submit = row["submit_time"] - row["submit_time"]
        if len(last_pend_time_df > 0):
            last_pend_time = last_pend_time_df["pend_time"].values[0]
            last_pend_time_submit = \
                row["submit_time"] - last_pend_time_df["submit_time"].values[0]

        ext_cols.append([using_cores,
                         spending_run_time,
                         pending_jobs,
                         last_pend_time,
                         last_pend_time_submit])

    ext_cols_df = pd.DataFrame(ext_cols, columns=["using_cores",
                                                  "spending_run_time",
                                                  "pending_jobs",
                                                  "last_pend_time",
                                                  "last_pend_time_submit"])
    ext_cols_df.index = range(len(ext_cols_df))
    #ext_cols_df["spending_run_time"] = (
    #    ext_cols_df["spending_run_time"] / np.timedelta64(1, "s")).astype(int)
    #try:
    #    ext_cols_df["last_pend_time_submit"] = \
    #        ext_cols_df["last_pend_time_submit"].dt.total_seconds()
    #except:
    #    print("Failed user: %s" % user)
    #    continue
    user_df = pd.concat([user_df, ext_cols_df], axis=1)

    user_df.drop(
        ["start_time", "end_time", "num_processors", "run_time", "cpu_time"],
        1, inplace=True)

    user_csv_file = "user_csv/accounting_%s.csv" % user
    user_df.to_csv(user_csv_file, sep=",", index=None)

print("Done")
