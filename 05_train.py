#!/usr/bin/env python

import argparse
import datetime
import numpy as np
import math
import os
import pandas as pd
import random
import sys
import tensorflow as tf
import tflearn
import time

from contextlib import contextmanager
from math import sqrt
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from tensorflow import reset_default_graph

TIMESTAMP = datetime.datetime.now().strftime("%y%m%d%H%M")

CSV_FILE = "train_test_dataset.csv"
TRAIN_SIZE = 0.8
LAYERS = [["fc", 128, 'relu', 0.8],
          ["fc", 128, 'relu', 0.8],
          ["fc", 128, 'relu', 0.8]]
OPTIMIZER = "adam"
WEIGHTS_INIT = "xavier"
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
N_EPOCH = 5
N_REPORT = 100
LR_DECAY=0.96
DECAY_EPOCH=1
CLASS_NUM = 6
SKIPFOOTER = 0

QUEUES = []
with open("queuelist") as fd:
    for line in fd:
        QUEUES.append(line.strip())
QUEUES = sorted(QUEUES)
QUEUE_NUM = len(QUEUES)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Use CPU by default
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# If --gpu option is enabled, use the following devices
CUDA_VISIBLE_DEVICES = "0"

pd.set_option("display.max_columns", None)


class Params:
    def __init__(
          self, csv_file=CSV_FILE, layers=LAYERS, optimizer=OPTIMIZER,
          weights_init=WEIGHTS_INIT, learning_rate=LEARNING_RATE,
          n_epoch=N_EPOCH, batch_size=BATCH_SIZE, train_size=TRAIN_SIZE,
          log_scale=False, lr_decay=LR_DECAY, decay_epoch=DECAY_EPOCH,
          decay_step=-1, use_using_cores=False, use_spending_run_time=False,
          use_pending_jobs=False, use_last_pend_time_submit=False,
          use_queue=False, use_submit_time=False, use_day_of_week=False,
          model_file="", without_save=False, check_per_class_accuracy=False,
          batch_norm=False, class_num=CLASS_NUM, standard_scale=False,
          predict_num=1000, skipfooter=SKIPFOOTER, queue_num=QUEUE_NUM):
        self.csv_file = csv_file
        self.layers = layers
        self.optimizer = optimizer
        self.weights_init = weights_init
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.train_size = train_size
        self.log_scale = log_scale
        self.lr_decay = lr_decay
        self.decay_epoch = decay_epoch
        self.decay_step = decay_step
        self.use_using_cores = use_using_cores
        self.use_spending_run_time = use_spending_run_time
        self.use_pending_jobs = use_pending_jobs
        self.use_last_pend_time_submit = use_last_pend_time_submit
        self.use_queue = use_queue
        self.use_submit_time = use_submit_time
        self.use_day_of_week = use_day_of_week
        self.model_file = model_file
        self.without_save = without_save
        self.check_per_class_accuracy = check_per_class_accuracy
        self.batch_norm = batch_norm
        self.class_num = class_num
        self.standard_scale = standard_scale
        self.predict_num = predict_num
        self.skipfooter = skipfooter
        self.queue_num = queue_num


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def load_data(p, verbose=True):
    timestamp_parser = lambda timestamp: \
        datetime.datetime.fromtimestamp(float(timestamp))
    df = pd.read_csv(p.csv_file, sep="\s*,\s*",
                     engine="python", skipfooter=p.skipfooter,
                     parse_dates=["submit_time"], date_parser=timestamp_parser)

    df.loc[df["last_pend_time"] == -1, "last_pend_time"] = 0

    df.drop(["user"], 1, inplace=True)
    df.reindex(
        columns=["pend_time", "using_cores", "spending_run_time",
                 "pending_jobs", "last_pend_time", "last_pend_time_submit",
                 "queue", "submit_time"])

    input_num = 1
    if p.use_using_cores:
        input_num += 1
    else:
        df.drop(["using_cores"], 1, inplace=True)

    if p.use_spending_run_time:
        input_num += 1
    else:
        df.drop(["spending_run_time"], 1, inplace=True)

    if p.use_pending_jobs:
        input_num += 1
    else:
        df.drop(["pending_jobs"], 1, inplace=True)

    if p.use_last_pend_time_submit:
        input_num += 1
    else:
        df.drop(["last_pend_time_submit"], 1, inplace=True)

    if p.use_queue:
        """
        le = LabelEncoder()
        queues = df["queue"].unique()
        p.queue_num = len(queues)
        le.fit(queues)
        df["queue"] = le.fit_transform(df["queue"])
        df = pd.concat(
            [df, pd.get_dummies(df["queue"], prefix="queue")], axis=1)
        print("* Queues")
        print(queues)
        """
        le = LabelEncoder()
        le.fit(QUEUES)
        df["queue"] = le.transform(df["queue"])
        queues = ["%d" % i for i in range(p.queue_num)]
        dummies = pd.get_dummies(df["queue"], prefix="", prefix_sep="")
        dummies = dummies.T.reindex(queues).T.fillna(0)
        queues_label = ["queue_%d" % i for i in range(p.queue_num)]
        dummies.columns = queues_label
        df = pd.concat([df, dummies], axis=1)
        #print("* Queues")
        #print(QUEUES)

        input_num += p.queue_num
    df.drop(["queue"], 1, inplace=True)

    if p.use_day_of_week:
        df["day_of_week"] = df["submit_time"].dt.dayofweek
        df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df.drop("day_of_week", 1, inplace=True)
        input_num += 2

    if p.use_submit_time:
        submit_time_h = df["submit_time"].dt.hour * 3600
        submit_time_m = df["submit_time"].dt.minute * 60
        submit_time_s = df["submit_time"].dt.second
        df["submit_time"] = submit_time_h + submit_time_m + submit_time_s
        df["sin_submit_time"] = np.sin(
            2 * np.pi * df["submit_time"] / 86400)
        df["cos_submit_time"] = np.cos(
            2 * np.pi * df["submit_time"] / 86400)
        input_num += 2
    df.drop(["submit_time"], 1, inplace=True)

    df.loc[df["pend_time"] < 600, "pend_time"] = 0
    df.loc[
        (600 <= df["pend_time"]) & \
        (df["pend_time"] < 1800), "pend_time"] = 1
    df.loc[
        (1800 <= df["pend_time"]) & \
        (df["pend_time"] < 3600), "pend_time"] = 2
    df.loc[
        (3600 <= df["pend_time"]) & \
        (df["pend_time"] < 10800), "pend_time"] = 3
    df.loc[
        (10800 <= df["pend_time"]) & \
        (df["pend_time"] < 21600), "pend_time"] = 4
    df.loc[21600 <= df["pend_time"], "pend_time"] = 5
    #print(df.head(5))

    if verbose:
        print("* Check imbalancing", flush=True)
    total_rows = df.shape[0]
    min_rows = total_rows
    for index, value in df["pend_time"].value_counts().iteritems():
        if verbose:
            print("%s, %d, %f" % (
                index, value, value / total_rows), flush=True)
        if value < min_rows:
            min_rows = value
    dfs = []
    for i in range(p.class_num):
        dfs.append(df[df["pend_time"] == i].sample(n=min_rows))
    df = pd.concat(dfs, axis=0)

    if verbose:
        print("* Balanced data", flush=True)
    for index, value in df["pend_time"].value_counts().iteritems():
        if verbose:
            print("%s, %d, %f" % (
                index, value, value / (min_rows * p.class_num)), flush=True)

    df = pd.concat(
        [df, pd.get_dummies(df["pend_time"], prefix="pend_time")], axis=1)
    df.drop(["pend_time"], 1, inplace=True)
    #print(df.tail(5))

    df = df.sample(frac=1).reset_index(drop=True)
    #print("* Input data")
    #print(df.head(5))

    return df, input_num


def split_data(df, train_size):
    pos = round(len(df) * train_size)
    return df[:pos], df[pos:]


def scale_data(df, p, train=True, save=True):
    if p.log_scale:
        df.loc[df["last_pend_time"] == 0, "last_pend_time"] = 1
        if train:
            log_scaler = FunctionTransformer(np.log2)
            df.loc[:, ["last_pend_time"]] = log_scaler.fit_transform(
                df[["last_pend_time"]])
            if save:
                joblib.dump(log_scaler, "log_scaler.save")
        else:
            log_scaler = joblib.load("log_scaler.save")
            df.loc[:, ["last_pend_time"]] = log_scaler.transform(
                df[["last_pend_time"]])

    scale_cols = ["last_pend_time"]
    if p.use_using_cores:
        scale_cols.append("using_cores")
    if p.use_spending_run_time:
        scale_cols.append("spending_run_time")
    if p.use_pending_jobs:
        scale_cols.append("pending_jobs")
    if p.use_last_pend_time_submit:
        scale_cols.append("last_pend_time_submit")
    if p.use_submit_time:
        scale_cols.append("sin_submit_time")
        scale_cols.append("cos_submit_time")
    if p.use_day_of_week:
        scale_cols.append("sin_day_of_week")
        scale_cols.append("cos_day_of_week")

    if train:
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        df.loc[:, scale_cols] = min_max_scaler.fit_transform(df[scale_cols])
        if save:
            joblib.dump(min_max_scaler, "min_max_scaler.save")
    else:
        min_max_scaler = joblib.load("min_max_scaler.save")
        df.loc[:, scale_cols] = min_max_scaler.transform(df[scale_cols])

    if p.standard_scale:
        if train:
            standard_scaler = StandardScaler()
            df.loc[:, scale_cols] = standard_scaler.fit_transform(
                df[scale_cols])
            if save:
                joblib.dump(standard_scaler, "standard_scaler.save")
        else:
            standard_scaler = joblib.load("standard_scaler.save")
            df.loc[:, scale_cols] = standard_scaler.transform(df[scale_cols])

    return df


def convert_to_numpy(df, p):
    input_cols = []
    if p.use_using_cores:
        input_cols.append("using_cores")
    if p.use_spending_run_time:
        input_cols.append("spending_run_time")
    if p.use_pending_jobs:
        input_cols.append("pending_jobs")
    input_cols.append("last_pend_time")
    if p.use_last_pend_time_submit:
        input_cols.append("last_pend_time_submit")
    if p.use_queue:
        for i in range(p.queue_num):
            input_cols.append("queue_%d" % i)
    if p.use_submit_time:
        input_cols.append("sin_submit_time")
        input_cols.append("cos_submit_time")
    if p.use_day_of_week:
        input_cols.append("sin_day_of_week")
        input_cols.append("cos_day_of_week")
    
    df_x = df[input_cols]
    pend_time_cols = ["pend_time_%d" % i for i in range(p.class_num)]
    df_t = df[pend_time_cols]
    return df_x.values, df_t.values.reshape(-1, p.class_num)


def build_network(p, input_num):
    network = tflearn.input_data(shape=[None, input_num])
    for layer in p.layers:
        network = tflearn.fully_connected(network, layer[1],
                                          activation=layer[2],
                                          weights_init=p.weights_init)
        if p.batch_norm:
            network = tflearn.batch_normalization(network)
        if layer[3] < 1.0:
            network = tflearn.dropout(network, layer[3])
 
    network = tflearn.fully_connected(
        network, p.class_num, activation="softmax")

    if p.optimizer == "sgd":
        optimizer = tflearn.SGD(learning_rate=p.learning_rate,
                                lr_decay=p.lr_decay, decay_step=p.decay_step)
    elif p.optimizer == "momentum":
        optimizer = tflearn.Momentum(learning_rate=p.learning_rate,
                                     lr_decay=p.lr_decay,
                                     decay_step=p.decay_step)
    elif p.optimizer == "adagrad":
        optimizer = tflearn.AdaGrad(learning_rate=p.learning_rate)
    else:
        optimizer = tflearn.Adam(learning_rate=p.learning_rate)
    network = tflearn.regression(network, optimizer=optimizer,
                                 loss="categorical_crossentropy")
    return network


def train(p):
    df, input_num = load_data(p)
    df_train, df_test = split_data(df, p.train_size)

    #print(df.head())
    df_train = scale_data(df_train, p, train=True)
    #print(df_train.head())
    df_test = scale_data(df_test, p, train=False)
    #sys.exit(1)

    train_x, train_t = convert_to_numpy(df_train, p)
    test_x, test_t = convert_to_numpy(df_test, p)

    if p.decay_step < 0:
        train_size = train_x.shape[0]
        steps_per_epoch = train_size // p.batch_size
        p.decay_step = p.decay_epoch * steps_per_epoch

    reset_default_graph()
    network = build_network(p, input_num)
    model = tflearn.DNN(network, tensorboard_verbose=0)

    epoch = 0
    if p.model_file != "" and p.model_file.endswith(".tflearn"):
        model.load(p.model_file)
        epoch += int(os.path.basename(p.model_file).split(".")[0])

    train_accuracy = []
    test_accuracy = []
    test_per_class_accuracy = []
    max_test_accuracy = 0
    for i in range(p.n_epoch + 1):
        if i > 0:
            with suppress_stdout():
                model.fit(train_x, train_t,
                          batch_size=p.batch_size, n_epoch=1)

        train_batch_mask = np.random.choice(train_x.shape[0], p.predict_num)
        test_batch_mask = np.random.choice(test_x.shape[0], p.predict_num)

        train_batch_x = train_x[train_batch_mask]
        test_batch_x = test_x[test_batch_mask]
        train_y = np.array(model.predict(train_batch_x)).argmax(axis=1)
        test_y = np.array(model.predict(test_batch_x)).argmax(axis=1)
   
        train_accuracy.append(
            np.mean(train_t[train_batch_mask].argmax(axis=1) == train_y))
        test_accuracy.append(
            np.mean(test_t[test_batch_mask].argmax(axis=1) == test_y))

        if not p.without_save and epoch >= 10:
            if test_accuracy[-1] > max_test_accuracy:
                max_test_accuracy = test_accuracy[-1]
                model_dir = "model/%s" % TIMESTAMP
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                model.save("%s/%d.tflearn" % (model_dir, epoch))

        print("Epoch, %d, Train acc, %f, Test acc, %f" % (
            epoch, train_accuracy[-1], test_accuracy[-1]), flush=True)

        if p.check_per_class_accuracy:
            per_class_accuracy = []
            per_class_incorrect = ""
            for i in range(p.class_num):
                batch_t = test_t[test_batch_mask]
                class_index = np.where(batch_t.argmax(axis=1) == i)
                total = class_index[0].shape[0]
                correct = np.sum(
                    batch_t[class_index].argmax(axis=1) == test_y[class_index])
                per_class_incorrect += "%d, " % i
                for j in range(p.class_num):
                    incorrect_index = np.where(test_y[class_index] == j)
                    per_class_incorrect += "%5d" % incorrect_index[0].shape[0]
                    if j != p.class_num - 1:
                        per_class_incorrect += ", "
                    elif i != p.class_num - 1:
                        per_class_incorrect += "\n"
                per_class_accuracy.append("%d/%d (%.3f)" % (
                    correct, total, correct / total))
            per_class_accuracy_str = "Epoch, %d, %s\n%s" % (
                epoch, ", ".join(map(str, per_class_accuracy)),
                per_class_incorrect)
            test_per_class_accuracy.append(per_class_accuracy_str)

        epoch += 1

    if p.check_per_class_accuracy:
        print("* Per class accuracy")
        for per_class_accuracy_str in test_per_class_accuracy:
            print(per_class_accuracy_str)


def print_args(args, show_layers=True, show_layer_num=True):
    print("##########", flush=True)
    for arg in vars(args):
        if not show_layer_num and arg == "layer_num":
            continue
        print("%s, %s" % (arg, getattr(args, arg)), flush=True)
    if show_layers:
        print("layers, %s" % LAYERS, flush=True)
    print("timestamp, %s" % TIMESTAMP, flush=True)
    print("##########", flush=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Trainer")
    argparser.add_argument(
        "-c", "--csv-file", default=CSV_FILE, help="[%s]" % CSV_FILE)
    argparser.add_argument(
        "-o", "--optimizer", default=OPTIMIZER, help="[%s]" % OPTIMIZER)
    argparser.add_argument(
        "-w", "--weights-init", default=WEIGHTS_INIT,
        help="[%s]" % WEIGHTS_INIT)
    argparser.add_argument(
        "-l", "--learning-rate", default=LEARNING_RATE, type=float,
        help="[%s]" % LEARNING_RATE)
    argparser.add_argument(
        "-e", "--n-epoch", default=N_EPOCH, type=int, help="[%s]" % N_EPOCH)
    argparser.add_argument(
        "-b", "--batch-size", default=BATCH_SIZE, type=int,
        help="[%s]" % BATCH_SIZE)
    argparser.add_argument(
        "-t", "--train-size", default=TRAIN_SIZE, type=float,
        help="[%s]" % TRAIN_SIZE)
    argparser.add_argument("-g", "--gpu", default=False, action="store_true")
    argparser.add_argument("--log-scale", default=False, action="store_true")
    argparser.add_argument(
        "--lr-decay", default=LR_DECAY, type=float, help="[%s]" % LR_DECAY)
    argparser.add_argument(
        "--decay-epoch", default=DECAY_EPOCH, type=int,
        help="[%s]" % DECAY_EPOCH)
    argparser.add_argument("--decay-step", default=-1, type=int, help="[-1]")
    argparser.add_argument(
        "--use-using-cores", default=False, action="store_true")
    argparser.add_argument(
        "--use-spending-run-time", default=False, action="store_true")
    argparser.add_argument(
        "--use-pending-jobs", default=False, action="store_true")
    argparser.add_argument(
        "--use-last-pend-time-submit", default=False, action="store_true")
    argparser.add_argument("--use-queue", default=False, action="store_true")
    argparser.add_argument(
        "--use-submit-time", default=False, action="store_true")
    argparser.add_argument(
        "--use-day-of-week", default=False, action="store_true")
    argparser.add_argument("--model", default="", help="[]")
    argparser.add_argument(
        "--without-save", default=False, action="store_true")
    argparser.add_argument(
        "--skipfooter", default=SKIPFOOTER, type=int, help="[%s]" % SKIPFOOTER)
    argparser.add_argument(
        "-v", "--check-per-class-accuracy", default=False, action="store_true")
    argparser.add_argument("--batch-norm", default=False, action="store_true")
    argparser.add_argument(
        "--standard-scale", default=False, action="store_true")
    argparser.add_argument("--predict-num", default=1000,
                           type=int, help="[1000]")
    args = argparser.parse_args()

    print_args(args)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    start_time = time.time()
    p = Params(
          csv_file=args.csv_file, layers=LAYERS, optimizer=args.optimizer,
          weights_init=args.weights_init, learning_rate=args.learning_rate,
          n_epoch=args.n_epoch, batch_size=args.batch_size,
          train_size=args.train_size, log_scale=args.log_scale,
          lr_decay=args.lr_decay, decay_epoch=args.decay_epoch,
          decay_step=args.decay_step, use_using_cores=args.use_using_cores,
          use_spending_run_time=args.use_spending_run_time,
          use_pending_jobs=args.use_pending_jobs,
          use_last_pend_time_submit=args.use_last_pend_time_submit,
          use_queue=args.use_queue, use_submit_time=args.use_submit_time,
          use_day_of_week=args.use_day_of_week,
          model_file=args.model, without_save=args.without_save,
          check_per_class_accuracy=args.check_per_class_accuracy,
          batch_norm=args.batch_norm, class_num=CLASS_NUM,
          standard_scale=args.standard_scale, predict_num=args.predict_num,
          skipfooter=args.skipfooter, queue_num=QUEUE_NUM)
    train(p)
    end_time = time.time()
    print("Elapsed time, %f" % (end_time - start_time), flush=True)
