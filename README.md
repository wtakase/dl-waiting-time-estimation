Waiting time estimation in batch system by Deep Learning
====

* Estimate batch system waiting time using Fully-connected neural network.

![dl_model](https://github.com/wtakase/dl-waiting-time-estimation/raw/master/images/dl_model.png "dl_model")

* The network predicts waiting time from 6 classes:
	* 0 to 10 mins,
	* 10 to 30 mins,
	* 30 to 1 hour,	
	* 1 to 3 hours,
	* 3 to 6 hours,
	* more than 6 hours.

## Input data

* Use accounting information of batch system.

### Raw input data

* Raw input data contains the following information:

|Job ID|Username|Number of processors used|Submit time|Start time|End time|Queue|CPU time|Run time|
|------|--------|-------------------------|-----------|----------|--------|-----|--------|--------|
|23561234|user_0|1|1574156230|1574165060|1574270380|queue_0|1024.0|1100|
|23578888|user_1|1|1574169710|1574170780|1574270420|queue_1|64.5|100|

* Example: `accounting_sample.csv`

```
"23561234";"user_0";"1";"1574156230";"1574165060";"1574270380";"queue_0";"1024.0";"1100"
"23578888";"user_1";"1";"1574169710";"1574170780";"1574270420";"queue_1";"64.5";"100"
"23480005";"user_2";"1";"1574111650";"1574133030";"1574290570";"queue_1";"512.8";"750"
"23480800";"user_1";"1";"1574100770";"1574122020";"1574290670";"queue_0";"2048.0";"2500"
. . .
```

### Enriched input data

* Based on the raw input data, the following parameters will be generated:
	* The last observed waiting time (User specific),
	* Submission time of the above (User specific),
	* Number of cores currently in use (User specific),
	* Run time currently spent (User specific),
	* Number of waiting jobs (User specific),
	* Day of submission (Job specific),
	* Submission time in 24 hours (Job specific),
	* Queue (Job specific).

## Usage

* Prepare your accounting information such as `accounting_sample.csv`:

```
$ ls
00_preprocess_accounting.sh          04_merge_and_split_dataset.py
01_create_userlist_and_queuelist.sh  05_train.py
02_split_accounting.sh               accounting_sample.csv
03_enrich_accounting.py
```

0. Preprocess your acconting information:

```
$ ./00_preprocess_accounting.sh accounting_sample.csv
```

1. Create `userlist` and `queuelist`:

```
$ ./01_create_userlist_and_queuelist.sh accounting_sample_preprocessed.csv 
```

2. Split accounting information for each user:

```
$ ./02_split_accounting.sh accounting_sample_preprocessed.csv
```

3. Enrich accounting information:

```
$ ./03_enrich_accounting.py
```

* It will take some time.

4. Merge data and split it to `train_test_dataset.csv` and `validation_dataset.csv`:

```
$ ./04_merge_and_split_dataset.py
```

5. Train:

```
$ ./05_train.py --gpu --n-epoch 5 --batch-size 100 --standard-scale --log-scale --weights-init variance_scaling --optimizer adam --learning-rate 0.001 --use-using-cores --use-spending-run-time --use-pending-jobs --use-last-pend-time-submit --use-queue --use-submit-time --use-day-of-week
```

* Output:

```
##########
csv_file, train_test_dataset.csv
optimizer, adam
weights_init, variance_scaling
learning_rate, 0.001
n_epoch, 5
batch_size, 100
train_size, 0.8
gpu, True
log_scale, True
lr_decay, 0.96
decay_epoch, 1
decay_step, -1
use_using_cores, True
use_spending_run_time, True
use_pending_jobs, True
use_last_pend_time_submit, True
use_queue, True
use_submit_time, True
use_day_of_week, True
model, 
without_save, False
skipfooter, 0
check_per_class_accuracy, False
batch_norm, False
standard_scale, True
predict_num, 1000
layers, [['fc', 128, 'relu', 0.8], ['fc', 128, 'relu', 0.8], ['fc', 128, 'relu', 0.8]]
timestamp, 1912091006
##########
* Check imbalancing
3, 163722, 0.295424
0, 104752, 0.189017
4, 93067, 0.167932
1, 72247, 0.130364
2, 70100, 0.126490
5, 50306, 0.090773
* Balanced data
5, 50306, 0.166667
4, 50306, 0.166667
3, 50306, 0.166667
2, 50306, 0.166667
1, 50306, 0.166667
0, 50306, 0.166667
Epoch, 0, Train acc, 0.172000, Test acc, 0.154000
Epoch, 1, Train acc, 0.771000, Test acc, 0.777000
Epoch, 2, Train acc, 0.785000, Test acc, 0.798000
Epoch, 3, Train acc, 0.837000, Test acc, 0.828000
Epoch, 4, Train acc, 0.831000, Test acc, 0.832000
Epoch, 5, Train acc, 0.828000, Test acc, 0.853000
Elapsed time, 178.022552
```

## Tuning

* You can change `optimizer`, `weights init`, `learning rate`, `number of epochs` and `batch size` by options of `05_train.py`.

* You can change the network by changing `LAYERS` variable in `05_train.py`.
	* The following means:
		* 1st layer has `128` neurons and activation function is `relu` and dropout ratio is `0.8`,
		* 2nd layer has `128` neurons and activation function is `relu` and dropout ratio is `0.8`,
		* 3rd layer has `128` neurons and activation function is `relu` and dropout ratio is `0.8`.

```
LAYERS = [["fc", 128, 'relu', 0.8],
          ["fc", 128, 'relu', 0.8],
          ["fc", 128, 'relu', 0.8]]
```

* You can select which input parameters to use by `--use--xxxxx` options of `05_train.py`.
