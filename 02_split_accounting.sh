#!/bin/sh

if [ ! -d user_csv ]; then
  mkdir user_csv
fi

for user in `cat userlist`; do
  echo $user
  out_csv_file="user_csv/accounting_raw_$user.csv"
  echo "submit_time,pend_time,user,queue,start_time,end_time,num_processors,run_time,cpu_time" > $out_csv_file
  grep $user $1 >> $out_csv_file
done
