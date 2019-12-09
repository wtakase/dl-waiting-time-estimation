#!/bin/sh

sed 's/"//g' $1 | sort -k4 -t';' | awk -v OFS=',' -F';' '{print $4,$5-$4,$2,$7,$5,$6,$3,$9,$8}' > `basename $1 .csv`_preprocessed.csv
