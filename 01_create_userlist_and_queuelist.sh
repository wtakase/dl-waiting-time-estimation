#!/bin/sh

awk -F',' '{print $3}' $1 | sort | uniq > userlist
awk -F',' '{print $4}' $1 | sort | uniq > queuelist
