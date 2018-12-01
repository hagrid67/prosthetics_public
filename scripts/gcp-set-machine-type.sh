#!/bin/bash



host=$1
mtype=$2


gcloud compute instances stop $host
gcloud compute instances set-machine-type $host --machine-type $mtype
gcloud compute instances start $host