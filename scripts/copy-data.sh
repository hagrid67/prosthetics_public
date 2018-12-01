#!/bin/bash

cd ~/projects/crowdai/prosthetics/main

#exec >> cron-copy.log  2>&1
function redlog() { echo "$(tput setaf 1)$*$(tput sgr 0)"; }
function greenlog() { echo "$(tput setaf 2)$*$(tput sgr 0)"; }

export TERM=screen

greenlog START============================================= >> cron-copy.log
date >> cron-copy.log

#scp -p "am0:projects/crowdai/prosthetics/model-data/ppo-0809b/*" model-data/ppo-0809b/ >> cron-copy.log 2>&1

remdir=projects/crowdai/prosthetics/model-data
idFile=~/.ssh/google_compute_engine
options="-o ConnectTimeout=1"
#for host in gc1 gc2 gc3 gc4 gc5
#for host in gc4 gc5
for host in gc2
do
    
    dir=../model-data/$host
    greenlog $host $dir >> cron-copy.log 2>&1
    #mkdir -p $dir $dir/ppo-0818 $dir/ppo-0819 >> cron-copy.log 2>&1
    mkdir -p $dir $dir/ppo-0924  >> cron-copy.log 2>&1
    mkdir -p $dir $dir/ppo-0925  >> cron-copy.log 2>&1
    scp $options -i $idFile -p $host:$remdir/eplog* $dir  >> cron-copy.log 2>&1
    scp $options -i $idFile -p $host:$remdir/ppo-0924/* $dir/ppo-0924  >> cron-copy.log 2>&1
    #scp $options -i $idFile -p $host:$remdir/ppo-180818/* $dir/ppo-180818  >> cron-copy.log 2>&1

done

date >> cron-copy.log
greenlog END============================================= >> cron-copy.log
