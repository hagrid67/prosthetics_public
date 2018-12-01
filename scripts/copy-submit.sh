#!/bin/bash

cd ~/projects/crowdai/prosthetics/main

logfile=cron-submit.log

#exec >> $logfile  2>&1

echo START======================================================>> $logfile 2>&1
date >> $logfile 2>&1

scp -p "am0:projects/crowdai/prosthetics/model-data/ppo-0809b/*" model-data/ppo-0809b/ >> $logfile 2>&1

dateSuffix=`date "+%y%m%d-%H%M"`
sfModel="model-data/ppo-0809b/submit-$dateSuffix"
echo "Save to $sfModel" >> $logfile 2>&1
mkdir -p "$sfModel"
cp -p model-data/ppo-0809b/* "$sfModel" >> $logfile 2>&1

source ~/miniconda3/bin/activate osim361

export PYTHONPATH=/home/jeremy/projects/crowdai/prosthetics/main:/home/jeremy/projects/crowdai/prosthetics/osim-rl-helper

python -m hagrid.prosthetics.runner --model-path-load ./model-data/ppo-0809b/ppo --num-timesteps 10000 --moments ./model-data-keep/Obs-MeanStd.json  --submit --token 50fb66bb6e5f4fdac76d1cb68a5b9038 --url http://grader.crowdai.org:1729 >> $logfile 2>&1

echo date >> $logfile 2>&1
echo END======================================================>> $logfile 2>&1
