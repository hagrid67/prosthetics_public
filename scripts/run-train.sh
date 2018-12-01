#!/bin/bash

# --moments ./model-data-keep/Obs-MeanStd.json - now removed
# -np 15 worked well
#mpirun --mca btl ^openib -np 15 

nCPU=`lscpu | grep "CPU(s)" | head -1 | awk '{print $2}'`

sDate=`date +%y%m%d`

echo "nCPU: $nCPU date: $sDate"

# gc1: 4 cpus,  actorbatch=256. using weights from am0.
# gc2: 16 cpus, actorbatch=256. using weights from am0.
# gc3: 32 cpus, actorbatch=256 (from orig 2048)
# gc4: 32 cpus, actorbatch=128
# gc5: 32 cpus, actorbatch=128, hid_size=256 (from 64)

c="mpirun --mca btl ^openib -np $nCPU"
c="$c python -m hagrid.prosthetics.runner"
c="$c --model-path-save=./model-data/ppo-$sDate/ppo --num-timesteps 2e7"
c="$c --train --moments ./model-data-keep/Obs-MeanStd.json"
c="$c --hid-size=256" # run this on gc5
c="$c --ts-per-batch 128" # run on gc4

$c

#--model-path-load=./model-data/ppo-0809b/ppo
