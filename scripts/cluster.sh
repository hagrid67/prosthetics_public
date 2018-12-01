#!/bin/bash

sHead=$1
shift

ray start --head --redis-port 8379 --num-cpus 48

sleep 5

parallel-ssh -H "$*" -i "cd projects/crowdai/prosthetics/main && source setenv.sh && ray start --redis-address $sHead:8379 --num-cpus 60"

sleep 5

python -m hagrid.prosthetics.rayrun ppo --bTrain --nWorkers 168 --sRedis $sHead:8379 --nHidWidth 128 --bUseLastCheckpoint

