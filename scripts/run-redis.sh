#!/bin/bash

sDate=`date +%m%d`

bRun=false

if [ $1 = "run" ]
then
    echo "Running the command..."
    shift
    bRun=true
fi


if [ "$1" = train ]
then
    echo Training command:
    c="python -m hagrid.prosthetics.runner"
    #c="$c --model-path-load ../model-data/gc5/ppo-180819/ppo" #"../model-data/ppo-0919-test "
    c="$c --model-path-save ../model-data/ppo-$sDate/ppo"
    c="$c --num-timesteps 1e7 --moments ./model-data-keep/Obs-MeanStd.json"
    c="$c --train --hid-size 256 "
    c="$c --ts-per-batch 256"  # should this be 64 x cpus?
    c="$c --ts-per-minibatch 64" # was 256
    c="$c --model-redis-save tf:ppo1"

    d=$c

    echo $d

    if [ $bRun = true ]
    then
        echo Running:
        $d
    fi


elif [ "$1" = agents ]
then
    
    nAgents=$2
    echo Command for $nAgents agents:

    c="python -m hagrid.prosthetics.runner"
    c="$c --model-redis-load tf:ppo1"
    c="$c --num-timesteps 1e7" 
    c="$c --moments ./model-data-keep/Obs-MeanStd.json"
    c="$c --play"
    
    #c="$c --nMaxSegSteps 256"
    c="$c --nMaxSegSteps 512" # in reality we never do more than 300 steps with prosthetics
    c="$c --hid-size 256 --step-timeout 10"
    c="$c --integrator-accuracy 5e-5" # 0.0005"
    

    for iAgent in $(seq 1 $nAgents)
    do
        d="$c --nMaxSegTime `expr 300 + $iAgent \* 5`"
        d="$d --seed `expr $iAgent \* 23 `"

        echo $d

        if [ $bRun = true ]
        then
            $d &
            sleep 5
        fi

    done

elif [ "$1" = kill ]
then

    echo "Killing..."
    ps -ef | grep -i python | grep -i "play" 
    ps -ef | grep -i python | grep -i "play" | awk '{print $2}' | xargs kill

fi


