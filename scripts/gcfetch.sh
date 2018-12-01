#!/bin/bash

#exec >> cron-copy.log  2>&1
sDate=$(date "+%y%m%d-%H%M%S")
function redlog() { sDate=$(date "+%y%m%d-%H%M%S"); echo "$(tput setaf 1)$sDate $*$(tput sgr 0)"; }
function greenlog() { sDate=$(date "+%y%m%d-%H%M%S"); echo "$(tput setaf 2)$sDate $*$(tput sgr 0)"; }

nDirs=$1
shift
echo "nDirs: $nDirs"

for sHost in $*
do
    # find the most recent folder under ray_results
    greenlog $sHost
    lsDirLatest=$(ssh $sHost ls -t ray_results | head -$nDirs)
    greenlog Dirs: $lsDirLatest

    for sDirLatest in $lsDirLatest
    do
        greenlog $sDirLatest

        # In that folder, find the most recent checkpoint and stub
        #sfCheckpoint=`ssh $sHost "cd ray_results/$sDirLatest; ls -t checkpoint-* | head -1"`
        sfCheckpoint=`ssh $sHost "cd ray_results/$sDirLatest; ls -t checkpoint_* | head -1"`
        #sfCheckpointStub=`echo $sfCheckpoint | cut -d. -f1`
        sfCheckpointStub=$(echo $sfCheckpoint | perl -n -e '/(checkpoint_[\d]+)/ && print $1' )
        greenlog $sfCheckpoint $sfCheckpointStub

        # allocate a local folder and ensure it exists
        #sDirLocal=~/projects/crowdai/prosthetics/model-data/$sHost/$sDirLatest
        sDirLocal=~/ray_results/$sDirLatest
        greenlog $sDirLocal
        mkdir -p $sDirLocal

        # Copy the checkpoint, event, json and csv files locally
        scp -pr $sHost:ray_results/$sDirLatest/{$sfCheckpointStub*,event*,*.json,*.csv} $sDirLocal
    done
done