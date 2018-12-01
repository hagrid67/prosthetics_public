#!/bin/bash

host=am0
dirDest=~/projects/crowdai/prosthetics/main/model-data/am0-16cpu-180809b-0818
cd $dirDest || echo "Cannot cd into destination folder $dirDest"
scp $host:projects/crowdai/prosthetics/model-data/log*.txt ~/projects/crowdai/prosthetics/main/model-data/am0-16cpu-180809b-0818

# Ep Rew: 163.20 len: 240

for sFile in log*.txt
do
	awk '/Ep Rew:/ { print $3 "," $5}' < $sFile > ep$sFile
done

