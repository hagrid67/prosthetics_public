#!/bin/bash

#tmux start-server
tmux new-session -d -s $(hostname)
# run training in window 0
tmux send-keys "cd ~/projects/crowdai/prosthetics/main" C-m "python -m hagrid.prosthetics.rayrun ppo --bTrain --bUseLastCheckpoint --nWorkers 60" C-m
# run tensorboard and top in window 1
tmux new-window ";" send-keys "cd ~/ray_results; tensorboard --logdir .&" C-m "top" C-m ";"
# switch to first window (python)
tmux select-window -t :0

