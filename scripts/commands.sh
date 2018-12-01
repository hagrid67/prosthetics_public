


# Run trajectory segment generation
python -m hagrid.prosthetics.runner --model-redis-load tf:ppo1 --num-timesteps 10000 --moments ./model-data-keep/Obs-MeanStd.json --play --nMaxSegTime 300 --nMaxSegSteps 1024

python -m hagrid.prosthetics.runner --model-path-load ../model-data/gc5/ppo-180819/ppo --model-path-save ../model-data/ppo-0919-test --num-timesteps 1e6 --moments ./model-data-keep/Obs-MeanStd.json --train --hid-size 256 --ts-per-batch 128 --model-redis-save tf:ppo1



# Old run_humanoid command from baselines, running with humanoid / flagrun from roboschool?
mpirun -np 16  python -m baselines.ppo1.run_humanoid --model-path=./model-ppo1d/ --load true --num-timesteps 100000000

# prosthetics runner
# --load doesn't work yet
#mpirun  --mca btl ^openib -np 14 python -m hagrid.prosthetics.runner --model-path-load=./model-data/ppo-0808b/ppo-0808a --model-path-save=./model-data/ppo-0809a/ppo --num-timesteps 1e6 --moments ./model-data-keep/Obs-MeanStd.json 
mpirun --mca btl ^openib -np 15 python -m hagrid.prosthetics.runner --model-path-load=./model-data/ppo-0809a/ppo --model-path-save=./model-data/ppo-0809b/ppo --num-timesteps 2e7 --moments ./model-data-keep/Obs-MeanStd.json --train

#Submit (token for hagrid67)
python -m hagrid.prosthetics.runner --model-path ./model-data/model-ppo-0805-1m-16cpu/ --load --num-timesteps 10000 --moments ./model-data-keep/Obs-MeanStd.json  --submit --token 50fb66bb6e5f4fdac76d1cb68a5b9038






# GCP
gcloud compute instances create gc1 --machine-type n1-standard-4 --zone us-east1-b --boot-disk-size 32G --preemptible --image-family ubuntu-minimal-1804-lts --image-project ubuntu-os-cloud


# list branches
git branch -a 

# rename the remote from origin to upstream, and add a new origin pointing to my fork
git remote -v
git remote rename origin upstream
git remote add origin https://github.com/hagrid67/osim-rl-grader
git remote -v

# just fetch updates, don't touch local branch
git fetch
# fetch and merge
git pull 
# switch to a branch jw - unstaged(?) changes are left in situ in the local copy(?)
git checkout jw

# push updates to remote
git push origin 
# set the upstream for the current branch "jw" to the remote "origin"
git push --set-upstream origin jw

# restore a file deleted (but not yet committed)
git checkout HEAD path/filename

# Run a git status in the main git branches / repos
for sDir in baselines gym osim-rl osim-rl-grader osim-rl-helper version . ; do echo $sDir===============; git -C $sDir status; done


# Build OpenSim
# https://github.com/opensim-org/opensim-core - got to bottom of readme - follow "In Terminal" instructions.

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

mkdir opensim_dependencies_build
cd opensim_dependencies_build
cmake ../opensim-core/dependencies/ \
      -DCMAKE_INSTALL_PREFIX='~/opensim_dependencies_install' \
      -DCMAKE_BUILD_TYPE=


cmake ../opensim-core  -DCMAKE_INSTALL_PREFIX="~/opensim_install2" -DCMAKE_BUILD_TYPE=Release  -DOPENSIM_DEPENDENCIES_DIR="~/opensim_dependencies_install"  -DBUILD_PYTHON_WRAPPING=ON   -DBUILD_JAVA_WRAPPING=ON  -DWITH_BTK=ON   -DOPENSIM_PYTHON_VERSION=3 -DPYTHON_INCLUDE_PATH=/home/jeremy/miniconda3/envs/osim361/include/python3.6m

#make[2]: *** No rule to make target '/home/jeremy/projects/crowdai/prosthetics/opensim-core/Bindings/Python/Bindings/preliminaries.i', needed by 'Bindings/Python/examplecomponents.py'. Stop.


#from gcloud - create machine
gcloud compute --project "vernal-design-171120" disks create "ngc3-1" --size "200" --zone "us-east1-b" --source-snapshot "snapshot-ngc0" --type "pd-standard"

gcloud beta compute --project=vernal-design-171120 instances create ngc3-1 --zone=us-east1-b --machine-type=n1-highcpu-16 --subnet=default --network-tier=PREMIUM --no-restart-on-failure --maintenance-policy=MIGRATE --service-account=28837673100-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --disk=name=ngc3-1,device-name=ngc3-1,mode=rw,boot=yes,auto-delete=yes

# This one worked
gcloud beta compute --project=vernal-design-171120 instances create ngc10 --zone=europe-west1-b --machine-type=n1-highcpu-64 --subnet=default --network-tier=PREMIUM --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible --service-account=28837673100-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --image=jwray2 --image-project=vernal-design-171120 --boot-disk-size=200GB --boot-disk-type=pd-standard --boot-disk-device-name=ngc10