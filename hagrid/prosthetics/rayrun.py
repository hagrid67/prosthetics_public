

import gym
from gym.spaces import Box, Tuple, Discrete
import tensorflow as tf
import ray
from ray.rllib.agents.ppo import PPOAgent, DEFAULT_CONFIG
import ray.rllib.agents.ppo
from ray.rllib.agents.dqn.apex import ApexAgent, APEX_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.apex import ApexDDPGAgent, APEX_DDPG_DEFAULT_CONFIG
from ray.rllib.utils import merge_dicts
from ray.tune.registry import register_env
import psutil
import time
import argh
import socket
import pickle
import numpy as np
from osim.env import osim
import scipy.special

from osim.http.client import Client
from osim.redis.client import Client as RedisClient
#from helper.wrappers import ClientToEnv, DictToListLegacy, ForceDictObservation, JSONable
#from helper.wrappers.Wrapper import EnvironmentWrapper

from hagrid.prosthetics import runner
from hagrid.prosthetics import preprocessors
from hagrid.prosthetics.SubprocEnv import SubprocEnv



import json, sys, os, re
from glob import glob

from tensorflow.python.client import device_lib

def floatstr(*lrVal):
    return " ".join(["{:.2f}".format(rVal) for rVal in lrVal])

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def env_creator(env_config):
    #import gym
    print("env_creator env_config:", env_config)
    return runner.IsolatedEnv(
        visualize=env_config["visualize"],
        step_timeout=2700, # 45 mins
        nObsVer=1,
        dEnvConfig=env_config,
        # dict(rBaseReward=2.),
        )
    #return gym.make("Prosthetics-v1")  # or return your own custom env

# does frameskip, DicttolistFull, squash actions
#class ProstheticsEnv2(osim.ProstheticsEnv):
class ProstheticsEnv2(SubprocEnv):

    def __init__(self, 
            visualize=False,
            dEnvConfig = {},
            frameskip=1,
            ):
        self.frameskip = frameskip
        #super(ProstheticsEnv2, self).__init__(
        super().__init__(
            visualize=visualize,
            dEnvConfig = dEnvConfig,
            )
        #self.action_space = Tuple([Discrete(2)] * 19) # shape=[18])
        self.action_space = Box(low=0.0, high=1.0, shape=[19], dtype=np.float32)
        #self.observation_space = Box(low=0.0, high=1.0, shape=[352], dtype=np.float32)


    def step(self, gAct):
        gAct2 = np.array(gAct)
        gAct2 = scipy.special.expit(gAct2*10)

        rCumRew = 0.
        for i in range(self.frameskip):
            dObs, rRew, bDone, dInfo = super().step(gAct2)
            rCumRew += rRew
            if bDone:
                break
        
        gObs = preprocessors.DictToListFull(dObs, nObsVer=1, dEnvConfig=self.dEnvConfig)
        #return [gObs, rRew, bDone, dInfo]
        return [gObs, rCumRew, bDone, dInfo]

    def get_observation_space_size(self):
        return self.observation_space.shape[0]

    def get_action_space_size(self):
        return 19

    def reset(self):
        dObsReset = super().reset() # project=False) # Oct 8
        gObs = preprocessors.DictToListFull(dObsReset, nObsVer=1, dEnvConfig=self.dEnvConfig)
        return gObs
    

def env_creator2(env_config):
    #import gym
    print("env_creator2 env_config:", env_config)
    oEnv = ProstheticsEnv2(
        visualize=env_config["visualize"],
        dEnvConfig=env_config,
        )
    #return gym.make("Prosthetics-v1")  # or return your own custom env
    #print ("Obs space: ", oEnv.get_observation_space_size())
    #print ("obs space: ", oEnv.observation_space)
    return oEnv


def env_creator_arm(env_config):
    #import gym
    print("env_creator env_config:", env_config)
    return osim.Arm2DEnv(integrator_accuracy=0.001)
    #return gym.make("Prosthetics-v1")  # or return your own custom env



def ppo(sCheckpoint = None, 
        bSubmit = False, 
        bTrain = False, 
        bUseLastCheckpoint = False,
        sCheckpointPattern = None,
        sRedis=None, 
        sURL=None,
        nWorkers=psutil.cpu_count() - 2,
        bVisualize=False,
        nStartDelay=60,
        nHidWidth=128,
        integrator_accuracy=0.0, # zero will be ignored
        ):

    ray.init(redis_address=sRedis, ignore_reinit_error=True)

    #if bSubmit:
    #    nWorkers = 1
    #else:
    #    nWorkers = psutil.cpu_count() - 2

    sHost = socket.gethostname()


    nGpusOnHost = len(get_available_gpus())
    nGpusToUse =  1 if nGpusOnHost>0 else 0
    print ("nGpus:", nGpusOnHost, nGpusToUse)

    nStepsPerWorkerBatch = 50 # 32 # down from 50

    nSgdMinibatch = 128 # apparently not used but generates warning



    config = merge_dicts(DEFAULT_CONFIG.copy(), {
        "gamma":                0.999, # 28/11 from 0.99.
        'num_sgd_iter':         20, # 30, # 27/11
        'num_workers':          nWorkers,
        "sample_batch_size":    nStepsPerWorkerBatch, # try oct 4, down from 100
        #'train_batch_size':     (psutil.cpu_count()-2) * 50,
        "sgd_minibatch_size":   nSgdMinibatch,
        'train_batch_size':     max(nSgdMinibatch,
                                    nWorkers * nStepsPerWorkerBatch * 1), # 3, # 2 didn't work for 1 cpu
        #config['sgd_batchsize'] = 128,
        #'model': {'fcnet_hiddens': [256, 256] }, # slow 
        'model': {
            'fcnet_hiddens':        [nHidWidth, nHidWidth], # gc2
            #'fcnet_hiddens':        [256, 256], # gc3
            #"squash_to_range":      True, 
            #"stochastic_action":    False,
             }, # not yet tried

        # new things to try
        # "fcnet_activation":   "tanh", # try relu etc
        "lr": 1e-4, # 27/11
        #"lr": 5e-5, # used for everything previously
        #"lr":                   3e-4, # 10 oct - didn't work well.
        # "num_envs_per_worker": 1, # try 4 envs per worker, fewer workers?
        

        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param":        1e5, # new 4 Oct
        "batch_mode":           "truncate_episodes", # try 4 Oct.
        # "batch_mode":           "complete_episodes", # 4 Oct.
        "sample_async":         True,
        "horizon":              1100, # 15 Oct 1000->1100. See also dEnvConfig timestep_limit
        # Which observation filter to apply to the observation
        "observation_filter": "ConcurrentMeanStdFilter",

        "num_gpus"                  : nGpusToUse,

        })

    bMusc=True
        # these get passed to env_creator
    dEnvConfig = config["env_config"] = {
        "visualize":            False if bSubmit else True if bVisualize else False, # sHost=="jwpc12",
        "nObsVer":              1,
        "rBaseReward":          2, # 5, # 2, # 10., # gc0 Oct 11.
        # which muscle info to include
        "activation":           bMusc, # same as action?
        "fiber_force":          bMusc, # similar to activation?
        "fiber_length":         bMusc, # available from joint angles
        "fiber_velocity":       bMusc, # who cares...
        "rPenPelvisRot":        0.5,
        "rPenHipAdd":           0, # 1, # 0.5, # penalize positive value for hip ab/aduction
        "rHipAddThresh":        -0.2, # 0.5, # penalize positive value for hip ab/aduction
        
        "rPenKneeStraight":     0, # 1,
        "rKneeThresh":          -0.2,

        #"rKinDenLoc":           2,
        #"rKinCoeffLoc":         0,
        #"rKinDenVel":           0.1,
        #"rKinCoeffVel":         0,
        #"rKinCoeff":            0.7,
    

        "integrator_accuracy":  integrator_accuracy if integrator_accuracy \
                                else 5e-5 if sHost == "jwpc12" else 0.001, 
                                # from 5e-5.  Now using 
                                # SimTK::SemiExplicitEuler2Integrator
                                # accuracy value taken from dEnvConfig in 
                                # Runner standalone_headless

        "timestep_limit":       1100,  # set a slightly longer horizon

        "debug":                sHost=="jwpc12",

        }
    

    #agent = PPOAgent(config, 'CartPole-v0')
    # agent = PPOAgent(config, 'Prosthetics-v1')
    print ("test 1")
    #register_env("Prosthetics-v2", env_creator)
    register_env("Prosthetics-v3", env_creator2)
    #register_env("Arm2D-v1", env_creator_arm)
    if True:
        print("testing 123b")
        agent = PPOAgent(
            env="Prosthetics-v3", # "Prosthetics-v2", 
            #env="Arm2D-v1",
            config = config,
            #config = merge_dicts(config.copy(), 
            #{
            #"env_config": { "visualize":bSubmit },  # config to pass to env creator
            #})
            )

    #agent.restore("/home/jeremy/ray_results/181001")
    #agent.restore("/home/jeremy/ray_results/PPO_Prosthetics-v1_2018-10-01_13-32-29hxgq_tqv/checkpoint-573")

    if bUseLastCheckpoint:
        #if sCheckpoint is not None:
        #    print("Discarding {} in favour of bUseLastCheckpoint".format(sCheckpoint))
        sCheckpoint = findLastSession(sCheckpointPattern)
        print("===="*100)
        print("bUseLastCheckpoint - using {}".format(sCheckpoint))

    if sCheckpoint:
        agent.restore(sCheckpoint)

    if bTrain:
        train(agent)

    if bSubmit:
        submit(agent, dEnvConfig, sURL, nStartDelay=nStartDelay)

    return agent
   


def play(sCheckpoint = None, 
        bSubmit = False, 
        bUseLastCheckpoint = False,
        sCheckpointPattern = None,
        sRedis=None, 
        sURL=None,
        bVisualize=False,
        nHidWidth=128,
        integrator_accuracy=0.0, # zero will be ignored
        ):

    #ray.init(redis_address=sRedis, ignore_reinit_error=True)

    #if bSubmit:
    #    nWorkers = 1
    #else:
    #    nWorkers = psutil.cpu_count() - 2

    sHost = socket.gethostname()


    bMusc=True
        # these get passed to env_creator
    dEnvConfig = {
        "visualize":            False if bSubmit else True if bVisualize else False, # sHost=="jwpc12",
        "nObsVer":              1,
        "rBaseReward":          2, # 10., # gc0 Oct 11.
        # which muscle info to include
        "activation":           bMusc, # same as action?
        "fiber_force":          bMusc, # similar to activation?
        "fiber_length":         bMusc, # available from joint angles
        "fiber_velocity":       bMusc, # who cares...
        "rPenPelvisRot":        0, # 0.5,
        "rPenHipAdd":           0, # 1, # 0.5, # penalize positive value for hip ab/aduction
        "rHipAddThresh":        -0.2, # 0.5, # penalize positive value for hip ab/aduction
        
        "rPenKneeStraight":     0, # 1,
        "rKneeThresh":          -0.2,

        "rPenStateLoc":         2,
        "rPenStateVel":         0.1,


        "integrator_accuracy":  integrator_accuracy if integrator_accuracy \
                                else 5e-5 if sHost == "jwpc12" else 0.001, 
                                # from 5e-5.  Now using 
                                # SimTK::SemiExplicitEuler2Integrator
                                # accuracy value taken from dEnvConfig in 
                                # Runner standalone_headless

        "timestep_limit":       1000,  # set a slightly longer horizon

        "debug":                sHost=="jwpc12",

        }
    

    dConfig = merge_dicts(ray.rllib.agents.ppo.DEFAULT_CONFIG, 
           {
               "env_config":dEnvConfig, 
               "model":{"fcnet_hiddens":[nHidWidth, nHidWidth]},
            }
        )

    #oE = ProstheticsEnv(difficulty=1, dEnvConfig=dEnvConfig)
    #dObs = oE.reset()
    lObs = preprocessors.DictToListFull(None, nObsVer=1, dEnvConfig=dEnvConfig, bDummy=True)
    oObsSpace = Box(low=-5, high=5, shape=(len(lObs),), dtype=np.float32)
    #oActionSpace = gym.spaces.Box(low=0, high=1, shape=(19,), dtype=np.float32)
    #oActionSpace = gym.spaces.MultiBinary(19)



    #agent = PPOAgent(config, 'CartPole-v0')
    # agent = PPOAgent(config, 'Prosthetics-v1')
    if True:
        register_env("Prosthetics-v2", env_creator)
        register_env("Prosthetics-v3", env_creator2)
  
    oEnv = env_creator2(dEnvConfig)

    #agent.restore("/home/jeremy/ray_results/181001")
    #agent.restore("/home/jeremy/ray_results/PPO_Prosthetics-v1_2018-10-01_13-32-29hxgq_tqv/checkpoint-573")

    if bUseLastCheckpoint:
        sCheckpoint = findLastSession(sCheckpointPattern)
        print("bUseLastCheckpoint - using {}".format(sCheckpoint))

    with open(sCheckpoint + ".agent_state", "rb") as fbIn:
        dAgentState = pickle.load(fbIn)

    sess=tf.Session()
    sess.__enter__()
    
    oPG = ray.rllib.agents.ppo.ppo_policy_graph.PPOPolicyGraph(oEnv.observation_space, oEnv.action_space, dConfig)


    dEval = pickle.loads(dAgentState["evaluator"])
    print("dEval:", type(dEval), dEval.keys())
    oState = dEval["state"]["default"]
    print("state->default:", type(oState), oState.shape)

    oPG.set_state(dEval["state"]["default"])

    oFil = dEval["filters"]["default"]
    print("oFil:", type(oFil), oFil)
    
    tAct = oPG.compute_single_action(lObs, [], is_training=False, )
    llAct = tAct[0]
    print("tAct:", tAct)
    print ("lAct:", type(llAct))
    print("lAct:",  " ".join([str(lAct[0]) for lAct in llAct]))



    if bSubmit:
        #submit(agent, dEnvConfig, sURL, nStartDelay=nStartDelay)
        client = RedisClient()

        # Create environment
        dObs = client.env_create()
        

        iEp = 0
        iStep = 0
        rCumRew = 0
        reward = 0

        """
        The grader runs N simulations of at most 1000 steps each. We stop after the last one
        A new simulation start when `clinet.env_step` returns `done==True`
        and all the simulatiosn end when the subsequent `client.env_reset()` returns a False
        """
        while True:

            lObs = preprocessors.DictToListFull(dObs, nObsVer=1, bRelPos=True, dEnvConfig=dEnvConfig)
            tAct = oPG.compute_single_action(lObs, [], is_training=False, )
            gAct = tAct[0]
            
            [dObs, reward, done, info] = client.env_step(gAct)

            rCumRew += reward
            iStep += 1

            #print(observation)
            if done:
                print("Ep ", iEp, "done. CumRew: ", rCumRew)
                iEp += 1
                reward = 0
                rCumRew = 0
                iStep = 0
                #observation = client_env.reset()
                dObs = client.env_reset()
                if not dObs:
                    print("Observation null - break loop")
                    break

            print(("Ep: {:} step: {:}  Rew: {:.3f} CumRew: {:.3f}, lObs:{:} " + 
                    "vx,z:{:.4f} {:.4f} tvx,z:{:.2f} {:.2f} pel_y:{:.2f}" #tvx,z:{:.2f} {:.2f} "
                    ).format(
                iEp, iStep, reward, rCumRew, -1, # len(lObs),
                dObs["body_vel"]["pelvis"][0], dObs["body_vel"]["pelvis"][2], 
                dObs["target_vel"][0], dObs["target_vel"][2], # pull out x,z, ignore y
                #lObs[-2], lObs[-1] # only x,z at the end of the list
                dObs["body_pos"]["pelvis"][1],
                ))

        client.submit()
    else:


    
        iEp = 0
        iStep = 0
        rCumRew = 0
        reward = 0

        lObs = oEnv.reset()
        print("lObs:", type(lObs), len(lObs))
        gObsFil = oFil(lObs)

        print("lObs:   ", floatstr(*lObs[:20]))
        print("gObsFil:", floatstr(*gObsFil[:20]))

        

        while True:

            #lObs = preprocessors.DictToListFull(dObs, nObsVer=1, bRelPos=True, dEnvConfig=dEnvConfig)
            gObsFil = oFil(lObs)
            tAct = oPG.compute_single_action(gObsFil, [], is_training=False, )
            gAct = tAct[0]
            #print ("gAct:", type(gAct), gAct)
            
            [lObs, reward, done, info] = oEnv.step(gAct)

            rCumRew += reward
            iStep += 1

            #print(observation)
            if done:
                print("Ep ", iEp, "done. CumRew: ", rCumRew)
                iEp += 1
                reward = 0
                rCumRew = 0
                iStep = 0
                #observation = client_env.reset()
                lObs = oEnv.reset()
                if not lObs:
                    print("Observation null - break loop")
                    break

            print(("Ep: {:} step: {:}  Rew: {:.3f} CumRew: {:.3f}, lObs:{:} " + 
                    "" #"vx,z:{:.4f} {:.4f} tvx,z:{:.2f} {:.2f} pel_y:{:.2f}" #tvx,z:{:.2f} {:.2f} "
                    ).format(
                iEp, iStep, reward, rCumRew, -1, # len(lObs),
                #dObs["body_vel"]["pelvis"][0], dObs["body_vel"]["pelvis"][2], 
                #dObs["target_vel"][0], dObs["target_vel"][2], # pull out x,z, ignore y
                #lObs[-2], lObs[-1] # only x,z at the end of the list
                #dObs["body_pos"]["pelvis"][1],
                ))



    sess.__exit__(None, None, None)
    sess.close()



def train(agent):

    tSave = time.time() - 1000 # set early so we always save the first iteration
    for i in range(10000):
        result = agent.train()
        print(result)

        if False:
            for i in range(1):
                print(agent.local_evaluator.get_filters() )
                for oRemEval in agent.remote_evaluators:
                    oFil = ray.get(oRemEval.get_filters.remote())["default"]
                    print(oFil, oFil.rs.mean[:10])
                #time.sleep(10)

        tIter = time.time()
        if (tIter - tSave) > 300: # 5 mins
            sDirCP = agent.save()
            print ("Saved in :", sDirCP)
            tSave = tIter




def submit(agent, dEnvConfig, sURL = "http://localhost:8050", nStartDelay=40):
        print("Submit!")
        
        #remote_base = "http://grader.crowdai.org:1730"
        #remote_base = "http://localhost:8050"
        remote_base = sURL

        #for i in range(2):
        #    print("*************Sleeping, hoping the agent settles down*****************")
        #    time.sleep(nStartDelay)

        #    print("************** training *************************************************")
        #    agent.train() # to initialise the RunningMeanStd

        print("**************** sleeping ****************************************")
        time.sleep(nStartDelay)


        #client_env = ClientToEnv(client)
        #client_env = DictToListLegacy(client_env)
        #client_env = JSONable(client_env)

        #print ("Dir of client_env:", dir(client_env))
        #print ("Dir of client:", dir(client))

        #observation = client_env.reset()


        with open("crowdAI-hagrid67-token.txt", "r") as fIn:
            sToken = fIn.read()
        print("submission token:", sToken)


        client = Client(remote_base)
        #observation = client.env_create(args.token, env_id="Run")
        dObs = client.env_create(sToken, env_id="ProstheticsEnv")
        print("observation: ", type(dObs))
        print("observation: ", len(dObs))


        iEp = 0
        iStep = 0
        rCumRew = 0
        reward = 0




        # Run a single step
        # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
        iStep = 0
        while True:
            #print("Ep:", iEp, "step:", iStep, "Rew: ", reward, "CumRew: ", rCumRew, "obs:", type(observation), len(observation))
            
            #print("dObs:", type(dObs), len(dObs), " ".join(dObs.keys()))
            
            #v = np.array(observation).reshape((dummy_env.observation_space.shape[0]))
            #action = agent.forward(v)
            #action = pi.act(stochastic=False, ob=np.array(observation))[0]
            #lObs = ObsDictToList(dObs)
            
            print(("Ep: {:} step: {:}  Rew: {:.3f} CumRew: {:.3f}, lObs:{:} " + 
                    "vx,z:{:.4f} {:.4f} tvx,z:{:.2f} {:.2f} pel_y:{:.2f}" #tvx,z:{:.2f} {:.2f} "
                    ).format(
                iEp, iStep, reward, rCumRew, -1, # len(lObs),
                dObs["body_vel"]["pelvis"][0], dObs["body_vel"]["pelvis"][2], 
                dObs["target_vel"][0], dObs["target_vel"][2], # pull out x,z, ignore y
                #lObs[-2], lObs[-1] # only x,z at the end of the list
                dObs["body_pos"]["pelvis"][1],
                ))

            with open("dObs.txt", "a") as fOut:
                json.dump(dObs, fOut)
                fOut.write("\n")

            lObs = preprocessors.DictToListFull(dObs, nObsVer=1, bRelPos=True, dEnvConfig=dEnvConfig)
            #print(len(lObs), dObs["target_vel"], lObs[-2:])
            #print("lObs:", type(lObs), len(lObs))
            action = agent.compute_action(lObs)
            #[observation, reward, done, info] = client_env.step(action.tolist())
            [dObs, reward, done, info] = client.env_step(action.tolist())
            rCumRew += reward
            iStep += 1
            if done:
                print("Ep ", iEp, "done. CumRew: ", rCumRew)
                iEp += 1
                reward = 0
                rCumRew = 0
                iStep = 0
                #observation = client_env.reset()
                dObs = client.env_reset()
                if dObs is None:
                    print("Observation null - break loop")
                    break
                
        print("Submit")
        #print("client_env:", client_env)
        #client_env.submit()
        print("client:", client)
        client.submit()

def findLastSession(sPattern=None):
    if sPattern is None:
        sPattern="*"

    # folder structure:
    # PPO_Prosthetics-v2_2018-10-16_06-39-182euhp0mc
    # checkpoint_5094c2bygkrm
    # checkpoint-5094.agent_state

    print("Searching for checkpoint folder with pattern:", sPattern)
    # PPO_Prosthetics-v2_2018-10-16_06-39-182euhp0mc
    lEntries = glob("/home/jeremy/ray_results/{}".format(sPattern))


    # check it's a folder and contains checkpoint folders:    
    # PPO_Prosthetics-v2_2018-10-16_06-39-182euhp0mc
    # checkpoint_5094c2bygkrm
    lDirs = [ sEntry for sEntry in lEntries if os.path.isdir(sEntry) # filter for directories only, 
        and len(glob(sEntry+"/checkpoint_[0-9]*")) > 0 ]  # containing checkpoint folders
    lDirs.sort(key=os.path.getmtime) # sort by date, most recent last
    sDir = lDirs[-1] # newest dir
    print("Looking for checkpoints in {}".format(sDir))
    #lCP = glob("{}/checkpoint-*.agent_state".format(sDir)) # list of checkpoint files
    #lCP.sort(key=lambda x: int(re.search("checkpoint-(\d+)[\.]", x).group(1)) ) # sort by index number

    # checkpoint_5094c2bygkrm
    lCP = glob("{}/checkpoint_*/checkpoint-*.agent_state".format(sDir))

    lCP.sort(key=lambda x: int(re.search("checkpoint-(\d+).agent_state", x).group(1)) ) # sort by index number
    #sLastCheckpoint = lCP[-1][:-11] # remove the ".extra_data" suffix as reqd
    #print("list of checkpoint folders:", lCP)
    sLatestCP = lCP[-1]
    #sLastCheckpoint = lCP[-1][:-11] # remove the ".extra_data" suffix as reqd
    #sLastCheckpoint = glob(sDirLatestCP + "/checkpoint-*.agent_state")[0]
    sLatestCP = sLatestCP[:-12]
    return sLatestCP


@ray.remote
def checkip():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()

def checkips(sRedis=None):
    ray.init(redis_address=sRedis)
    setNodes = set(ray.get([checkip.remote() for i in range(1000)]))
    print(setNodes)


def start():
    ray.init(ignore_reinit_error=True)


def main():
    #ray.init(ignore_reinit_error=True)
    print("main")
    parser = argh.ArghParser()
    parser.add_commands([ppo, play, checkips, findLastSession])
    parser.dispatch()
    

if __name__ == "__main__":
    main()
