#!/usr/bin/env python3
import os
#from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
#from baselines.common import tf_util as U
#from baselines import logger

# from mpi4py import MPI
#from baselines.bench import Monitor
#from baselines.common import set_global_seeds


from osim.http.client import Client
#from helper.wrappers import ClientToEnv, DictToListLegacy, ForceDictObservation, JSONable
#from helper.wrappers.Wrapper import EnvironmentWrapper

#import roboschool
from osim.env.osim import ProstheticsEnv
import atexit, h5py
from os import path
import time, socket
import numpy as np

import random
from datetime import datetime
import gym
from gym.spaces import Box, Tuple, Discrete

from multiprocessing import Process, Pipe
import json
import sys
import traceback
from hagrid.tf import tfutil
from hagrid.prosthetics import preprocessors
#from hagrid.prosthetics import ppoagent, ppolearner
import redis
import pickle
import tensorflow as tf

# bind our custom version of pelvis too low judgement function to original env
def bind_alternative_pelvis_judgement(runenv, val):
    def is_pelvis_too_low(self):
        return (self.current_state[self.STATE_PELVIS_Y] < val)
    import types
    runenv.is_pelvis_too_low = types.MethodType(is_pelvis_too_low, runenv)


# make_wrapped_env ->
#  IsolatedEnv(creates Process) ->
#  standalone_headless_isolated -> 
# {ProstheticsEnv, h5pyEnvLogger}



# Function which is run in a separate process that holds a RunEnv instance.
# (Apparently) This has to be done since RunEnv() in the same process results in interleaved running of simulations.
def standalone_headless_isolated(
        conn, 
        visualize=False, 
        n_obstacles=0, 
        run_logs_dir="./logs",
        additional_info={},
        higher_pelvis=0.65,
        dMoments=None,
        integrator_accuracy=5e-5,
        dEnvConfig={},
        ):
    try:
        
        # override arg with value from dEnvConfig
        if "integrator_accuracy" in dEnvConfig:
            integrator_accuracy = dEnvConfig["integrator_accuracy"]

        env = ProstheticsEnv(visualize=visualize,
            integrator_accuracy = integrator_accuracy, # now pass as arg again.  TODO: stick to dEnvConfig
            difficulty = 1, 
            seed=None,
            dEnvConfig=dEnvConfig,
            ) #, max_obstacles=n_obstacles)

        if False: # I don't think this is necessary now we can pass difficulty and seed in constructor
            env.change_model(
                model='3D',
                prosthetic=True,
                difficulty=1, # set to 1 for prosthetics round 2 with velocity vector
                seed=None)
            #if higher_pelvis != 0.65:
            #    bind_alternative_pelvis_judgement(e, higher_pelvis)

        #if dMoments is not None:
        #    env = ObsProcessor(env, dMoments=dMoments)
        #else:
        #    print("Not using external moments / normalizing")

        # log the normalized obs...
        #env = h5pyEnvLogger(env, log_dir=run_logs_dir, additional_info=additional_info)
        

        while True:
            lMsg = conn.recv() # wait for command

            # messages should be tuples,
            # msg[0] should be string

            if lMsg[0] == 'reset':
                #print("Reset")
                #o = env.reset(difficulty=lMsg[1], seed=lMsg[2]) # how did this ever work?
                #gObsReset = env.reset()
                dObsReset = env.reset(project=False) # Oct 8
                #print("Reset done")
                conn.send(dObsReset)

            elif lMsg[0] == 'step':
                dObs = env.step(lMsg[1], project=False) # Oct 8
                #env.osim_model.model.updVisualizer().updSimbodyVisualizer()
                if visualize:
                    env.render()
                conn.send(dObs)

            elif lMsg[0] == 'close':
                env.close()
                conn.send(None)

                import psutil
                current_process = psutil.Process()
                children = current_process.children(recursive=True)
                for child in children:
                    child.terminate()
                return
    except Exception as oEx:
        import traceback
        print(traceback.format_exc())
        conn.send(oEx)


class IsolatedEnv(gym.Env): # (gym.Wrapper): # jw - prev had no super
    """ Create subprocess holding env (using standalone_headless_isolated)
        Handle multiprocess pipe comms to/from subprocess
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, visualize=False,
            n_obstacles=3, 
            run_logs_dir="./logs", 
            additional_info={}, 
            step_timeout=2700, # seconds (4 Oct 1000->1200->1800->2700)
            higher_pelvis=0.65,
            dMoments=None,
            integrator_accuracy=5e-5,
            nObsVer=1,
            dEnvConfig = {},
            ):
        self.running = False

        # TODO: HACK - should really be getting this from the env
        #self.action_space = Box(low=0, high=1, shape=[19], dtype=np.float32) # shape=[18])
        self.action_space = Tuple([Discrete(2)] * 19) # shape=[18])

        #self.env = ProstheticsEnv(visualize=False, difficulty=1, seed=None) # dummy! memory leak?  need for ray?
        
        if nObsVer == 0:
            self.nObsSize = 162 # 160 # 158 # self.env.get_observation_space_size()
        elif nObsVer >= 1:
            lObsDummy = preprocessors.DictToListFull(None, nObsVer=nObsVer, dEnvConfig=dEnvConfig, bDummy=True)
            self.nObsSize = len(lObsDummy)

        self.observation_space = Box(low=-5, high=+5, shape=[self.nObsSize], dtype=np.float32) # shape=[41])
        #print(self.observation_space)
        self.reward_range = [0, 10000]
        self.visualize = visualize
        self.n_obstacles = n_obstacles
        self.run_logs_dir = run_logs_dir
        self.additional_info = additional_info
        self.pc = None
        self.step_timeout = step_timeout
        self.lastobs = None
        self.p = None
        self.higher_pelvis = higher_pelvis
        self.dMoments=dMoments
        self.nMpiRank = 0 # MPI.COMM_WORLD.Get_rank()
        self.bTimeout = False
        self.integrator_accuracy=integrator_accuracy
        self.nObsVer = nObsVer
        self.dEnvConfig = dEnvConfig

        #class DummyUnwrapped(object):
        #    spec=None
        #    def __init__(self):
        #        pass
        
        #class DummyEnv(object):
        #    unwrapped = DummyUnwrapped()
        #    def __init__(self):
        #        pass
        #print("self", type(self), self)
        #print(type(self.unwrapped), self.unwrapped)
        #self.unwrapped = DummyUnwrapped()


    def reset(self, difficulty=None, seed=None):

        # kill the child process if it exists already
        if self.running:
            self.pc.send(('close',))
            self.p.join()

        while True:

            self.pc, self.cc = Pipe()
            self.p = Process( ########################### New Process here ####################
                target=standalone_headless_isolated, # running env created in child process here
                args=(self.cc,),
                kwargs=dict(visualize=self.visualize, 
                    n_obstacles=self.n_obstacles, 
                    run_logs_dir=self.run_logs_dir, 
                    additional_info=self.additional_info, 
                    higher_pelvis=self.higher_pelvis,
                    dMoments=self.dMoments,
                    integrator_accuracy=self.integrator_accuracy,
                    dEnvConfig=self.dEnvConfig,
                    ),
            )
            # self.p.daemon = True
            self.p.start()
            self.pc.send(('reset', difficulty, seed), )

            finished = self.pc.poll(10) # Allow 10 seconds for reset.  2 is usually enough.
            if finished: # reset finished on time
                #print("Reset completed on time")
                self.bTimeout=False
                break
            
            else: # the reset timed out.
                dtNow = datetime.now()
                print('env {:} Env reset Timed out {:}s at {}'.format(self.nMpiRank, self.step_timeout, dtNow.strftime("%m%d-%H%M%S")))
                print("terminating...")
                self.p.terminate() # jw
                time.sleep(2)
                print("joining...")
                self.p.join()
                

        self.running = True

        res = self.pc.recv() # receive the output of the reset (dObs)
        self.lastobs = res
        if isinstance(res, Exception):
            raise res
        if self.nObsVer==0:
            return res # no projection (not working now)
        elif self.nObsVer>=1:
            return preprocessors.DictToListFull(res, nObsVer=self.nObsVer,
                dEnvConfig=self.dEnvConfig)

    def step(self, actions):

        #print(" ".join([str(rAct) for rAct in actions]))

        if self.bTimeout:
            print("env {:} timeout state".format(self.nMpiRank))
        else:
            self.pc.send(('step', list(actions)), )

        finished = self.pc.poll(self.step_timeout)
        if finished: # ie the step finished on time
            self.bTimeout=False
            res = self.pc.recv()
            if isinstance(res, Exception):
                raise res
            self.lastobs = res[0]
            #print(self.nMpiRank, type(res), len(res))
            if self.nObsVer==0: 
                return res
            elif self.nObsVer>=1:
                return [ 
                    preprocessors.DictToListFull(res[0],
                        nObsVer=self.nObsVer, dEnvConfig=self.dEnvConfig),
                    res[1], res[2], res[3] ]
        else: # the step timed out.
            self.bTimeout=True
            dtNow = datetime.now()
            print("###################### Env Timeout ####################")
            print('env {:} Env Timed out {:}s at {}'.format(self.nMpiRank, self.step_timeout, dtNow.strftime("%m%d-%H%M%S")))
            
            # self.p.terminate() # no need - agent will send reset which will terminate

            return self.lastobs, -0., True, {"Timed out": True} # ie bDone = True
            #tRV = (None, -0.045, False, {"Timed out": True})
            #print("returning:", tRV)
            #raise ValueError("env timeout")
            #return tRV

    def close(self):
        if self.pc is not None:
            self.pc.send(('close',))
            res = self.pc.recv()
            if isinstance(res, Exception):
                raise res

    def __del__(self):
        if self.pc is not None:
            self.pc.send(('close',))
        return


def make_wrapped_env(seed=123,
        visualize=False, 
        run_logs_dir="./run_logs/", 
        dMoments=None,
        step_timeout=10,
        integrator_accuracy = 5e-5,
        ):
    """
    The intention was to create a wrapped, monitored gym.Env,
    by creating the whole chain explicitly here.
    But it still just calls the first in a chain, which implicitly calls the second, etc.
    """
    rank = 0 #  MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed + 10000 * rank)
    print("Making wrapped env")
    env = IsolatedEnv(visualize=visualize,
        run_logs_dir=run_logs_dir,
        dMoments=dMoments,
        step_timeout=step_timeout,
        integrator_accuracy=integrator_accuracy
        )
    #print("IsolatedEnv: ", type(env))
    #env = ProstheticsEnv(visualize=visualize)
    #env = Monitor(env, os.path.join(logger.get_dir(), str(rank)))
    #print("h5pyEnvLogger:")
    #env = h5pyEnvLogger(env, "obs-logs", str(rank))
    #print("h5pyEnvLogger:", type(env))
    #env.seed(seed) # jw
    return env


# from nnaisense
def get_random_token():
    s = random.getstate()
    random.seed()
    token = ''.join(random.choice('0123456789abcdef') for n in range(5)) # changed from 30
    random.setstate(s)
    return token


dummy='''
class ObsProcessor(EnvironmentWrapper):
    """ 
    Wraps an environment, pre-processing the observations.  For now just normalize 
    using mean, std.
    """
    def __init__(self, env, dMoments):
        super(self.__class__, self).__init__(env)
        self.dMoments = dMoments
        self.gMean = dMoments["mean"]
        self.gStd = np.clip(dMoments["std"], 1e-4, 9e9)


    # Change _reset to reset
    def reset(self, difficulty=None, seed=None):
        #print("h5pyEnvLogger reset", type(self.env))
        gObs = self.env.reset() # difficulty=difficulty, seed=seed)
        
        if isinstance(gObs, bool) and gObs == False:
            return gObs

        #print("reset obs: ", type(gObs), len(gObs), gObs[:10], "...")
        gObs = self.normalize(gObs)
        #print("normalized obs: ", type(gObs), len(gObs), gObs[:10], "...")

        return gObs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.normalize(obs), reward, done, info

    def render(self):
        self.env.osim_model.model.updVisualizer().updSimbodyVisualizer()


    def normalize(self, oObs):

        # try to preserve type - either None, list or numpy array
        if oObs is None:
            return None
        
        if isinstance(oObs, list):
            gObs = np.array(oObs)
        else:
            gObs = oObs
        
        gNormd = (gObs - self.gMean) / self.gStd
        
        if isinstance(oObs, list):
            return gNormd.tolist()
        else:
            return gNormd
        
'''


class h5pyEnvLogger(gym.Wrapper):
    """ Wraps any environment saving the observations and actions to a dir/[filename_prefix][date-unique-string].h5py """
    def __init__(self, env, log_dir, filename_prefix="", additional_info={}):
        super(self.__class__, self).__init__(env)
        self.log_dir = log_dir
        self.filename_prefix = filename_prefix
        self.additional_info = additional_info

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        self.f = None
        atexit.register(self._try_save)  # If the last one was not properly finished (with done=True)

    # Change _reset to reset
    def reset(self, difficulty=None, seed=None):
        #print("h5pyEnvLogger reset", type(self.env))
        obs = self.env.reset() # difficulty=difficulty, seed=seed)
        if isinstance(obs, bool) and obs == False:
            return obs

        self._try_save(done=False)
        self.f = self._get_new_file()

        # Create initial lists to store obs, rew, action
        self.obs_seq = [obs]
        self.rew_seq = [0]
        self.action_seq = []
        self.time_seq = []
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.action_seq.append(action)
        self.obs_seq.append(obs)
        self.rew_seq.append(reward)
        self.time_seq.append(datetime.now().timestamp())
        # print(len(self.obs_seq))
        if done:
            self._try_save(done=True)
        return obs, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self._try_save(done=False)
        if self.env:
            self.env.close()

    def _get_new_file(self):
        token = get_random_token()
        now = datetime.now()
        fname = self.filename_prefix + "{:%Y-%m-%d-%H-%M-%S}_{}.hdf5".format(now, token)
        
        f = h5py.File(path.join(self.log_dir, fname))
        #print("new file:", f)

        f.attrs['timestamp'] = int(time.time())
        f.attrs['host'] = socket.gethostname()
        f.attrs['user'] = os.environ['USER']
        for k, v in self.additional_info.items():
            f.attrs[k] = str(v)
        return f

    def _try_save(self, done=False):
        if self.f is None:
            return
        self.f.attrs['done'] = done
        self.f.create_dataset("observations", data=np.asarray(self.obs_seq, dtype=np.float32), compression="gzip")
        self.f.create_dataset("actions", data=np.asarray(self.action_seq, dtype=np.float32), compression="gzip")
        self.f.create_dataset("rewards", data=np.asarray(self.rew_seq, dtype=np.float32), compression="gzip")
        self.f.create_dataset("times", data=np.asarray(self.time_seq, dtype=np.float32), compression="gzip")
        if hasattr(self.env, 'obs_names'):
            self.f.attrs['obs_names'] = str(self.env.obs_names)
        self.f.close()
        self.f = None




def train(num_timesteps, seed,
        model_path_load=None, 
        model_path_save=None,
        model_redis_save=None,
        hid_size=64,
        num_hid_layers=2,
        ts_per_batch=256, # orig 2048
        ts_per_minibatch=None,
        bLoad=True, visualize=False, dMoments=None):

    #env_id = 'Humanoid-v2'
    #env_id = 'RoboschoolHumanoidFlagrun-v1'
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__() # this starts a long-running session
    def policy_fn(name, ob_space, ac_space):
        #print("policy_fn ob_space:", ob_space.shape, ob_space)
        return mlp_policy.MlpPolicy(name=name, 
            ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, 
            num_hid_layers=num_hid_layers
            )
    #env = make_mujoco_env(env_id, seed)

    # this is just to get obs and action space
    env = make_wrapped_env(
        visualize=visualize, 
        run_logs_dir = ("./" if model_path_save is None else model_path_save) +"/run_logs/", # problem with this arg not set!
        dMoments=dMoments
        )

    ob_space = env.observation_space
    ac_space = env.action_space

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, 0.1)
    #env = RewScale(env, 1) # jw

    nCpu = 0 # MPI.COMM_WORLD.Get_size()
    nRank = 0 # MPI.COMM_WORLD.Get_rank()
    
    if num_timesteps==1:
        ts_per_batch = 64 
    
    print("Rank {:} nCpu: {:} ts_per_batch {:}".format(
        nRank, nCpu, ts_per_batch))

    if False:
        pi = pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=ts_per_batch,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, 
                optim_stepsize=3e-4, 
                optim_batchsize=64, 
                gamma=0.99, 
                lam=0.95,
                schedule='linear',
                model_path_load = model_path_load,
                model_path_save = model_path_save
            )
    else:
        print("calling ppolearner")
        pi = ppolearner.learn(ob_space, ac_space, policy_fn,
                max_timesteps=num_timesteps,
                ts_per_batch=ts_per_batch,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, 
                optim_stepsize=3e-4, 
                ts_per_minibatch=ts_per_minibatch, # none means use a single batch
                gamma=0.99, 
                lam=0.95,
                schedule='linear',
                model_path_load = model_path_load,
                model_path_save = model_path_save,
                model_redis_save = model_redis_save,
            )

    env.close()

    # Try not to overwrite previously saved state if we want to Load:
    if model_path_save:
        U.save_state(model_path_save)
        
    return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale

    




def main():
    logger.configure()
    nRank = 0# MPI.COMM_WORLD.Get_rank()
    parser = mujoco_arg_parser()
    parser.add_argument('--model-path-load', default=None, help="file path to load model weights from (learner)")
    parser.add_argument('--model-redis-load', default=None, help="redis path actor loads model weights from")
    parser.add_argument('--model-path-save', default=os.path.join(logger.get_dir(), 'model-ppo2'), help="file path learner saves weights to")
    parser.add_argument('--model-redis-save', default="tf:ppo1", help="redis path learner saves weights to")

    parser.add_argument("--hid-size", default=64, type=int, help="hidden layer size / width")
    parser.add_argument("--num-hid-layers", default=2, type=int, help="number of hidden layers")
    parser.add_argument("--ts-per-batch", default=4096, type=int, help="ts read by learner, across multiple segments")
    parser.add_argument("--ts-per-minibatch", default=None, type=int, help="ts per minibatch")
    parser.add_argument("--step-timeout", default=10, type=int, help="step timeout in actor")
    parser.add_argument("--nMaxSegTime", default=60, type=int, help="segment timeout in actor")
    parser.add_argument("--nMaxSegSteps", default=64, type=int, help="max steps yielded in segment by actor ")
    parser.add_argument("--integrator-accuracy", default=5e-5, type=float)


    #parser.add_argument('--load', dest="load", action="store_true")
    #parser.set_defaults(load=False)

    parser.add_argument("--moments", help="file path to moments json file")
    parser.add_argument("--train", dest="train", action="store_true", help="run learner training")
    parser.set_defaults(train=False)

    parser.add_argument("--submit", dest="submit", action="store_true")
    parser.set_defaults(submit=False)

    parser.add_argument("--token")
    parser.add_argument("--url", default="http://localhost:8050")


    parser.set_defaults(num_timesteps=int(2e7))
   
    args = parser.parse_args()
    
    print ("play: ", args.play)
    print("moments", args.moments)
    print("model_path_load:", args.model_path_load)
    print("model_path_save:", args.model_path_save)
    print("model_redis_load:", args.model_redis_load)
    print("model_redis_save:", args.model_redis_save)
    print("ts_per_minibatch:", args.ts_per_minibatch)


    if args.moments is not None:
        with open(args.moments, "r") as fMoments:
            dMoments = json.load(fMoments)
            #print("Moments loaded - but not yet implemented")
            #print(dMoments)
    else:
        dMoments=None

    if args.train:
        # train the model
        # this will be run multiple times by mpirun
        try:
            train(num_timesteps=args.num_timesteps, 
                seed=args.seed, 
                model_path_load = args.model_path_load,
                model_path_save = args.model_path_save,
                model_redis_save = args.model_redis_save,
                hid_size = args.hid_size,
                num_hid_layers = args.num_hid_layers,
                ts_per_batch = args.ts_per_batch,
                ts_per_minibatch = args.ts_per_minibatch,
                dMoments = dMoments,
                bLoad = False
                )
        except:
            print("env %i Exception during training" % nRank)
            exc_info = sys.exc_info()
            print(1)
            traceback.print_exception(*exc_info)
            print(2)
            oType, oValue, oTraceback = exc_info
            print(3)
            traceback.print_tb(oTraceback)
            sys.stderr.write("stderr\n")
            sys.stderr.flush()

            print(4)
            sys.exit()


    if args.play:
        print("Play!")
        # construct the model object, load pre-trained model and render
        # This also creates its own env.

        bVisualize = False

        # works
        #print ("env %i testing errors play 0a" % nRank)
        #print (1/0)

        pi = train(num_timesteps=1, # indicates no learning reqd
            seed=args.seed, 
            visualize=bVisualize,
            hid_size = args.hid_size,
            num_hid_layers = args.num_hid_layers,

            dMoments=dMoments,
            #model_path=args.model_path,
            bLoad=True)


        # works
        #print ("env %i testing errors play 0b" % nRank)
        #print (1/0)

        #if args.model_redis_load:
        #    print("Loading model from redis", args.model_redis_load)
        #    tfutil.RedisLoad(args.model_redis_load)
        #else:
        #    print("Loading model from files:", args.model_path_load)
        #    U.load_state(args.model_path_load)

        #env = make_mujoco_env('Humanoid-v2', seed=0)
        #env = make_mujoco_env('RoboschoolHumanoid-v1', seed=0)
        #env = make_mujoco_env('RoboschoolHumanoidFlagrunHarder-v1', seed=0)
        #env = make_mujoco_env('RoboschoolHumanoidFlagrun-v1', seed=0)

        # This does not contain a "RewScale"...
            
        env = make_wrapped_env(visualize=bVisualize,
                dMoments=dMoments,
                step_timeout=args.step_timeout,
                seed=args.seed, 
                integrator_accuracy=args.integrator_accuracy)
        env = RewScale(env, 0.1)

        
        #print ("env %i testing errors play 0c" % nRank)
        #print (1/0)

        # temporary - this doesn't return.
        ppoagent.traj_segment_generator(pi, env, 
            nMaxLen= args.nMaxSegSteps,
            bStochastic=True,
            rMaxTime = args.nMaxSegTime, 
            sqRedis="trajsegs",
            sPiRedis = "tf:ppo1"
            )

        if False:
            ob = env.reset()        
            iStep = 0
            iEp = 0
            rCumRew = 0

            # fails
            print ("env %i testing errors play 0d" % nRank)
            #print (1/0)


            for nStep in range(10000):
                if ob is None:
                    print("ob none: play skipping / waiting for env to step")
                else:
                    action = pi.act(stochastic=False, ob=ob)[0]
                ob, reward, done, _ =  env.step(action)
                rCumRew += reward
                print(iEp, iStep, reward, rCumRew)
                iStep += 1

                #if iStep > 10:
                #    print ("env %i testing errors play 1a" % nRank)
                #    print (1/0)

                #env.render()
                if done:
                    print("Episode:", iEp, "CumRew:", rCumRew)
                    ob = env.reset()
                    iEp +=1
                    iStep = 0
                    rCumRew = 0

                    #print ("env %i testing errors play 1b" % nRank)
                    #print (1/0)



        
    if args.submit:    
        print("Submit!")

        dummy_env = make_wrapped_env(visualize=False)

        remote_base = args.url

        client = Client(remote_base)
        #observation = client.env_create(args.token, env_id="Run")
        client.env_create(args.token, env_id="ProstheticsEnv")

        client_env = ClientToEnv(client)
        client_env = DictToListLegacy(client_env)
        client_env = ObsProcessor(client_env, dMoments=dMoments)
        client_env = JSONable(client_env)

        print ("Dir of client_env:", dir(client_env))
        
        #print(type(observation), len(observation), observation)

        # dummy training to create agent, policy, network
        pi = train(num_timesteps=1,
            seed=args.seed, visualize=False,
            dMoments=dMoments,
            #model_path=args.model_path
            )
        U.load_state(args.model_path_load)



        # Create environment

        observation = client_env.reset()

        iEp = 0
        iStep = 0
        rCumRew = 0
        reward = 0

        # Run a single step
        # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
        iStep = 0
        while True:
            print("Ep:", iEp, "step:", iStep, "Rew: ", reward, "CumRew: ", rCumRew, "obs:", type(observation), len(observation))
            #v = np.array(observation).reshape((dummy_env.observation_space.shape[0]))
            #action = agent.forward(v)
            action = pi.act(stochastic=False, ob=np.array(observation))[0]
            [observation, reward, done, info] = client_env.step(action.tolist())
            rCumRew += reward
            iStep += 1
            if done:
                print("Ep ", iEp, "done. CumRew: ", rCumRew)
                iEp += 1
                reward = 0
                rCumRew = 0
                iStep = 0
                observation = client_env.reset()
                if observation is None:
                    print("Observation null - break loop")
                    break
            
                
        print("Submit")
        print("client_env:", client_env)

        client_env.submit()


if __name__ == '__main__':
    main()
