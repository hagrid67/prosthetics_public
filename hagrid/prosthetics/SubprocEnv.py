
import os
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
import pickle
import tensorflow as tf
import psutil




#  SubprocEnv(creates Process) ->
#  PipeMain -> 
#  ProstheticsEnv



# Function which is run in a separate process that holds a RunEnv/ProstheticsEnv instance.
# (Apparently) This has to be done since RunEnv() in the same process results in interleaved running of simulations.
def PipeMain(
        conn, 
        visualize=False, 
        dEnvConfig={},
        seed=None
        ):
    try:
        
        env = ProstheticsEnv(visualize=visualize,
            difficulty = 1, 
            seed=seed,
            dEnvConfig=dEnvConfig,
            )

        while True:
            lMsg = conn.recv() # wait for command

            # messages should be tuples,
            # msg[0] should be string

            if lMsg[0] == 'reset':
                dObsReset = env.reset(project=False, ) # Oct 8
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


class SubprocEnv(gym.Env): # (gym.Wrapper): # jw - prev had no super
    """ Create subprocess holding env (using standalone_headless_isolated)
        Handle multiprocess pipe comms to/from subprocess
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, visualize=False,
            dEnvConfig = {},
            ):
        self.running = False

        #self.action_space = Box(low=0, high=1, shape=[19], dtype=np.float32) # shape=[18])
        self.action_space = Tuple([Discrete(2)] * 19) # shape=[18])

        lObsDummy = preprocessors.DictToListFull(None, dEnvConfig=dEnvConfig, bDummy=True)
        self.nObsSize = len(lObsDummy)

        self.observation_space = Box(low=-5, high=+5, shape=[self.nObsSize], dtype=np.float32) # shape=[41])
        self.reward_range = [0, 10000]
        self.visualize = visualize

        self.oPipeClient = None
        self.oPipeServer = None

        self.tLastObs = None
        self.oProc = None
        self.nPidChild=0

        self.dEnvConfig = dEnvConfig





    def reset(self, difficulty=None):

        # kill the child process if it exists already
        if self.running:
            self.oPipeClient.send(('close',))
            self.oProc.join()

        while True:

            self.oPipeClient, self.oPipeServer = Pipe()
            # set the seed on creation in the worker process
            # time x 1e9 ie 10-millionths of a second 
            # include the pid in case of simultaneous creation across CPUs, machines
            seed = int(1e9*(time.time() - int(time.time()))) + os.getpid() + hash(socket.gethostname())

            self.oProc = Process( ########################### New Process here ####################
                target=PipeMain, # running env created in child process here
                args=(self.oPipeServer,),
                kwargs=dict(
                    visualize=self.visualize, 
                    dEnvConfig=self.dEnvConfig,
                    seed = seed
                    ),
                )
            self.nPidChild = self.oProc.pid
            self.oProc.start()
            self.oPipeClient.send(('reset', difficulty, seed), )

            finished = self.oPipeClient.poll(10) # Allow 10 seconds for reset.  2 is usually enough.
            if finished: # reset finished on time
                tRes = self.oPipeClient.recv() # receive the output of the reset (dObs)

                # don't return an exception - just keep trying
                if isinstance(tRes, Exception):
                    dtNow = datetime.now()
                    print('env {:} Env reset exception at {}'.format(self.nPidChild, dtNow.strftime("%m%d-%H%M%S")))
                    self.oProc.terminate() # jw
                    self.oProc.join()

                else:
                    break
            
            else: # the reset timed out.
                dtNow = datetime.now()
                print('env {:} Env reset Timed out at {}'.format(self.nPidChild, dtNow.strftime("%m%d-%H%M%S")))
                print("terminating...")
                self.oProc.terminate() # jw
                self.oProc.join()

            # carry on trying            

        self.running = True

        
        #return preprocessors.DictToListFull(res, 
        #    dEnvConfig=self.dEnvConfig)
        self.tLastObs = tRes
        return tRes

    def step(self, actions):
        self.oPipeClient.send(('step', list(actions)), )
        tRes = self.oPipeClient.recv()
        if isinstance(tRes, Exception):
            dtNow = datetime.now()
            print('env {:} Env step exception at {}'.format(self.nPidChild, dtNow.strftime("%m%d-%H%M%S")))
            self.tLastObs[2] = True # done
            return self.tLastObs

        self.tLastObs = tRes
        return tRes            



    def close(self):
        if self.oPipeClient is not None:
            self.oPipeClient.send(('close',))
            res = self.oPipeClient.recv()
            if isinstance(res, Exception):
                raise res

    def __del__(self):
        if self.oPipeClient is not None:
            self.oPipeClient.send(('close',))
        return

