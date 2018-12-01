

import os
#from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
#from baselines.common import tf_util as U
#from baselines import logger

from mpi4py import MPI
#from baselines.bench import Monitor
#from baselines.common import set_global_seeds


#from osim.http.client import Client
#from helper.wrappers import ClientToEnv, DictToListLegacy, ForceDictObservation, JSONable
#from helper.wrappers.Wrapper import EnvironmentWrapper

#import roboschool
#from osim.env.osim import ProstheticsEnv
#import atexit, h5py
from os import path
import time, socket
import numpy as np

import random
from datetime import datetime
#import gym
#from gym.spaces import Box

from multiprocessing import Process, Pipe
import json
import sys

import traceback


#def longrunning():
#    for i in range(100):
#        time.sleep(1)
#        sys.stdout.write(".")


def standalone_headless_isolated(conn):
    try:
        iStep = 0

        while True:
            msg = conn.recv() # wait for command

            print ("msg: ", msg)
            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':
                print("Reset")
                iStep = 0
                #o = env.reset(difficulty=msg[1], seed=msg[2])
                #print("Reset done")
                conn.send((0.1, 0.1, False, {} ) )

            elif msg[0] == 'step':
                print ("step")
                time.sleep(1)
                iStep += 1
                #conn.send("got step %i" % iStep)
                conn.send( (0.1, 0.1, iStep > 5, {} ) )

            elif msg[0] == 'close':
                #env.close()
                conn.send(None)
                return

    except Exception as oEx:
        import traceback
        print(traceback.format_exc())
        conn.send(oEx)




class parent(object):

    def __init__(self):
        self.running = False
        self.step_timeout=2
        self.nMpiRank = MPI.COMM_WORLD.Get_rank()
        

    def reset(self, difficulty=None, seed=None):
        if self.running:
            print ("env %i running - close and join" % self.nMpiRank)
            self.pc.send(('close',))
            self.p.join()
        else:
            print ("env %i Not running - no join" % self.nMpiRank)

        print ("env %i create new process" % self.nMpiRank)
        self.pc, self.cc = Pipe()
        self.p = Process(
            target=standalone_headless_isolated,
            args=(self.cc,),
            )
        # self.p.daemon = True
        self.p.start()
        self.pc.send(('reset', difficulty, seed))
        self.running = True

        res = self.pc.recv()
        print ("env %i received: %s" % (self.nMpiRank, str(res)))
        self.lastobs = res
        if isinstance(res, Exception):
            raise res
        return res

    def step(self):
        self.pc.send(('step',))

        finished = self.pc.poll(self.step_timeout)

        if finished: # ie the step finished on time
            self.bTimeout=False
            res = self.pc.recv()
            if isinstance(res, Exception):
                raise res
            self.lastobs = res[0]
            #print(self.nMpiRank, type(res), len(res))
            return res

        else: # the step timed out.
            self.bTimeout=True
            dtNow = datetime.now()
            print('Actor {:} Env Timed out {:}s at {}'.format(self.nMpiRank, self.step_timeout, dtNow.strftime("%m%d-%H%M%S")))
            #self.p.terminate() # jw
            #return self.lastobs, -0.045, True, {"Timed out": True}
            tRV = (None, -0.045, False, {"Timed out": True})
            #print("returning:", tRV)
            #raise ValueError("env timeout")
            return tRV


def main():
    nRank = MPI.COMM_WORLD.Get_rank()

    #print("env %i test error 3" % nRank)
    #print (1/0)

    sys.stderr.write("env %i stderr 1\n" % nRank)
    sys.stderr.flush()

    oParent = parent()

    #print("env %i test error 4" % nRank)
    #print (1/0)

    sys.stderr.write("env %i stderr 2\n" % nRank)
    sys.stderr.flush()

    oParent.reset()

    sys.stderr.write("env %i stderr 3\n" % nRank)
    sys.stderr.flush()

    sys.stderr.write("env %i test error 2\n" % nRank)
    sys.stderr.flush()
    try:
        rDummy = 1/0
    except: # Exception as oEx:
        #print(traceback.format_exc())
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        oType, oValue, oTraceback = exc_info
        traceback.print_tb(oTraceback)
        sys.exit()


    print("Starting step loop")
    while True:
        tRV = oParent.step()
        print ("env %i step got: %s" % (nRank, tRV))
        sys.stderr.write("env %i stderr 4\n" % nRank)
        sys.stderr.flush()

        (rew, ob, bDone, dInfo) = tRV
        if bDone:
            oParent.reset()
            sys.stderr.write("env %i stderr 5\n" % nRank)
            sys.stderr.flush()
            print("test error 1")
            print (1/0)

def main2():
    main()

if __name__ == "__main__":
    main2()