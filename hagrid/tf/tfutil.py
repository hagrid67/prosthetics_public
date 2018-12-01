

import tensorflow as tf
import tensorflow.contrib as tc
import redis, pickle

# copied / forked from sosrl.models, some of which was from from ctmakro cans

class TFModel(object):

    
    def __init__(self, name, srPath=None):
        """ just initialise with a name, used for the tf top-level scope
        """
        self.name = name
        self.lPH=[]
        self.lopAssign = []
        
        self.dVars = {}
        self.dPH = {}
        self.dOpAssign = {}
        self.srPath = srPath
    
    
    def createVarList(self):
        """ Save the variable list as soon as the network is created,
            to avoid reading/storing new ops created by assignment ops
        """
        lVars = self.lVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        lPH = self.lPH = [] #  placeholders for loading values
        lopAssign = self.lopAssign = [] # assign operations for copying from PHs to vars
        dVars = self.dVars
        dOpAssign = self.dOpAssign
        dPH = self.dPH

        print("Setting up placeholders for assignment", self.name)
        for oVar in lVars:
            sName = oVar.name.split(":")[0] # remove the ":0" from the end...
            oPH = tf.placeholder(dtype=oVar.dtype, shape=oVar.shape, name=sName+"_ph")
            opAssign = oVar.assign(oPH)
            
            lPH.append(oPH)
            dPH[sName] = oPH
            
            lopAssign.append(opAssign)
            dOpAssign[sName] = opAssign
            
            dVars[sName] = oVar
            
            print(oPH, opAssign)


    def getValues(self, sess):
        #print("in getValues: ", self.name)
        lVals = sess.run(self.lVars) # get all the values
        dVals = {}
        for iVar, oVar in enumerate(self.lVars):
            oVal = lVals[iVar]
            #print(oVar, oVar.name, type(oVal), oVal.shape)
            sName = oVar.name.split(":")[0]
            dVals[sName] = oVal

        #print("Done getValues")
        return dVals

    def loadValues(self, lVals, sess):
        if len(lVals) != len(self.lVars):
            print("Lengths of variables and values differ")
            assert(len(lVals) == len(self.lVars))

        dFeed = {}
        print("loadValues ", self.name)

        # Create feed dict of placeholders->values
        for iVar, oPH in enumerate(self.lPH):
            #oVar = self.lVars[iVar]
            oVal = lVals[iVar]
            #print(oPH, oVar.name,  oVal.shape)
            dFeed[oPH] = oVal
        
        # run the assigments
        # Vars <- AssignOps <- Placeholders <- Vals
        sess.run(self.lopAssign, feed_dict=dFeed)

    def loadValuesDict(self, dVals, sess):
        if len(dVals) != len(self.lVars):
            print("Lengths of variables and values differ")
            #assert(len(dVals) == len(self.lVars))

        dFeed = {}
        lOpAssign = []

        print("loadValues ", self.name)

        # Create feed dict of placeholders->values
        for sName in dVals.keys():
            #oVar = self.lVars[iVar]
            oPH = self.dPH[sName]
            opAssign = self.dOpAssign[sName]
            lOpAssign.append(opAssign)
            oVal = dVals[sName]
            #print(oPH, oVar.name,  oVal.shape)
            
            dFeed[oPH] = oVal
        
        # run the assigments
        # Vars <- AssignOps <- Placeholders <- Vals
        sess.run(lOpAssign, feed_dict=dFeed)



    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    def saveToRedis(self, sRedis=None, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        if sRedis is None:
            sRedis = self.srPath
        dVals = self.getValues(sess)
        pkVals = pickle.dumps(dVals)
        fR = redis.Redis()
        fR.set(sRedis, pkVals)

    def loadFromRedis(self, sRedis=None, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        if sRedis is None:
            sRedis = self.srPath
        fR = redis.Redis()
        sModelVals = fR.get(sRedis)
        dVals = pickle.loads(sModelVals)
        self.loadValuesDict(dVals, sess)

    def loadFromRedisQ(self, sRedis, iModel=None, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        if iModel is None:
            iModel = getModelIndex()

        fR = redis.Redis()
        sModelVals = fR.lindex(sRedis, iModel)
        dVals = pickle.loads(sModelVals)
        self.loadValuesDict(dVals, sess)

    def getModelIndex(self):
        fR = redis.Redis()
        srPath2 = self.srPath + "Q:iNext"
        nIndex = int(fR.get(srPath2))
        return nIndex

    def incModelIndex(self):
        fR = redis.Redis()
        srPath2 = self.srPath + "Q:iNext"
        nIndex = int(fR.get(srPath2))
        nIndex += 1
        fR.set(srPath2, nIndex)
        return nIndex


def RedisLoad(sRedis):
    fR = redis.Redis()
    sModelVals = fR.get(sRedis)
    dVals = pickle.loads(sModelVals)
    sess = tf.get_default_session()
    omPi = TFModel("pi")
    omPi.createVarList()
    #dVals = omPi.getValues(sess)
    omPi.loadValuesDict(dVals, sess)

def RedisSave(sRedis):
    fR = redis.Redis()
    sess = tf.get_default_session()
    omPi = TFModel("pi") # assumes model already created and loaded
    omPi.createVarList()
    dVals = omPi.getValues(sess)
    pkVals = pickle.dumps(dVals)
    fR.set(sRedis, pkVals)
