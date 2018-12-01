from googleapiclient import discovery
import libcloud
import os, sys
from clize import Parameter, run
import argh
import pandas as pd
import hagrid.aws.host
import subprocess
from glob import glob


# GCP API 
#def getRunning1():
#    compute = discovery.build('compute', 'v1')
#    oList = compute.instances().list(project="gcp-hagrid", zone="us-east1-b").execute()
#    #print(oList)


def getRunning(bDebug=False, sProject="hagrid"):
    """ get running GCP nodes using libcloud
    """
    cls = libcloud.get_driver(libcloud.DriverType.COMPUTE, libcloud.DriverType.COMPUTE.GCE)

    if sProject=="vernal":
        driver = cls("28837673100-compute@developer.gserviceaccount.com",
                project="vernal-design-171120",
                key="/home/jeremy/.ssh/gcp-vernal.json"
                )

    elif sProject=="hagrid":
        driver = cls("hagrid1@gcp-hagrid.iam.gserviceaccount.com",
                project="gcp-hagrid",
                key="/home/jeremy/.ssh/gcp-hagrid1.json"
                )


    lNodes = driver.list_nodes()

    if bDebug:
        dfNodes = pd.DataFrame([
            [o.name, o.state, o.public_ips[0], o.size]
            for o in  lNodes],
            columns=["name", "state", "ip", "type"])
        print(dfNodes)
    return lNodes

def keepRunning(*lHosts): # NOT FINISHED!
    lHosts = list(lHosts)
    print("Keep Running:", lHosts)

    lNodes = getRunning()
    print("lNodes:", lNodes)
    dfNodes = pd.DataFrame([
            [o.name, o.state, o.public_ips[0], o.size]
            for o in  lNodes],
            columns=["name", "state", "ip", "type"])
    dfNodes = dfNodes[dfNodes.name.isin(lHosts)]

    lNodes[2]



def fixSsh(sProject="hagrid"):
    """ Fix the ~/.ssh/config file to include the AWS hosts as am0, am1 etc, 
    ordered by instance creation time.
    """
    lNode = getRunning(sProject=sProject)
    sConfig = """
Host * 
    StrictHostKeyChecking no

"""

    lIPs = []

    for oNode in lNode:
        print (oNode.name)
        sHost = oNode.public_ips[0]
        sEntry = """Host {}
HostName {}
User jeremy
""".format(oNode.name, sHost)  # #Port 22
        sConfig += sEntry + "\n"
        lIPs.append(sHost)

    with open(os.environ["HOME"] + "/.ssh/config-gcp", "w") as fOut:
        fOut.write(sConfig)

    s1=""
    for sFile in ["config-home", "config-ec2", "config-gcp"]:
        with open(os.environ["HOME"] + "/.ssh/" + sFile, "r") as f1:
            s1+=f1.read()

    print(sConfig)
    #print(s1)
    with open(os.environ["HOME"] + "/.ssh/config", "w") as fOut:
        fOut.write(s1)

    print("Removing from known_hosts...")
    for sIP in lIPs:
        if sIP is not None:
            lCmd = "ssh-keygen -f /home/jeremy/.ssh/known_hosts -R {}".format(sIP).split(" ")
            print(" ".join(lCmd))
            subprocess.run(lCmd)
    
def tidyRay(sDirPattern="/home/jeremy/ray_results/*"):
    lsDir = glob(sDirPattern)

    for sDir in lsDir:
        lsDirCP = glob(sDir+"/checkpoint*")
        if len(lsDirCP) == 0:
            print(sDir)
            
        else:
            print("not: ", sDir)


def main():
    # clize
    #run(getRunning, fixSsh, keepRunning)

    # argh
    parser = argh.ArghParser()
    parser.add_commands([getRunning, fixSsh, keepRunning, tidyRay])
    parser.dispatch()


if __name__ == "__main__":
    main()