
import boto3
import pandas as pd
import re
import datetime as dt
import os
from subprocess import Popen, check_output, PIPE

#import argparse
from clize import Parameter, run

def getRunning():
    """ Get the running hosts / instances as a pandas DataFrame, indexed in order of launch time.
    """ 
    ec2 = boto3.resource("ec2")
    lFields = re.split("\s+", "id instance_type public_ip_address public_dns_name launch_time")
    dtNow = dt.datetime.utcnow()
    lInstance = ec2.instances.filter( Filters=[
    {'Name': 'instance-state-name', 'Values': ['running']},
    #{"Name": "instance-type", "Values":["c4.8xlarge"]}
    ]) 
    dfInst = pd.DataFrame([ 
        map(oInst.__getattribute__, lFields)
        for oInst in lInstance ],
        columns=lFields).sort_values("launch_time")
    dfInst["uptime"] = dt.datetime.now(tz=dt.timezone.utc) - pd.DatetimeIndex(dfInst.launch_time, tz=dt.timezone.utc) 
    #dfInst["inst"] = [o for o in lInstance ]
    dfInst.sort_values("uptime", inplace=True, ascending=False)
    dfInst.reset_index(inplace=True, drop=True)
    #dfInst.iloc[:, :-1].style.format(dFormat)`

    return dfInst


def fixSsh():
    """ Fix the ~/.ssh/config file to include the AWS hosts as am0, am1 etc, 
    ordered by instance creation time.
    """
    dfInst = getRunning()
    sConfig = """
Host * 
    StrictHostKeyChecking no

"""

    for iHost, sHost in enumerate(dfInst.public_dns_name):
        sEntry = """Host am{:}
HostName {}
User ubuntu
Port 22
""".format(iHost, sHost)
        sConfig += sEntry + "\n"

    with open(os.environ["HOME"] + "/.ssh/config-ec2", "w") as fOut:
        fOut.write(sConfig)

    s1=""
    for sFile in ["config-home", "config-ec2"]:
        with open(os.environ["HOME"] + "/.ssh/" + sFile, "r") as f1:
            s1+=f1.read()

    with open(os.environ["HOME"] + "/.ssh/config", "w") as fOut:
        fOut.write(s1)

def spotRequest(sType="c4.4xlarge"):
    """ Make a spot request for instance type sType
    :param sType: Instance type, eg c4.4xlarge, p3.2xlarge, etc
    """
    client = boto3.client('ec2')
    sToken = "Dummy-" + dt.datetime.now().strftime("%y%m%d-%H%M%S")
    print("Token:", sToken)

    response = client.request_spot_instances(
        DryRun=False,
        SpotPrice='1.10', # '0.90',   # 2018/06/20 increased from 90c to 110c 
        ClientToken=sToken, # change this to a new value each time...
        InstanceCount=1,    # instances
        Type='one-time',
        LaunchSpecification={
            #'ImageId': "ami-fa6e7d9c",  # this is our cuda AMI
            'ImageId': "ami-6babae12", # amazon deep learning v6 ubuntu 16.04

            'KeyName': 'jwkp1',
            'SecurityGroups': ['default'],
            # instance type -
            # c4.large = 2 core, 3c/hr (3.06)
            # c4.xlarge = 4 core, 7c/hr (now 6c/h)
            # c4.2xlarge = 8 core,  13c/hr (12c/h)
            # c4.4xlarge = 16 core, 25c/hr (24c/h)
            # c4.8xlarge = 36 core, 50c/hr (48c/h) ($12/day)
            # p2.xlarge = 30c/hr GPU k80 2496 12gb 4vCPU 60Gb
            # g2.2xlarge = 21c/hr - legacy?
            # p3.2xlarge = $1/h (was 50c/hr) 8 core, 1 new V100 GPU
            'InstanceType': sType,
            #'InstanceType': 'p2.xlarge',   
            #'InstanceType': 'p3.2xlarge',   
            'Placement': {
                'AvailabilityZone': 'eu-west-1b',  # my region
            },
            "BlockDeviceMappings": [
                {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeType": "gp2",
                    "VolumeSize": 64, # 32,
                    #"SnapshotId": "snap-0c35c3619e4fdbf8f"
                }
                }
            ],
            'EbsOptimized': True,
            'Monitoring': {
                'Enabled': False # was True
            },
            'SecurityGroupIds': [
                'sg-ab160bcc'  # same sec group as default
            ]
        }
    )
    print(response)




def do_nothing():
    print("No args specified")

def main():
    run(getRunning, spotRequest, fixSsh)

if __name__ == "__main__":
    main()

