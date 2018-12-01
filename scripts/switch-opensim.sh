#!/bin/bash


case "$1" in
new) bTest=false; sDirLink=/home/jeremy/opensim_install2/lib;;
old) bTest=false; sDirLink=./saf;;
test) bTest=true;;
*) 
    echo "Specify old, new or test for version of opensim"
    exit 1
esac

#lFiles="libosimActuators.so libosimAnalyses.so libosimCommon.so libosimExampleComponents.so libosimJavaJNI.so libosimLepton.so libosimSimmFileWriter.so libosimSimulation.so libosimTools.so"
lFiles="libosimActuators.so libosimAnalyses.so libosimCommon.so libosimExampleComponents.so libosimJavaJNI.so libosimLepton.so libosimSimulation.so libosimTools.so"

echo "sDirLink $sDirLink"
echo "bTest $bTest"

echo lib:
cd ~/miniconda3/envs/osim361/lib

if [ `pwd` = "/home/jeremy/miniconda3/envs/osim361/lib" ]
then
    echo In correct folder: `pwd`
else
    echo NO - wrong folder
    pwd
    exit 1
fi

#ls -l libosim*
if [ $bTest = false ]; then
    for sFile in $lFiles
    do
        #echo file: $sFile
        if [ -e $sFile ] # if it exists
        then 
            if [ -L $sFile ] # if it's a link
            then 
                echo Removing link: $sFile
                rm $sFile
                ln -s $sDirLink/$sFile .
            else 
                echo $sFile exists but NOT A LINK!
                ls -l $sFile
            fi
        else
            ln -s $sDirLink/$sFile .
        fi
    done
fi

ls -l libosim*

echo site-packages/opensim:
cd ~/miniconda3/envs/osim361/lib/python3.6/site-packages/opensim

if [ `pwd` = "/home/jeremy/miniconda3/envs/osim361/lib/python3.6/site-packages/opensim" ]
then
    echo In correct folder: `pwd`
else
    echo NO - wrong folder
    pwd
    exit 1
fi


#ls -l libosim*

if [ $bTest = false ]; then
    for sFile in $lFiles
    do
        if [ -e $sFile ] # if it exists
        then 
            if [ -L $sFile ]
            then 
                echo Removing link: $sFile
                rm $sFile
                ln -s $sDirLink/$sFile .
            else 
                echo $sFile exists but NOT A LINK!
                ls -l $sFile
            fi
        else
            ln -s $sDirLink/$sFile .
        fi

    done
fi

ls -l libosim*