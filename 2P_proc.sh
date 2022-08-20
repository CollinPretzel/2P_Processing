#!/bin/bash
# HPC Module load
# module load python/3.6
echo -n "Name of directory for scan:";
read dir;
echo $dir;
# Define scan time
for SCAN in $dir/*; do
    echo $SCAN
    export SCAN
    python 2P_DA.py $SCAN
    python 2P_WF.py $SCAN
    prefix=$(echo $SCAN| cut -d '_PMT' -f 1)
    WF="${prefix}_WF.tif"
    IR="${prefix}_IR.tif"
    # Take the prefix for scan and then assign a new value to it
    if [$file = *"Zstack"*]; then
        echo Processing Zstack
        OT="${prefix}_OT.tif"
        #VE="${prefix}_VE.tif"
        #python 2P_AR.py $SCAN
        python 2P_IR.py $WF
        python 2P_OT.py $IR
        python 2P_VE.py $OT
