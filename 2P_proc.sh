#!/bin/bash
# HPC Module load
# module load python/3.6
echo -n "Name of directory for scan:";
read dir;
echo $dir;
# Define scan time
for SCAN in ${"$dir/*_C6.ome.tif"}; do
# Check and only upload 1 of the 
    echo $SCAN
    export SCAN
    prefix=$(echo $SCAN| cut -d '_PMT' -f 1)
    python 2P_DA.py $SCAN
    python 2P_WF.py $SCAN
    WF="${prefix}_WF.tif"
    python 2P_IR.py $WF
    IR="${prefix}_IR.tif"
    # Take the prefix for scan and then assign a new value to it
    if [[$file = *"Timelapse"*]|[$file = *"Bolus"*]]; then
        echo Processing Timelapse
        echo ${"Finished processing ${prefix}"}
    else
        echo Processing Z-stack
        # Should process Timepoints and all Z-stacks
        OT="${prefix}_OT.tif"
        #VE="${prefix}_VE.tif"
        #python 2P_AR.py $SCAN
        python 2P_OT.py $IR
        python 2P_VE.py $OT
        echo ${"Finished processing ${prefix}"}
