import sys, glob, os
import math
import time
import warnings
import numpy as np
from tifffile import imwrite
from tifffile import TiffFile
from matplotlib import pyplot as plt
from skimage import color, data, restoration, exposure, io
from skimage.morphology import disk, reconstruction
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

### Call all functions in necessary pipeline... maybe.
### Might be easier to bashscript

maindir = sys.argv[1]
keyFiles = glob.glob(maindir + '*_C7*')

for filename in keyFiles:
    print("Processing " + filename)
    prefix = filename[0:filename.find('PMT')]
    #print("Processing " + prefix)
    os.system("python 2P_DA.py \"" + filename + "\"")
    os.system("python 2P_WF.py \"" + filename + "\"")
    if ("timeseries" in filename) or ("bolus" in filename):
        print("This is a timeseries")
        os.system("python 2P_IR.py " + prefix + "_WF.tif")
        print("Finished processing this timeseries")
    else:
        print("This should be a Z Stack")
        #Could throw in AR here if it ever happens
        os.system("python 2P_OT.py " + prefix + "_WF.tif")
        os.system("python 2P_VE.py " + prefix + "_OT.tif")


# Now we need to figure out how to run 2P_ID
bolus = glob.glob('*bolus*IR*')
vessels = glob.glob('*VE*')
print(vessels)
keyVessel = input("Put in the index number of the first scan: ")
os.system("python 2P_ID.py " + vessels[keyVessel] + " " + bolus)