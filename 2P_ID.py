import sys
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
from scipy.ndimage import median_filter

"""Two Photon Arteriole/Venule Identification
Designed to identify arterioles and venules using the bolus through the 
Function call - 2P_ID.py <VE_filename> <Bolus_filename>
I still need to reduce the number of imports and see if I can expedite the process
Save out vessels with 0 being arterioles and 1 being venules"""

def trans(img):
    tSave = np.transpose(img,(2,1,0))
    tSave = np.rot90(tSave,3,axes=(1,2))
    tSave = np.flip(tSave,2)
    return tSave

### Maybe add a function for masking the FITC or picking 3-4 slices to determine time
def analyzeVessel(vessel, bStack):
    [sliceNums, sliceWidth, sliceHeight] = bStack.shape
    means = np.zeros((1, sliceNums))
    for num, t in enumerate(bStack): # Analyze each individual time point
        mask = t*vessel
        mask[mask == 0] = np.nan
        means[num] = np.nanmean(mask)
    return means

def eventID(ms):
    # Identifies the instance/slice of the most likely bolus event
    # Uses the slope, could use maximum, might be more effective?
    oldSlope = 0
    for i in range(1,len(ms)-1):
        newSlope = ms[i]-ms[i-1]
        if(newSlope > oldSlope):
            oldSlope = newSlope
            index = I
    return index


# Structure of function call: python 2P_ID.py <filtered filename> <> <>

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

veFN = sys.argv[1]
bolFN = sys.argv[2]

# Threshold with rhodamine to identify vessels
tif = TiffFile(veFN)
ve = tif.asarray() # Only has extracted vessels
tif = TiffFile(bolFN)
bolScan = tif.asarray()
bol = bolScan[0] # Only need FITC scan for bolus analysis, assuming all slices have been registered

[imSlices, imHeight, imWidth] = ve.shape

# Parameter Extraction from filename - would like to do it from TIFF tags, but running into issues
# Example filename - 2022-03-29_Baseline_Stack_1_lam_880nm_eom_100_power_6_75_pmt_56_size_400x400mic_pixels_510x510_freq_800_LinAvg_1_range_0mic-neg200mic_slice_1micPMT - PMT [HS_1] _C6.ome
# Might actually need it here
width = int(veFN[veFN.find('size_')+5:veFN.find('size_')+8])
height = int(veFN[veFN.find('mic')-3:veFN.find('mic')])
depth = int(veFN[veFN.find('slice_')+6:veFN.find('micPMT')])
frame = int(bolFN[bolFN.find('PMT -')-2:bolFN.find('PMT -')]) # IDs the frame the bolus is shot in


# Analyze each vessel labeled individually
start = time.perf_counter()
vMask = ve[frame,...]
label_ve = label(vMask)
vNum = np.max(label_ve)
means = np.zeros((vNum, imSlices))
events = np.zeros(vNum)
for v in range(1, vNum+1):
    vessel = np.zeros_like(label_ve)
    idx = np.where(label_ve)
    vessel[idx] = 1 # Now we have just a mask of a singular vessel
    vSlice = vessel[frame]
    means[v-1,...] = analyzeVessel(vSlice, bol)
    # Maybe smooth linear time profile here?
    means[v-1] = median_filter(means[v-1], 3) # Adjust number for filtering
    # Identify the largest slope in means (eventID)
    events[v-1] = eventID(means[v-1])

# Now pick out events and then separate them into arterioles and venules

# Now I need to smooth the linear time profile of the means and then determine the index of their greatest + slope (bolus change)

print(time.perf_counter()-start)