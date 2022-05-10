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
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

"""Two Photon Arteriole/Venule Identification
Designed to align all images to reduce the impact of breathing or motion artifacts
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
    for num, t in enumerate(Stack): # Analyze each individual time point
        mask = t*vessel
        mask[mask == 0] = np.nan
        means[1,num] = np.nanmean(mask)
    return means


# Structure of function call: python 2P_Proc.py <filtered filename> <> <>

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


# Try registration without otsu thresholding
start = time.perf_counter()
vMask = ve[frame,...]
label_ve = label(vMask)
vNum = np.max(label_ve)
means = np.zeros((vNum, imSlices))
for v in range(1, vNum+1):
    vessel = np.zeros_like(label_ve)
    idx = np.where(label_ve)
    vessel[idx] = 1 # Now we have just a mask of a singular vessel
    means[v-1,...] = analyzeVessel(vessel, bol)


print(time.perf_counter()-start)