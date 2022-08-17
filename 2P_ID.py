import sys, csv
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
Save out vessels with 1 being arterioles and 2 being venules"""

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
            index = i
    return index


# Structure of function call: python 2P_ID.py <filtered filename> <> <>

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

veFN = sys.argv[1]
vePrefix = veFN[0:veFN.find('_VESSEL_Mask.tif')]
veCSV = vePrefix + '.csv'
bolFN = sys.argv[2]
bolPrefix = bolFN[0:bolFN.find('_PMT -')]
bolCSV = bolPrefix + '.csv'

# Threshold with rhodamine to identify vessels
tif = TiffFile(veFN)
ve = tif.asarray() # Only has extracted vessels
tif = TiffFile(bolFN)
bolScan = tif.asarray()
bol = bolScan[0] # Only need FITC scan for bolus analysis, assuming all slices have been registered

[imSlices, imHeight, imWidth] = ve.shape

# Import parameters from csv file
veParams = []
with open(veCSV, newline = '') as f:
    fReader = csv.reader(f, delimiter = ',')
    for row in fReader:
        veParams.append(row)

bolParams = []
with open(bolCSV, newline = '') as f:
    fReader = csv.reader(f, delimiter = ',')
    for row in fReader:
        bolParams.append(row)

veStart = veParams[1][3]
veEnd = veParams[1][4]
veRes = (veEnd - veStart)/imSlices # Units of um/slice
bolPos = bolParams[1][3]
frame = round((bolPos-veStart)/veRes) # Calculates the approximate frame number that correlates to bolus

# Analyze each vessel labeled individually
start = time.perf_counter()
label_ve = label(ve)
vMask = ve[frame,...]
label_vMask = label(vMask)
vNum = np.max(label_vMask)
means = np.zeros((vNum, imSlices))
filtMeans = means
events = np.zeros(vNum)
for v in range(1, vNum+1):
    vessel = np.zeros_like(label_vMask)
    idx = np.where(label_vMask == vNum, label_vMask)
    vessel[idx] = 1 # Now we have just a mask of a singular vessel
    means[v-1,...] = analyzeVessel(vessel, bol)
    # Maybe smooth linear time profile here?
    filtMeans[v-1] = median_filter(means[v-1], 3) # Adjust number for filtering
    # Identify the largest slope in means (eventID)
    events[v-1] = eventID(filtMeans[v-1])

# Now pick out events and then separate them into arterioles and venules
# Maybe assess the mean value of event times as an indicator of As and Vs
midpoint = events.mean()
arterioles = np.zeros_like(ve)
venules = np.zeros_like(ve)
for v in range(1, vNum+1):
    if events[v-1] > midpoint: # The bolus is flushing through a venule
        vIdx = np.where(label_ve == v, label_ve)
        venules[vIdx] = 1
    else:
        vIdx = np.where(label_ve == v, label_ve)
        arterioles[vIdx] = 1

# Save out vessel masks
artSave = trans(arterioles).astype('float32')
venSave = trans(venules).astype('float32')
artFN = vePrefix + '_art.tif'
venFN = vePrefix + '_ven.tif'
imwrite(artFN, artSave, photometric='minisblack')
imwrite(venFN, venSave, photometric='minisblack')
print(time.perf_counter()-start)