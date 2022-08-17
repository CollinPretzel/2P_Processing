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
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

"""Two Photon Registration Between Scans Function
Designed to align all images to reduce the impact of breathing or motion artifacts
Function call - 2P_RB.py <moving_filename> <baseline_filename>
I still need to reduce the number of imports and see if I can expedite the process
OH and check whether it's more effective before or after WF"""

def trans(img):
    tSave = np.transpose(img,(2,1,0))
    tSave = np.rot90(tSave,3,axes=(1,2))
    tSave = np.flip(tSave,2)
    return tSave

### FIX BOTH OF THESE!!! - Theoretically fixed, but I need a few test runs
def reg(ref_img, mov_img): # Registers
    # Normalize both images
    rmax = np.max(ref_img)
    rmin = np.min(ref_img)
    mmax = np.max(mov_img)
    mmin = np.min(mov_img)
    ref_img = (ref_img-rmin)/rmax
    mov_img = (mov_img-mmin)/mmax
    # Use estimated optical flow for registration
    v, u = optical_flow_tvl1(ref_img, mov_img) # Alternative is ILK algorithm
    nr, nc = ref_img.shape
    rc, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    warp_img = warp(mov_img, np.array([rc + v, cc + u]), mode='edge')
    warp_img = (warp_img*mmax)+mmin
    return v, u, warp_img

# reg 2 is a second registration used to register the FITC scans 
# which should have few identifiable landmarks
def reg2(v, u, mov_img):
    # Normalize moving image, transform it, and then re-amplify it
    mmax = np.max(mov_img)
    mmin = np.min(mov_img)
    mov_img = (mov_img-mmin)/mmax
    # Use input optical flow for registration
    nr, nc = mov_img.shape
    rc, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    warp_img = warp(mov_img, np.array([rc + v, cc + u]), mode='edge')
    warp_img = (warp_img*mmax)+mmin
    return warp_img

# Structure of function call: python 2P_Proc.py <filtered filename> <> <>

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

moveFN = sys.argv[1]
movePrefix = moveFN[0:moveFN.find('_PMT -')]
baseFN = sys.argv[2]
basePrefix = baseFN[0:baseFN.find('_PMT -')]
mCSV = movePrefix + '.csv'
bCSV = basePrefix + '.csv'

# Threshold with rhodamine to identify vessels
tif = TiffFile(moveFN)
mScan = tif.asarray() # Imports as 'CZYX', C = 0 is
mFitcStack = mScan[0]
mRhodStack = mScan[1]

tif = TiffFile(baseFN)
bScan = tif.asarray()
bFitcStack = bScan[0]
bRhodStack = bScan[1]

[mImSlices, mImHeight, mImWidth] = mRhodStack.shape
[bImSlices, bImHeight, bImWidth] = bRhodStack.shape

# Parameter Extraction from csv file
# Need width, height, starting, ending positions of both scans to check
# Relative placement prior to registration
mParams = []
with open(mCSV, newline = '') as f:
    fReader = csv.reader(f, delimiter = ',')
    for row in fReader:
        mParams.append(row)

bParams = []
with open(bCSV, newline = '') as f:
    fReader = csv.reader(f, delimiter = ',')
    for row in fReader:
        bParams.append(row)

mStart = mParams[1][3]
mEnd = mParams[1][4]
bStart = bParams[1][3]
bEnd = bParams[1][4]

## Check whether the scans are physically aligned to the same space
## If not, pad scans with empty arrays for total alignement
# Starting position
if bStart != mStart:
    if bStart > mStart: # Moving scan starts above the baseline
        topBSlices = (bStart - mStart)/((bStart-bEnd)/bImSlices)
        newBFitc = np.zeros((bImSlices+topBSlices, bImHeight, bImWidth))
        newBRhod = np.zeros((bImSlices+topBSlices, bImHeight, bImWidth))
        [bImSlices, bImHeight, bImWidth] = newBRhod.shape
        newBFitc[topBSlices:bImSlices,:,:] = bFitcStack
        newBRhod[topBSlices:bImSlices,:,:] = bRhodStack
        bFitcStack = newBFitc
        bRhodStack = newBRhod
    else:
        topMSlices = (mStart - bStart)/((mStart-mEnd)/mImSlices)
        newMFitc = np.zeros((mImSlices+topMSlices, mImHeight, mImWidth))
        newMRhod = np.zeros((mImSlices+topMSlices, mImHeight, mImWidth))
        [mImSlices, mImHeight, mImWidth] = newMRhod.shape
        newMFitc[topMSlices:mImSlices,:,:] = mFitcStack
        newMRhod[topMSlices:mImSlices,:,:] = mRhodStack
        mFitcStack = newMFitc
        mRhodStack = newMRhod
# Ending positions
if bEnd != mEnd:
    if bEnd > mEnd: # Moving scan ends above the baseline
        botBSlices = (bEnd - mEnd)/((bStart-bEnd)/bImSlices)
        newBFitc = np.zeros((bImSlices+botBSlices, bImHeight, bImWidth))
        newBRhod = np.zeros((bImSlices+botBSlices, bImHeight, bImWidth))
        newBFitc[0:bImSlices,:,:] = bFitcStack
        newBRhod[0:bImSlices,:,:] = bRhodStack
        [bImSlices, bImHeight, bImWidth] = newBRhod.shape
        bFitcStack = newBFitc
        bRhodStack = newBRhod
    else:
        botMSlices = (mEnd - mStart)/((mStart-mEnd)/mImSlices)
        newMFitc = np.zeros((mImSlices+botMSlices, mImHeight, mImWidth))
        newMRhod = np.zeros((mImSlices+botMSlices, mImHeight, mImWidth))
        newMFitc[0:mImSlices,:,:] = mFitcStack
        newMRhod[0:mImSlices,:,:] = mRhodStack
        [mImSlices, mImHeight, mImWidth] = newMRhod.shape
        mFitcStack = newMFitc
        mRhodStack = newMRhod

# Try registration between baseline (bScan) and moving (mScan)
start = time.perf_counter()
mRhodRegStack = np.empty((mImHeight, mImWidth))
mFitcRegStack = np.empty((mImHeight, mImWidth))

for sid in range(0, mImSlices):
    v, u, rhodImg = reg(bRhodStack[sid], mRhodStack[sid])
    fitcImg = reg2(v, u, mFitcStack[sid])
    rhodRegStack = np.dstack((mRhodRegStack, rhodImg))
    fitcRegStack = np.dstack((mFitcRegStack, fitcImg))

mRhodRegStack = mRhodRegStack[:,:,1:mImSlices+1]
mFitcRegStack = mFitcRegStack[:,:,1:mImSlices+1]
print(time.perf_counter()-start)

# Save registered stack
mRhodRegSave = trans(mRhodRegStack).astype('float32')
mFitcRegSave = trans(mFitcRegStack).astype('float32')
fullSave = np.stack((mFitcRegSave,mRhodRegSave), axis = -1)
fullSave = np.transpose(fullSave, (3, 0, 1, 2))
outFN = movePrefix + '_' + basePrefix + '.tif'