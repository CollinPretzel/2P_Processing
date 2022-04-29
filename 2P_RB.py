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
baseFN = sys.argv[2]

# Threshold with rhodamine to identify vessels
tif = TiffFile(moveFN)
mScan = tif.asarray() # Imports as 'CZYX', C = 0 is
mFitcStack = mScan[0]
mRhodStack = mScan[1]

tif = TiffFile(baseFN)
bScan = tif.asarray()
bFitcStack = bScan[0]
bRhodStack = bScan[1]

[imSlices, imHeight, imWidth] = mRhodStack.shape

# Parameter Extraction from filename - would like to do it from TIFF tags, but running into issues
# Example filename - 2022-03-29_Baseline_Stack_1_lam_880nm_eom_100_power_6_75_pmt_56_size_400x400mic_pixels_510x510_freq_800_LinAvg_1_range_0mic-neg200mic_slice_1micPMT - PMT [HS_1] _C6.ome
width = int(moveFN[moveFN.find('size_')+5:moveFN.find('size_')+8])
height = int(moveFN[moveFN.find('mic')-3:moveFN.find('mic')])
depth = int(moveFN[moveFN.find('slice_')+6:moveFN.find('micPMT')])

# Try registration between baseline (bScan) and moving (mScan)
start = time.perf_counter()
mRhodRegStack = np.empty((imHeight, imWidth))
mFitcRegStack = np.empty((imHeight, imWidth))

for sid in range(0, imSlices):
    v, u, rhodImg = reg(bRhodStack[sid], mRhodStack[sid])
    fitcImg = reg2(v, u, mFitcStack[sid])
    rhodRegStack = np.dstack((mRhodRegStack, rhodImg))
    fitcRegStack = np.dstack((mFitcRegStack, fitcImg))

mRhodRegStack = mRhodRegStack[:,:,1:imSlices+1]
mFitcRegStack = mFitcRegStack[:,:,1:imSlices+1]
print(time.perf_counter()-start)

# Save registered stack