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

"""Two Photon Internal Registration Function
Designed to align all images to reduce the impact of breathing or motion artifacts
Function call - 2P_IR.py <filename>
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

filename = sys.argv[1]

# Threshold with rhodamine to identify vessels
tif = TiffFile(filename)
scan = tif.asarray() # Imports as 'CZYX', C = 0 is
fitcStack = scan[0]
rhodStack = scan[1]

[imSlices, imHeight, imWidth] = rhodStack.shape

# Parameter Extraction from filename - would like to do it from TIFF tags, but running into issues
# Example filename - 2022-03-29_Baseline_Stack_1_lam_880nm_eom_100_power_6_75_pmt_56_size_400x400mic_pixels_510x510_freq_800_LinAvg_1_range_0mic-neg200mic_slice_1micPMT - PMT [HS_1] _C6.ome
width = int(filename[filename.find('size_')+5:filename.find('size_')+8])
height = int(filename[filename.find('mic')-3:filename.find('mic')])
depth = int(filename[filename.find('slice_')+6:filename.find('micPMT')])

# Try registration without otsu thresholding
start = time.perf_counter()
rhodRegStack = np.empty((imHeight, imWidth))
rhodRegStack = np.dstack((rhodRegStack, rhodStack[0]))
fitcRegStack = np.empty((imHeight, imWidth))
fitcRegStack = np.dstack((fitcRegStack, fitcStack[0]))
for sid in range(1, imSlices):
    v, u, rhodImg = reg(rhodRegStack[...,sid], rhodStack[sid])
    fitcImg = reg2(v, u, fitcStack[sid])
    rhodRegStack = np.dstack((rhodRegStack, rhodImg))
    fitcRegStack = np.dstack((fitcRegStack, fitcImg))

rhodRegStack = rhodRegStack[:,:,1:imSlices+1]
fitcRegStack = fitcRegStack[:,:,1:imSlices+1]
print(time.perf_counter()-start)