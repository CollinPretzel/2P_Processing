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

### Rewrite for thresholding with otsu and then internal registration
### Not really sure how to tackle this yet

def otsuThresh(img, radius):
    img = img.astype('uint16')
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    # threshold_global_otsu = threshold_otsu(img)
    # global_otsu = img >= threshold_global_otsu
    return img >= local_otsu # Was >=

def trans(img):
    tSave = np.transpose(img,(2,1,0))
    tSave = np.rot90(tSave,3,axes=(1,2))
    tSave = np.flip(tSave,2)
    return tSave

# Structure of function call: python 2P_Proc.py <filtered filename> <> <>

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

warnings.filterwarnings('ignore', '.*rank.*') # Ignores warnings in Otsu thresh about bit depth

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

# Apply Otsu Thresholding
threshStack = np.empty((imHeight, imWidth))
for image in rhodRegStack:
    threshImage = otsuThresh(image, 15)
    threshStack = np.dstack((threshStack, threshImage))

threshStack = threshStack[:,:,1:imSlices+1] # removes initial empty array
threshStack = trans(threshStack)

# Save otsu mask, doesn't need fitc counterpart
