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

### Perivascular Region creation from vessels identified using centroids and known dimensional parameters

"""Two Photon Perivascular Regions
Designed to produce a mask of the perivascular regions around vessels
Analyze the centroid region of the vessels, create a disk 10 um wide around the vessel, then subtract the vessel masks from these disks
Function call - 2P_ID.py <VE_filename> <Bolus_filename>
I still need to reduce the number of imports and see if I can expedite the process
Save out vessels with 0 being arterioles and 1 being venules"""

def trans(img):
    tSave = np.transpose(img,(2,1,0))
    tSave = np.rot90(tSave,3,axes=(1,2))
    tSave = np.flip(tSave,2)
    return tSave

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

veFN = sys.argv[1]

# Threshold with rhodamine to identify vessels
tif = TiffFile(veFN)
ve = tif.asarray() # Only has extracted vessels

[imSlices, imHeight, imWidth] = ve.shape

# Parameter Extraction from filename - would like to do it from TIFF tags, but running into issues
# Example filename - 2022-03-29_Baseline_Stack_1_lam_880nm_eom_100_power_6_75_pmt_56_size_400x400mic_pixels_510x510_freq_800_LinAvg_1_range_0mic-neg200mic_slice_1micPMT - PMT [HS_1] _C6.ome
# Might actually need it here
width = int(veFN[veFN.find('size_')+5:veFN.find('size_')+8])
height = int(veFN[veFN.find('mic')-3:veFN.find('mic')])
depth = int(veFN[veFN.find('slice_')+6:veFN.find('micPMT')])

prMasks = np.zeros((imHeight, imWidth))
for image in ve:
    labelImg = label(image)
    regions = regionprops(labelImg)
    pRad = int(10*height/imHeight) # Number of pixels in a 10 um radius, maybe it should be diameter?
    prIndices = []
    prMask = np.zeros((imHeight, imWidth))
    for num, x in enumerate(regions):
        # Check Centroid vs. Centroid_Weighted
        [xcentroid, ycentroid] = x.centroid
        # Create a circle at that point of pRad radius
        for col in range(imWidth):
            for row in range(imHeight):
                # Create a circle of given radius around centroid coordinates and label it by vessel number
                prMask[row, col] = (((row-ycentroid)**2+(col-xcentroid)**2) < pRad**2) * (num + 1)
    prMask = prMask - image # removes internal vessel spaces from perivascular regions
    prMasks = np.dstack((prMasks, prMask))

prMasks = prMasks[:,:,1:imSlices]
prMasks = trans(prMasks).astype('float32')
prFN = veFN[0:veFN.find('_VESSEL_Mask.tif')] + '_PR_Mask.tif'