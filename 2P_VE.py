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

def trans(img):
    tSave = np.transpose(img,(2,1,0))
    tSave = np.rot90(tSave,3,axes=(1,2))
    tSave = np.flip(tSave,2)
    return tSave

# Structure of function call: python 2P_VE.py <Otsu filename>
"""2P_VE.py <Otsu_thresh filename> - Extracts the vessels from the thresholded scan
   using connected components analysis"""

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

warnings.filterwarnings('ignore', '.*rank.*') # Ignores warnings in Otsu thresh about bit depth

filename = sys.argv[1]

# Read in file
tif = TiffFile(filename)
threshStack = tif.asarray()

[imSlices, imHeight, imWidth] = threshStack.shape

# Parameter importation also likely unnecessary for vessel extraction

# Remove any excessive labels -= How to make more efficient
# And how to use the Area masks for branching
aMasks = np.empty((imHeight, imWidth))
eMasks = np.empty((imHeight, imWidth))
cMasks = np.empty((imHeight, imWidth))
for image in threshStack:
    labelImg = label(image)
    regions = regionprops(labelImg)
    aIndices = []
    areas = []
    eccIndices = []
    eccs = []
    circIndices = []
    circs = []
    aMask = np.zeros((imHeight, imWidth))
    eMask = np.zeros((imHeight, imWidth))
    cMask = np.zeros((imHeight, imWidth))
    for num, x in enumerate(regions):
        area = x.area_filled # What if you did convex area, or feret_diameter_max
        perimeter = x.perimeter
        ecc = x.eccentricity
        if(perimeter > 0):
            circ = (4*math.pi*area)/(perimeter**2)
        else:
            circ = 10000
        
        if (area > 100) and (area < 500):# and (ecc < 0.78) and (circ > 0.25):
            aIndices.append(num)
            
            if (ecc < 0.8):
                eccIndices.append(num)
                areas.append(area)
                #eccs.append(ecc)
                if (circ > 0.15) and (circ < 1):
                    circIndices.append(num)
                    #circs.append(circ)
    for index in aIndices:
        aMask += (labelImg==index+1).astype(int)
    for index in eccIndices:
        eMask += (labelImg==index+1).astype(int)
    for index in circIndices:
        cMask += (labelImg==index+1).astype(int)
    # Fill in mask w/ skimage.reconstruction - erosion
    seed = np.copy(aMask)
    seed[1:-1,1:-1] = 1
    aMask = reconstruction(seed, aMask, method = 'erosion')
    seed = np.copy(eMask)
    seed[1:-1,1:-1] = 1
    eMask = reconstruction(seed, eMask, method = 'erosion')
    seed = np.copy(aMask)
    seed[1:-1,1:-1] = 1
    cMask = reconstruction(seed, cMask, method = 'erosion')
    aMasks = np.dstack((aMasks, aMask))
    eMasks = np.dstack((eMasks, eMask))
    cMasks = np.dstack((cMasks, cMask))

aMasks = aMasks[:,:,1:imSlices+1]
eMasks = eMasks[:,:,1:imSlices+1]
cMasks = cMasks[:,:,1:imSlices+1]

## NOW TACKLE THESE MASKS SIDEWAYS TO GET SOME MORE CONNECTED COMPONENTS - start w/ emask
## Takes 68.5 seconds for a 510x201x510 scan

vMasks = np.empty((imHeight, imSlices))
for scan in cMasks:
    scan = np.array(scan)
    exScan = np.zeros_like(scan)
    for rid, row in enumerate(scan):
        pidx = np.where(row==1)[0]
        for pid in pidx:
            sect = np.array(scan[rid,pid:pid+15])
            idx = np.where(sect == 1)[0]
            ext = idx[idx.size-1]
            exScan[rid,pid:pid+ext] = 1
    vMasks = np.dstack((vMasks,exScan))

vMasks = vMasks[:,:,1:imWidth+1]
vMasks = np.transpose(vMasks,(0,2,1))

# Connected component issue, might be dependent on depth of scan
## Another connected component analysis, 3D, to isolate and remove the smaller regions to try to reduce error
lv = label(vMasks)
regions = regionprops(lv)
fullMask = np.empty((imHeight,imWidth,imSlices))
vidx = []
areas = []
minVol = (imHeight*imWidth*imSlices)/6535 # Experimentally concluded, I'm not sure how else to determine the volume percentage
for num, x in enumerate(regions):
    area = x.area
    if (area > minVol): #Calculate approximate volume based on image dimensions
        vidx.append(num)
        areas.append(area)

for index in vidx:
    fullMask += (lv==index+1).astype(int)

# Special transformation for saving the full mask
fullMask = np.transpose(fullMask, (2,0,1))
fullMask = np.rot90(fullMask, 1, axes=(1,2))
fullMask = np.flip(fullMask,1)

# Saving process to have same orientation in ImageJ and display, might be unnecessary?
aMasks = trans(aMasks)
eMasks = trans(eMasks)
cMasks = trans(cMasks)
aSave = aMasks.astype('float32')
eSave = eMasks.astype('float32')
cSave = cMasks.astype('float32')
fullSave = fullMask.astype('float32')
areaOFN = filename[0:filename.find('_OT.tif')] + '_AREA_Mask.tif'
eccOFN = filename[0:filename.find('_OT.tif')] + '_ECC_Mask.tif'
circOFN = filename[0:filename.find('_OT.tif')] + '_CIRC_Mask.tif'
vesselOFN = filename[0:filename.find('_OT.tif')] + '_VESSEL_Mask.tif'
imwrite(areaOFN, aSave, photometric='minisblack')
imwrite(eccOFN, eSave, photometric='minisblack')
imwrite(circOFN, cSave, photometric='minisblack')
imwrite(vesselOFN, fullSave, photometric='minisblack')