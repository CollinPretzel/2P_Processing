import sys, csv
import math
import time
import numpy as np
from tifffile import imwrite
from tifffile import TiffFile
from matplotlib import pyplot as plt

from skimage.measure import label, regionprops

### Perivascular Region creation from vessels identified using centroids and known dimensional parameters

"""Two Photon Perivascular Regions
Function call - 2P_ID.py <Arteriole_filename> <Venule_filename>
Designed to produce a mask of the perivascular regions around vessels
Analyze the centroid region of the vessels, create a disk 10 um wide around the vessel, then subtract the vessel masks from these disks
I still need to reduce the number of imports and see if I can expedite the process"""

def trans(img):
    tSave = np.transpose(img,(2,1,0))
    tSave = np.rot90(tSave,3,axes=(1,2))
    tSave = np.flip(tSave,2)
    return tSave

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

artFN = sys.argv[1]
venFN = sys.argv[2]
prefix = artFN[0:artFN.find('_art')]

# Threshold with rhodamine to identify vessels
tif = TiffFile(artFN)
arterioles = tif.asarray() # Only has extracted vessels
tif = TiffFile(venFN)
venules = tif.asarray()

[imSlices, imHeight, imWidth] = arterioles.shape

# Parameter extraction needed for resolution calculations
vesselCSV = prefix + '.csv'
vesselParams = []
with open(vesselCSV, newline = '') as f:
    fReader = csv.reader(f, delimiter = ',')
    for row in fReader:
        vesselParams.append(row)

height = [1][1]
width = [1][0]

# Assuming arterioles are approx. 12 um across, have an equivalent perivascular space
# Venules are around 8 um across and have a similar perivascular space
artPR = np.zeros_like(arterioles)
venPR = artPR

artPRad = 10
venPRad = 7.5

for sliceNum in imSlices:
    artRadPix = int(artPRad*height/imHeight) # Number of pixels in a 10 um radius, maybe it should be diameter?
    venRadPix = int(venPRad*height/imHeight)
    artRegions = label(arterioles[sliceNum])
    artRegions = regionprops(artRegions)
    venRegions = label(venules[sliceNum])
    venRegions = regionprops(venRegions)
    prMask = np.zeros((imHeight, imWidth))
    for num, x in enumerate(artRegions):
        # Check Centroid vs. Centroid_Weighted
        [xcentroid, ycentroid] = x.centroid
        # Create a circle at that point of pRad radius
        for col in range(imWidth):
            for row in range(imHeight):
                # Create a circle of given radius around centroid coordinates and label it by vessel number
                prMask[row, col] = (((row-ycentroid)**2+(col-xcentroid)**2) < artPRad**2)
    artPR[sliceNum] = prMask - arterioles[sliceNum] # removes internal vessel spaces from perivascular regions
    prMask = np.zeros((imHeight, imWidth))
    for num, x in enumerate(venRegions):
        # Check Centroid vs. Centroid_Weighted
        [xcentroid, ycentroid] = x.centroid
        # Create a circle at that point of pRad radius
        for col in range(imWidth):
            for row in range(imHeight):
                # Create a circle of given radius around centroid coordinates and label it by vessel number
                prMask[row, col] = (((row-ycentroid)**2+(col-xcentroid)**2) < venPRad**2)
    venPR[sliceNum] = prMask - venules[sliceNum]

artPR = trans(artPR).astype('float32')
artPRFN = prefix + '_artPR.tif'
venPR = trans(venPR).astype('float32')
venPRFN = prefix + '_venPR.tif'
imwrite(artPRFN, artPR, photometric = 'minisblack')
imwrite(venPRFN, venPR, photometric = 'minisblack')