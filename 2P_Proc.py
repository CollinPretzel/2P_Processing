import sys
import cv2
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
from skimage.measure import label, regionprops, regionprops_table

### Rewrite for auto local thresholding with Otsu (tested in ImageJ)
### Worked the best for cross-sectional analysis
### - Needs a method to assess circularity of various labeled sections
### - Should also have a method to assess the # of pixels and exclude capillaries
###    and lateral vessels of course

# Define functions for visualization
def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

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

# Variables that will be used throughout the processing pipeline
global stack, threshStack
global imHeight, imWidth, imSlices, xRes, yRes, zRes
# Extract these variables from filename, idk how to access tags
global height, width, depth
#global exw, emw, freq

filename = sys.argv[1]

# Read in file, seems to invert it? Or the imwrite inverts it
tif = TiffFile(filename)
scan = tif.asarray() # Imports as 'CZYX', C = 0 is
#dStack = scan[0]
#rStack = scan[1]

[imSlices, imHeight, imWidth] = scan.shape

# Transform to be XYZ, might need to flip and rotate again?
#dStack = np.transpose(dStack, (2,1,0))
#rStack = np.transpose(rStack, (2,1,0))

# Parameter Extraction from filename - would like to do it from TIFF tags, but running into issues
# Example filename - 2022-03-29_Baseline_Stack_1_lam_880nm_eom_100_power_6_75_pmt_56_size_400x400mic_pixels_510x510_freq_800_LinAvg_1_range_0mic-neg200mic_slice_1micPMT - PMT [HS_1] _C6.ome
width = int(filename[filename.find('size_')+5:filename.find('size_')+8])
height = int(filename[filename.find('mic')-3:filename.find('mic')])
depth = int(filename[filename.find('slice_')+6:filename.find('micPMT')])

# Apply Otsu Thresholding
threshStack = np.empty((imHeight, imWidth))
for image in scan:
    threshImage = otsuThresh(image, 15)
    threshStack = np.dstack((threshStack, threshImage))

threshStack = threshStack[:,:,1:imSlices+1] # removes initial empty array
threshStack = trans(threshStack)

# Remove any excessive labels
start = time.perf_counter()
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
        circ = (4*math.pi*area)/(perimeter**2)
        if (area > 100) and (area < 500):# and (ecc < 0.78) and (circ > 0.25):
            aIndices.append(num)
            areas.append(area)
            if (ecc < 0.8):
                eccIndices.append(num)
                eccs.append(ecc)
                if (circ > 0.15) and (circ < 1):
                    circIndices.append(num)
                    circs.append(circ)
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
print(time.perf_counter()-start)

## NOW TACKLE THESE MASKS SIDEWAYS TO GET SOME MORE CONNECTED COMPONENTS - start w/ emask
## Takes 68.5 seconds for a 510x201x510 scan

vMasks = np.empty((imHeight, imSlices))
for scan in eMasks:
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

# Another connected component analysis, 3D, to isolate and remove the smaller regions to try to reduce error
lv = label(vMasks)
regions = regionprops(lv)
fullMask = np.empty((imHeight,imWidth,imSlices))
vidx = []
areas = []
for num, x in enumerate(regions):
    area = x.area
    if (area > 8000):# and (ecc < 0.78) and (circ > 0.25):
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
tSave = threshStack.astype('float32')
fullSave = fullMask.astype('float32')
threshOFN = filename[0:filename.find('.ome.tif')] + '_OTSU_Mask.tif'
areaOFN = filename[0:filename.find('.ome.tif')] + '_AREA_Mask.tif'
eccOFN = filename[0:filename.find('.ome.tif')] + '_ECC_Mask.tif'
circOFN = filename[0:filename.find('.ome.tif')] + '_CIRC_Mask.tif'
vesselOFN = filename[0:filename.find('.ome.tif')] + '_FULL_Mask.tif'
imwrite(threshOFN, tSave, photometric='minisblack')
imwrite(areaOFN, aSave, photometric='minisblack')
imwrite(eccOFN, eSave, photometric='minisblack')
imwrite(circOFN, cSave, photometric='minisblack')
imwrite(vesselOFN, fullSave, photometric='minisblack')