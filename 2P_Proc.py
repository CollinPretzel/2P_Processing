import sys
import numpy as np
from tifffile import imwrite
from tifffile import TiffFile
from matplotlib import pyplot as plt
from skimage import color, data, restoration, exposure, io
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

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
    img = img_as_ubyte(data.page())

    radius = 15
    selem = disk(radius)

    local_otsu = rank.otsu(img, selem)
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu
    return img >= local_otsu

# Structure of function call: python 2P_Proc.py <filtered filename> <> <>

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

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
dStack = scan[0]
rStack = scan[1]

[imSlices, imHeight, imWidth] = rStack.shape

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
for image in rStack:
    threshImage = otsuThresh(image, PSF, 2.8) # What is this 2.8 - balance
    threshStack = np.dstack((threshStack,threshImage))

threshStack = threshStack[:,:,1:imSlices+1] # removes initial empty array

# Saving process to have same orientation in ImageJ and display, might be unnecessary?
tSave = np.transpose(procStack,(2,1,0))
tSave = np.rot90(tSave,3,axes=(1,2))
tSave = np.flip(tSave,2)
outfilename = filename[0:filename.find('.ome.tif')] + '_thresh_rhodamine.tif'
imwrite('test.tif', tSave, photometric='minisblack')