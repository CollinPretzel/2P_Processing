import psf
import sys
import numpy as np
from tifffile import imwrite
from tifffile import TiffFile
from matplotlib import pyplot as plt
from skimage import color, data, restoration, exposure, io

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

# Structure of function call: python 2P_Proc.py <filename> <> <>

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

# Variables that will be used throughout the processing pipeline
global stack, procStack
global imHeight, imWidth, imSlices, xRes, yRes, zRes
# Extract these variables from filename, idk how to access tags
global height, width, depth
global exw, emw, freq

filename = sys.argv[1]

# Read in file, seems to invert it? Or the imwrite inverts it
tif = TiffFile(filename)
for page in tif.pages:
    image = page.asarray()
    stack = np.dstack((stack,image))

[imHeight, imWidth, imSlices] = stack.shape

# Parameter Extraction from filename - would like to do it from TIFF tags, but running into issues
# Example filename - 2022-03-29_Baseline_Stack_1_lam_880nm_eom_100_power_6_75_pmt_56_size_400x400mic_pixels_510x510_freq_800_LinAvg_1_range_0mic-neg200mic_slice_1micPMT - PMT [HS_1] _C6.ome
exw = int(filename[filename.find('nm')-3:filename.find('nm')])
width = int(filename[filename.find('size_')+5:filename.find('size_')+8])
height = int(filename[filename.find('mic')-3:filename.find('mic')])
depth = int(filename[filename.find('slice_')+6:filename.find('micPMT')])

## Create idealized PSF for Weiner Filter
emw = 520
args = {
    'shape': (imWidth, imHeight), # number of samples in z and r direction
    'dims': (200, 200), # size in z and r direction in micrometers - why not 200, and what is r?
    'ex_wavelen': exw,
    'em_wavelen': emw, # Conventionally 520... I think
    'num_aperture': 0.95,
    'refr_index': 1.35, # refraction index of ultrasound gel
    'magnification': 20,
    'pinhole_radius': None,
    'pinhole_shape': 'round',
}

obsvol = psf.PSF(psf.ISOTROPIC | psf.TWOPHOTON, **args)
psf1 = obsvol.slice(0)
psf2 = np.flip(psf1, axis=1)
psf3 = np.concatenate((psf2,psf1),axis=1)
psf4 = np.flip(psf3, axis=0)
PSF = np.concatenate((psf4,psf3),axis=0)

PSF = PSF[255:765,255:765]

# Apply Weiner Filter
for scan in stack:
    procImage = restoration.weiner(scan, PSF, 2.8) # What is this 2.8 - balance
    procStack = np.dstack((procStack,procImage))

# Saving process to have same orientation in ImageJ and display
tSave = np.transpose(procStack,(2,1,0))
tSave = np.rot90(tSave,3,axes=(1,2))
tSave = np.flip(tSave,2)
outfilename = filename[0:filename.find('.tif')] + '_WF.tif'
imwrite(outfilename, tSave, photometric='minisblack')