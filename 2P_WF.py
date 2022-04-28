import psf
import sys
import numpy as np
from tifffile import imwrite
from tifffile import TiffFile
from matplotlib import pyplot as plt
from skimage import color, data, restoration, exposure, io

"""Currently only applied to Rhodamine scans, need emission wavelength 
   for FITC/DiO, easily applied to those though.
   Function Call: 2P_WF.py <filename>"""

# Structure of function call: python 2P_Proc.py <filename> <> <>

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 12})

filename = sys.argv[1]

# Read in file, seems to invert it? Or the imwrite inverts it
tif = TiffFile(filename)
scan = tif.asarray() # Imports as 'CZYX', C = 0 is
dStack = scan[0]
rhodStack = scan[1]

[imSlices, imHeight, imWidth] = rhodStack.shape

# Transform to be XYZ, might need to flip and rotate again?
#dStack = np.transpose(dStack, (2,1,0))
#rStack = np.transpose(rStack, (2,1,0))

# Parameter Extraction from filename - would like to do it from TIFF tags, but running into issues
# Example filename - 2022-03-29_Baseline_Stack_1_lam_880nm_eom_100_power_6_75_pmt_56_size_400x400mic_pixels_510x510_freq_800_LinAvg_1_range_0mic-neg200mic_slice_1micPMT - PMT [HS_1] _C6.ome
exw = int(filename[filename.find('nm')-3:filename.find('nm')])
width = int(filename[filename.find('size_')+5:filename.find('size_')+8])
height = int(filename[filename.find('mic')-3:filename.find('mic')])
depth = int(filename[filename.find('slice_')+6:filename.find('micPMT')])

## Create idealized PSF for Weiner Filter, rhodamine
emw = 520
args_rhod = {
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

obsvol = psf.PSF(psf.ISOTROPIC | psf.TWOPHOTON, **args_rhod)
psf1 = obsvol.slice(0)
psf2 = np.flip(psf1, axis=1)
psf3 = np.concatenate((psf2,psf1),axis=1)
psf4 = np.flip(psf3, axis=0)
PSF = np.concatenate((psf4,psf3),axis=0)

PSF = PSF[255:765,255:765]

# Apply Weiner Filter
procStack = np.empty((imHeight, imWidth))
for image in rhodStack:
    procImage = restoration.wiener(image, PSF, 2.8,clip=False) # What is this 2.8 - balance
    procStack = np.dstack((procStack,procImage))

procStack = procStack[:,:,1:imSlices+1] # removes initial empty array

# Saving process - Make sure to save both channels
tSave = np.transpose(procStack,(2,1,0))
tSave = np.rot90(tSave,3,axes=(1,2))
tSave = np.flip(tSave,2)
tSave = tSave.astype('float32')
outfilename = filename[0:filename.find('.ome.tif')] + '_WF_rhodamine.tif'
imwrite(outfilename, tSave, imagej=True, photometric='minisblack')