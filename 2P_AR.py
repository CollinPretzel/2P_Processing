import sys, csv
import math
import time
import warnings
import numpy as np
from tifffile import imwrite
from tifffile import TiffFile
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

"""2P_AR.py <Scan_filename>
   Artifact removal for all Two-Photon images, which focuses on the Rhodamine scans as
   they demonstrate more significant features and then average the same FITC sections.
   The windowing correction method is adapted from: """

def trans(img):
    tSave = np.transpose(img,(2,1,0))
    tSave = np.rot90(tSave,3,axes=(1,2))
    tSave = np.flip(tSave,2)
    return tSave

def CCcalc(sect1, sect2):
    # Apply a gaussian filter, trouble finding a mean filter, could make my own
    [width, height] = sect1.shape
    filt1 = ndi.gaussian_filter(sect1, 1)
    filt2 = ndi.gaussian_filter(sect2, 1)
    avg1 = np.mean(filt1)
    avg2 = np.mean(filt2)
    numerator, denom1, denom2 = [0,0,0]
    for x in range(width+1):
        for y in range(height+1):
            numerator += ((filt1[x,y]-avg1)*(filt2[x,y]-avg2))
            denom1 += (filt1[x,y]-avg1)**2
            denom2 += (filt2[x,y]-avg2)**2
    CC = numerator/(np.sqrt(denom1)*np.sqrt(denom2))
    return CC

# General structure involves iterating over x number of slices
numReps = 4
