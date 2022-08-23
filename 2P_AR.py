from socket import NI_NUMERICHOST
import sys, csv
import math
import time
import warnings
import numpy as np
from tifffile import imwrite
from tifffile import TiffFile
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

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
    for x in range(width):
        for y in range(height):
            numerator += ((filt1[x,y]-avg1)*(filt2[x,y]-avg2))
            denom1 += (filt1[x,y]-avg1)**2
            denom2 += (filt2[x,y]-avg2)**2
    CC = numerator/(np.sqrt(denom1)*np.sqrt(denom2))
    return CC

# General structure involves iterating over x number of slices
filename = sys.argv[1]
prefix = filename[0:filename.find('_WF')]

# Read in file
tif = TiffFile(filename)
scan = tif.asarray() # Imports as 'CZYX', C = 0 is
fitcStack = scan[0]
rhodStack = scan[1]

# Define dimensions for sectioning and correction
[imSlices, imHeight, imWidth] = rhodStack.shape
numReps = 4
secWidth = imWidth/4
secHeight = imHeight/4
numSects = 16
corrRhodStack = np.zeros((imSlices/numReps, imHeight, imWidth))
corrFitcStack = np.zeros((imSlices/numReps, imHeight, imWidth))

# This method assumes that there are multiple (numReps) scans taken for every desired depth
for sliceNum in range(0,imSlices,numReps):
    rhodChunk = rhodStack[sliceNum:sliceNum+numReps]
    fitcChunk = fitcStack[sliceNum:sliceNum+numReps]
    rhodSegments = np.zeros((numSects, numReps, secHeight, secWidth))
    fitcSegments = rhodSegments
    corrRhodSeg = np.zeros((numSects, secHeight, secWidth))
    corrFitcSeg = np.zeros((numSects, secHeight, secWidth))
    corrRhodImg = np.zeros((imSlices/numReps, imHeight, imWidth))
    corrFitcImg = corrRhodImg
    i = 0
    for row in range(0,4):
        for col in range(0,4):
            rhodSegments[i,:,:,:] = rhodChunk[:,row*secHeight:(row+1)*secHeight,col*secWidth:(row+1)*secWidth]
            i += 1
    CCtable = np.zeros((numReps, numReps, numSects))
    for sectNum in range(0,numSects):
        for frame2 in range(0,numReps):
            for frame1 in range(0,numReps):
                cc = CCcalc(rhodSegments[sectNum,frame1,:,:],rhodSegments[sectNum,frame2,:,:])
                CCtable[frame1,frame2,sectNum] = cc
    uniqueCCs = 0
    for k in range(1,numReps):
        uniqueCCs =  uniqueCCs + numReps - k
    CCvals = np.zeros(numSects,uniqueCCs)
    for sectNum in range(0,numSects):
        tempCCs = 0
        prev = 0
        for k in range(1,numReps):
            tempCCs = tempCCs + numReps - k
            CCvals[numSects,prev:tempCCs] = CCtable[k-1,k,sectNum]
            prev = tempCCs + 1
    # Threshold values of CCs to maximize relevancy
    CCvals[np.where(CCvals > 0)] = -2
    # Begin k-means clustering to analyze the most closely related images
    kidx = np.zeros(uniqueCCs/2,numSects)
    kidx[:] = -1
    for i in range(0,numSects):
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(CCvals[i,:].reshape(-1,1))
        kMax = kmeans.labels_[np.where(CCvals[i,:] == np.max(CCvals[i,:]))]
        temp = np.where(kmeans.labels_ == kMax)
        kidx[0:len(temp),i] = temp
    for i in range(0,numSects):
        frameChoices = np.where(kidx[:,i] != -1)
        # Trying to determine an algorithm to automatically adapt the pairing of frames
        # Right now, it's more exclusionary and only accepts numReps = 6
        for j in range(0,len(frameChoices)):
            pair = frameChoices[j]
            if pair < numReps-1:
                frame1 = 0
                frame2 = frame1 + 1 + pair
            elif pair < 2*numReps - 3:
                frame1 = 1
                frame2 = 3 + pair - numReps
            elif pair < 3*numReps - 6:
                frame1 = 2
                frame2 = 6 + pair - 2*numReps
            elif pair < 2*numReps + 2: # This is garnered directly toward 6 reps
                frame1 = 3
                frame2 = pair - 8
            else: # Final pairing
                frame1 = numReps-2
                frame2 = numReps-1
            corrRhodSeg[i] += rhodSegments[i,frame1,:,:] + rhodSegments[i,frame2,:,:]
            corrFitcSeg[i] += fitcSegments[i,frame1,:,:] + fitcSegments[i,frame2,:,:]
        # Take the average of all added segments
        corrRhodSeg[i] = corrRhodSeg[i]/(2*(j+1))
        corrFitcSeg[i] = corrFitcSeg[i]/(2*(j+1))
    # Now to replace all corrected segments into their relevant frame
    i = 0
    for row in range(0,4):
        for col in range(0,4):
            corrRhodImg[:,row*secHeight:(row+1)*secHeight,col*secWidth:(col+1)*secWidth] = corrRhodSeg[i]
            corrFitcImg[:,row*secHeight:(row+1)*secHeight,col*secWidth:(col+1)*secWidth] = corrFitcSeg[i]
            i += 1
    corrRhodStack = np.dstack(corrRhodStack, corrRhodImg)
    corrFitcStack = np.dstack(corrFitcStack, corrFitcImg)

rhodSave = trans(corrRhodStack).astype('float32')
fitcSave = trans(corrFitcStack).astype('float32')
fullSave = np.stack((fitcSave, rhodSave), axis = -1)
fullSave = np.transpose(fullSave, (3, 0, 1, 2))
outfilename = prefix + '_AR.tif'
imwrite(outfilename, fullSave, imagej=True, photometric='minisblack', metadata = {'axes': 'ZCYX'})