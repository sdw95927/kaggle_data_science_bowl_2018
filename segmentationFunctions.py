#Functions for nuclei segmentation in Kaggle 2018

import numpy                    as np
import matplotlib.image         as mpimg
import matplotlib.pyplot        as plt
from sklearn                    import preprocessing
import scipy.misc
import cv2
import skimage
from skimage                    import measure
from skimage                    import img_as_bool, io, color, morphology, segmentation
from skimage.morphology         import binary_closing, binary_opening, disk

import time
import re
import sys
import os
import openslide
from openslide                  import open_slide, ImageSlide
import matplotlib.pyplot        as plt

import pandas                   as pd
import xml.etree.ElementTree    as ET
from skimage.draw import polygon
import random

#####################################################################
#Functions for color deconvolution
#####################################################################
def checkChannel(channel):
    """
    Input: a channel in C before normalization
    Output: whether there is any True signals in that channel (yes: 1; no: 0)
    """
    channel = removeInfinities(channel)
    if np.var(channel.reshape(-1)) < 0.02:
        return (0)
    else:
        return (1)


def countInfinites(mat):
    """
    counts the number of infinities within a 1D matrix or list
    """
    isFinite = np.all(np.isfinite(mat))
    
    if not isFinite:
        count = 0
        indices = []
        for i in range(0,len(mat)):
            if mat[i] in [-np.inf,np.inf]:
                count+=1
                indices.append(i)

def removeInfinities(mat):
    """
    removes infinities from a matrix
    returns a matrix with the infinities replaced with
    the average of the matrix values
    """
    isFinite = np.all(np.isfinite(mat))
    
    if not isFinite:
        nrow, ncol = mat.shape
        matCopy = mat.copy()
        matReshaped = matCopy.reshape(-1)
        minVal = np.nanmin(matReshaped[matReshaped != -np.inf])
        maxVal = max(matReshaped)
        
        #count infinities and get their indicies
        #countInfinites(matReshaped) 
        
        #replace all infinities with nan
        for i in range(0,len(matReshaped)):
            if matReshaped[i] in [-np.inf, np.inf]:
                matReshaped[i] = minVal
        '''
        #get average of matrix values and replace the nan with averages
        def averageVal(array):
            total = 0
            count = 0
            for elmt in array:
                if elmt != -90001:
                    total+=elmt
                    count+=1
            return total/count

        average = averageVal(matReshaped)

        #replace nan with average
        for i in range(0,len(matReshaped)):
            if matReshaped[i] == -90001:
                matReshaped[i] = average
        '''
        #catch any potential errors        
        if np.any(np.isinf(matReshaped)):
            raise ValueError

        return matReshaped.reshape((nrow, ncol))
    else:
        return mat
            
def normalize(mat):
    matCopy = mat.copy()
    nrow, ncol = mat.shape
    
    #reshape matrix
    matReshaped = matCopy.reshape(-1)
    
    maxVal = max(matReshaped)
    minVal = min(matReshaped)
    
    #normalize by dividing by max value
    for i in range(0,len(matReshaped)):
        matReshaped[i] = (matReshaped[i]-minVal)/(maxVal-minVal)
    
    maxVal = max(matReshaped)
    minVal = min(matReshaped)
    
    return matReshaped.reshape((nrow, ncol))

def convert_to_optical_densities(rgb, r0, g0, b0):
    OD = rgb.astype(float)
    OD[:,:,0] /= r0
    OD[:,:,1] /= g0
    OD[:,:,2] /= b0
    
    return -np.log(OD+0.00001)

def colorDeconvolution(rgb, r0, g0, b0, verbose=False, stainingType="HDB"):
    if stainingType == "HDB":
        #stain_OD = np.asarray([[0.18,0.20,0.08],[0.10,0.21,0.29],[0.754,0.077,0.652]]) #hematoxylin, DAB, background
        stain_OD = np.asarray([[0.650,0.704,0.286],[0.268,0.570,0.776],[0.754,0.077,0.652]]) #hematoxylin, DAB, background
    elif stainingType == "HRB":
        stain_OD = np.asarray([[0.650,0.704,0.286],[0.214,0.851,0.478],[0.754,0.077,0.652]]) #hematoxylin, red dye, background
    elif stainingType == "HDR":
        stain_OD = np.asarray([[0.650,0.704,0.286],[0.268,0.570,0.776],[0.214,0.851,0.478]]) #hematoxylin, DAB, red dye
    elif stainingType == "HEB":
        stain_OD = np.asarray([[0.550,0.758,0.351],[0.398,0.634,0.600],[0.754,0.077,0.652]]) #hematoxylin, Eosin, background
    else:
        print("Staining type not defined. Choose one from the following: HDB, HRB, HDR, HEB")
        return 0

    n = []
    for r in stain_OD:
        n.append(r/np.linalg.norm(r))

    normalized_OD = np.asarray(n)
    D = np.linalg.inv(normalized_OD)
    OD = convert_to_optical_densities(rgb,r0,g0,b0)

    ODmax = np.max(OD,axis=2) 
    #plt.figure()
    #plt.imshow(ODmax>.1)

    # reshape image on row per pixel
    rOD = np.reshape(OD,(-1,3))

    # do the deconvolution
    rC = np.dot(rOD,D)

    #restore image shape
    C = np.reshape(rC,OD.shape)

    #remove problematic pixels from the the mask
    ODmax[np.isnan(C[:,:,0])] = 0
    ODmax[np.isnan(C[:,:,1])] = 0
    ODmax[np.isnan(C[:,:,2])] = 0
    ODmax[np.isinf(C[:,:,0])] = 0
    ODmax[np.isinf(C[:,:,1])] = 0
    ODmax[np.isinf(C[:,:,2])] = 0
    
    return (ODmax,C)

def plotFourImages(img1,img2,img3,img4):
    #plot each channel
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(img1)
    ax[0].set_title('1')
    
    ax[1].imshow(img2)
    ax[1].set_title('2')
    
    ax[2].imshow(img3)
    ax[2].set_title('3')
    
    ax[3].imshow(img4)
    ax[3].set_title('4')


##################################################################
#Functions for morphological operations
##################################################################
def make8UC(mat):
    """
    Converts the matrix to the equivalent matrix of the unsigned 8 bit integer datatype
    Returns the equivalent uint8 matrix
    """
    mat_256 = mat[:,:]# *255
    mat_256.round()
    mat_8UC = np.uint8(mat_256)
    
    return mat_8UC

def make8UC3(mat):
    """
    Converts the matrix to the equivalent matrix of the unsigned 8 bit integer datatype with 3 channels
    Returns the equivalent uint8 matrix
    """
    mat_8UC = make8UC(mat)
    mat_8UC3 = np.stack((mat_8UC,)*3, axis = -1)
    
    return mat_8UC3

windowIndex = 0
# Equivalent of MATLAB's imfill(BW, 'holes')
def fillHoles(bwMask):
    rWidth,cWidth  = bwMask.shape
    # Needs to be 2 pixels larger than image sent to floodFill per API (not sure why)
    mask = np.zeros((rWidth+4, cWidth+4), np.uint8)
    # Add one pixel of padding all around so that objects touching border aren't filled against border
    bwMaskCopy = np.zeros((rWidth+2, cWidth+2), np.uint8)
    bwMaskCopy[1:(rWidth+1), 1:(cWidth+1)] = bwMask
    cv2.floodFill(bwMaskCopy, mask, (0, 0), 255)
    bwMask = bwMask | (255-bwMaskCopy[1:(rWidth+1), 1:(cWidth+1)])
    return bwMask

# Equivalent of bwareaopen(BW, P)
def deleteSmallObjects(bwMask, minPixelCount):
    maskToDelete = np.ones(bwMask.shape, np.uint8)*255
    im, contours, h = cv2.findContours(bwMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minPixelCount:
            # first -1 indicates draw all contours, 255 is value to draw, -1 means draw interiors
            cv2.drawContours(maskToDelete, [contour], -1, 0, -1)
    bwMask = bwMask & maskToDelete
    return bwMask

# Used for computing circularity - see deleteNonCircular
def circularity(area, perim):
    return (perim*perim)/(4*math.pi*area)

# Remove objects from bwMask with circularity lower than circThreshold 
# Circularity calculated using circularity function above
def deleteNonCircular(bwMask, circThreshold):
    maskToDelete = np.ones(bwMask.shape, np.uint8)*255
    im, contours, h = cv2.findContours(bwMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if  area == 0 or circularity(area, cv2.arcLength(contour,True)) > circThreshold:
            print(circularity(area, cv2.arcLength(contour,True)))
            # first -1 indicates draw all contours, 255 is value to draw, -1 means draw interiors
            cv2.drawContours(maskToDelete, [contour], -1, 0, -1)
    bwMask = bwMask & maskToDelete
    return bwMask

# Mask generation using Otsu thresholding
def showProcessing(img, thresh = None, plotImage = False, fillHole = False):
    """
    input must be a numpy array and 8 bit unsigned integer datatype
    Returns the processed image: mask_holesFilled
    Opening and closing were not considered since they created problems
    """
    if thresh is None:
        # Perform Otsu thresholding
        thresh, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else: 
        thresh, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        
    # Delete Small Objects 
    numPixelsInImage = img.shape[0] * img.shape[1]
    minPixelCount = 2
    mask_smallDeleted = deleteSmallObjects(mask, minPixelCount)
    
    # Fill holes
    if fillHole:
        mask_holesFilled = fillHoles(mask_smallDeleted)
    else:
        mask_holesFilled = mask_smallDeleted  
    
    if plotImage:
        plt.imshow(img)
        plt.title("Original")
        plt.show()

        plt.imshow(mask)
        plt.title("Otsu thresholding")
        plt.colorbar()
        plt.show()

        plt.imshow(mask_smallDeleted)
        plt.title("Small objects deleted")
        plt.show()
        
        if fillHole:
            plt.imshow(mask_holesFilled)
            plt.title("Filled holes")
            plt.show()
            
    return mask_holesFilled

def watershed(mask, img, plotImage = False, kernelSize = None):
    """
    Do watershed segmentation on a non noisy binary image
    Returns the image with the nuclei segmented
    """
    imgCopy = img.copy()
    maskCopy = np.array(mask.copy(), dtype=np.uint8)
    
    if kernelSize is None:
        kernelSize = 2

    # Finding sure foreground area
    #dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0) #change the second argument to change the sensitivity 
    maskClosed = skimage.morphology.closing(np.array(maskCopy, dtype=np.uint8))
    maskClosed = skimage.morphology.closing(np.array(maskClosed, dtype=np.uint8))
    kernel = np.ones((kernelSize,kernelSize), np.uint8)
    # maskCopy = img_as_bool(maskCopy)
    sure_fg = cv2.erode(maskClosed, kernel, iterations = 2) ###
    sure_fg = skimage.morphology.closing(np.array(sure_fg, dtype=np.uint8))
    # kernel = np.ones((2,2), np.uint8)
    # sure_fg = binary_closing(sure_fg, kernel)
    
    # sure background area
    #kernel = np.ones((5, 5), np.uint8)
    #sure_bg = cv2.dilate(mask, kernel, iterations = 1)
    sure_fg_bool = 1 - img_as_bool(sure_fg)
    # sure_bg = np.uint8(1 - morphology.medial_axis(sure_fg_bool)) ### 
    sure_bg = np.uint8(1 - morphology.skeletonize(sure_fg_bool))
    sure_bg[0, :] = 1
    sure_bg[-1, :] = 1
    sure_bg[:, 0] = 1
    sure_bg[:, -1] = 1
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    if plotImage:
        plt.figure()
        plt.imshow(sure_fg)
        plt.title("Inner Marker")
        plt.figure()
        plt.imshow(sure_bg)
        plt.title("Outer Marker")
        plt.figure()
        plt.imshow(unknown)
        plt.title("Unknown")
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==1] = 0
    
    if plotImage:
        plt.figure()
        plt.imshow(markers, cmap='jet')
        plt.title("Markers")
    
    # Do watershed
    markers = cv2.watershed(imgCopy, markers)
    
    imgCopy[markers == -1] = [0, 255 ,0]

    if plotImage:
        plt.figure()
        plt.imshow(markers,cmap='jet')
        plt.title("Mask")
        plt.figure()
        plt.imshow(img)
        plt.title("Original Image")
        plt.figure()
        plt.imshow(imgCopy)
        plt.title("Marked Image")
        plt.show()

    return markers

def channelDeconvolution (patch, stainingType, plotImage = False):
    """
    Input: a numpy array patch of a tissue slide with RGB channels
    Output: different channels of input image
    """
    
    if stainingType == "HDB":
        channels = ("Hematoxylin", "DAB", "Background") #hematoxylin, DAB, background
    elif stainingType == "HRB":
        channels = ("Hematoxylin", "Fast Red", "Background") #hematoxylin, red dye, background
    elif stainingType == "HDR":
        channels = ("Hematoxylin", "DAB", "Fast Red") #hematoxylin, DAB, red dye
    elif stainingType == "HEB":
        channels = ("Hematoxylin", "Eosin", "Background") #hematoxylin, Eosin, background
    else:
        print("Staining type not defined. Choose one from the following: HDB, HRB, HDR, HEB")
        return 0
    
    #plt.imshow(patch)
    #height, width, channel = patch.shape
    #print(height,width,channel)
    
    ODmax, C = colorDeconvolution(patch, 255, 255, 255, stainingType = stainingType)
    
    #define each channel
    Channel1 = C[:,:,0] #hematoxylin
    Channel2 = C[:,:,1] #DAB
    Channel3 = C[:,:,2] #background

    #remove problematic pixels in each layer
    Channel1 = removeInfinities(Channel1)
    Channel2 = removeInfinities(Channel2)
    Channel3 = removeInfinities(Channel3)

    #normalize each layer
    Channel1 = normalize(Channel1)
    Channel2 = normalize(Channel2)
    Channel3 = normalize(Channel3)
    
    #plot each channel
    if plotImage: 
        fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(patch)
        ax[0].set_title("Original image")

        ax[1].imshow(Channel1, cmap = "gray")
        ax[1].set_title(channels[0])

        ax[2].imshow(Channel2, cmap = "gray")
        ax[2].set_title(channels[1])

        ax[3].imshow(Channel3, cmap = "gray")
        ax[3].set_title(channels[2])
        
        plt.show()


    return C, Channel1, Channel2, Channel3

def generateMask(channel, plotProcess = False, plotResult = False, fillHole = False, normalizeImg = True,
                 originalImg = None, overlapColor = (0, 1, 0), title = "", useWatershed = True,
                 watershedKernelSize = None,
                 saveImg = False, savePath = None, thresh = None):
    """
    Input: channel before normalization
    Input: originalImg for plotting overlapped segmentation result
    Output: binary mask
    """  
    if not checkChannel(channel):
        #if there is not any signal
        nrow, ncol = channel.shape
        mask = np.zeros((nrow, ncol))
        print("No signals detected for this channel")
        return mask
    else:
        channel = removeInfinities(channel)
        #channel[channel < 0] = 0
        if normalizeImg:
            channel = normalize(channel)
        if useWatershed:
            mask_threshold = showProcessing(make8UC(channel), plotImage = plotProcess, fillHole = fillHole, thresh = thresh)
            marker = watershed(mask_threshold, make8UC3(channel), plotImage = plotProcess, kernelSize = watershedKernelSize)     
            # create mask
            mask = np.zeros(marker.shape)
            mask[marker == 1] = 1
            mask = 1-mask
            # Set border as mask from threshold
            mask[0, :] = mask_threshold[0, :] == 255
            mask[-1, :] = mask_threshold[-1, :] == 255
            mask[:, 0] = mask_threshold[:, 0] == 255
            mask[:, -1] = mask_threshold[:, -1] == 255
            if plotResult or saveImg:
                if originalImg is None:
                    #if original image is not provided
                    plt.figure()
                    plt.imshow(mask, cmap = "gray")
                    if plotResult:
                        plt.show()
                else:
                    if len(originalImg.shape) == 3:
                        #create overlapped image
                        #overlappedImg = originalImg.copy()
                        #overlappedImg[marker == -1] = overlapColor
                        overlappedImg = segmentation.mark_boundaries(originalImg, skimage.measure.label(mask), overlapColor, mode = "thick")

                        #if RGB image provided
                        fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True,
                                             subplot_kw={'adjustable': 'box-forced'})
                        ax = axes.ravel()
                        ax[0].imshow(mask, cmap = "gray")
                        ax[0].set_title(str(title)+" Mask")
                        ax[1].imshow(overlappedImg)
                        ax[1].set_title("Overlapped with Original Image")
                        if plotResult:
                            plt.show()         
                    elif len(originalImg.shape) == 2:
                        #create overlapped image
                        #overlappedImg = originalImg.copy()
                        #overlappedImg[marker == -1] = overlapColor
                        overlappedImg = segmentation.mark_boundaries(originalImg, skimage.measure.label(mask), overlapColor, mode = "thick")

                        #if RGB image provided
                        fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True,
                                             subplot_kw={'adjustable': 'box-forced'})
                        ax = axes.ravel()
                        ax[0].imshow(mask, cmap = "gray")
                        ax[0].set_title(str(title)+" Mask")
                        ax[1].imshow(overlappedImg, cmap = "gray")
                        ax[1].set_title("Overlapped with Original Channel")
                        if plotResult:
                            plt.show()
                    else:
                        print ("Error manipulate original image.")
        else:
            mask = showProcessing(make8UC(channel), plotImage = plotProcess, fillHole = fillHole, thresh = thresh)
            if plotResult:
                if originalImg is None:
                    #if original image is not provided
                    plt.figure()
                    plt.imshow(mask, cmap = "gray")
                    if plotResult:
                        plt.show()
                else:
                    if len(originalImg.shape) == 3:
                        #create overlapped image
                        overlappedImg = segmentation.mark_boundaries(originalImg, skimage.measure.label(mask), overlapColor, mode = "thick")

                        #if RGB image provided
                        fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True,
                                             subplot_kw={'adjustable': 'box-forced'})
                        ax = axes.ravel()
                        ax[0].imshow(mask, cmap = "gray")
                        ax[0].set_title(str(title)+" Mask")
                        ax[1].imshow(overlappedImg)
                        ax[1].set_title("Overlapped with Original Image")
                        if plotResult:
                            plt.show()         
                    elif len(originalImg.shape) == 2:
                        #create overlapped image
                        overlappedImg = segmentation.mark_boundaries(originalImg, skimage.measure.label(mask), overlapColor, mode = "thick")

                        #if RGB image provided
                        fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True,
                                             subplot_kw={'adjustable': 'box-forced'})
                        ax = axes.ravel()
                        ax[0].imshow(mask, cmap = "gray")
                        ax[0].set_title(str(title)+" Mask")
                        ax[1].imshow(overlappedImg, cmap = "gray")
                        ax[1].set_title("Overlapped with Original Channel")
                        if plotResult:
                            plt.show()
                    else:
                        print ("Error manipulate original image.")
        if saveImg:
            plt.savefig(savePath)
        plt.close()
    return mask


def getMaskForSlideImage(filePath, displayProgress=False):
    slide = open_slide(filePath)
    
    # Want to capture whole image, so take first level with size less than MAX_NUM_PIXELS
    levelDims = slide.level_dimensions
    levelToAnalyze = len(levelDims)-1
    dimsOfSelected = levelDims[-1]

    if displayProgress:
        print('Selected image of size (' + str(levelDims[levelToAnalyze][0]) + ', ' + str(levelDims[levelToAnalyze][1]) + ')')
    slideImage = slide.read_region((0, 0), levelToAnalyze, levelDims[levelToAnalyze])
    slideImageCV = np.array(slideImage)
    # Imported image is RGB, flip to get BGR, this way imshow will understand correct ordering
    slideImageCV = slideImageCV[:,:,(2,1,0)]
    if displayProgress:
        plt.figure()
        plt.imshow(slideImageCV)
        
    # Perform Otsu thresholding
    threshB, maskB = cv2.threshold(slideImageCV[:,:,0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshG, maskG = cv2.threshold(slideImageCV[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshR, maskR = cv2.threshold(slideImageCV[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if displayProgress:
        plt.figure()
        plt.imshow(maskR)
        plt.figure()
        plt.imshow(maskG)
        plt.figure()
        plt.imshow(maskB)

    # Add the channels together
    bwMask = ((255-maskR) | (255-maskG) | (255-maskB))
    if displayProgress:
        plt.figure()
        plt.imshow(bwMask)

    # ADDED FOR LUNG CANCER (not in google)---------------------
    # Dilate the image
    kernel = np.ones((3,3), np.uint8)
    bwMask = cv2.dilate(bwMask, kernel, iterations=3)
    #-----------------------------------------------------------
    if displayProgress:
        plt.figure()
        plt.imshow(bwMask)
    
    # Delete small objects
    numPixelsInImage = dimsOfSelected[0] * dimsOfSelected[1]
    minPixelCount = 0.0005 * numPixelsInImage
    bwMask = deleteSmallObjects(bwMask, minPixelCount)
    if displayProgress:
        plt.figure()
        plt.imshow(bwMask)
        
    # Dilate the image
    kernel = np.ones((3,3), np.uint8)
    bwMask = cv2.dilate(bwMask, kernel, iterations=5)
    bwMask = cv2.erode(bwMask, kernel, iterations=3)
    bwMask = cv2.dilate(bwMask, kernel, iterations=2)
    # Fill holes
    bwMask = fillHoles(bwMask)
    if displayProgress:
        plt.figure()
        plt.imshow(bwMask)
    
    #---------------------------------------------------------
    # BEGIN OF Delete square-like objects
    # Add up the second derivative around perimieter to get curvature and delete
    # objects with very low curvature (linear objects)
    
    rWidth, cWidth = bwMask.shape
    # Add 1 pixel padding
    maskPad = np.zeros((rWidth+2, cWidth+2), np.uint8)
    maskPad[1:(rWidth+1), 1:(cWidth+1)] = bwMask
    kernel = np.ones((3,3), np.uint8)
    
    maskReduced = cv2.erode(maskPad, kernel, iterations=1)
    maskPerim = maskReduced - maskPad
    # Remove the one pixel padding
    maskPerim = maskPerim[1:(rWidth+1), 1:(cWidth+1)]
    im, contours, h = cv2.findContours(maskPerim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im, contoursFull, h = cv2.findContours(bwMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maskToDelete = np.ones(bwMask.shape, np.uint8)*255
    for index in range(0,len(contours)):
        contour = contours[index]
        xCoords = []
        yCoords = []
        for point in contour:
            xCoords.append(point[0][0])
            yCoords. append(point[0][1])
        total = np.sum(np.abs(np.diff(np.diff(xCoords)))) + np.sum(np.abs(np.diff(np.diff(yCoords))))	
        total = total/(len(xCoords))
        if total < 0.20:
            print("Deleting contour with lin value of " + str(total))
            cv2.drawContours(maskToDelete, [contoursFull[index]], -1, 0, -1)
        else:
            pass

    # END OF Delete square-like objects
    #---------------------------------------------------------
    
    # Delete artifacts such as slide labels by circularity
    if displayProgress:
        plt.figure()
        plt.imshow(bwMask)
        plt.figure()
        plt.show()
    return bwMask, slideImageCV
    
##################################################################
#Functions for extracting patches from slide image
##################################################################
def extractPatchByLocation(filepath, location, patchSize = (500, 500), 
                           plotImage = False, levelToAnalyze = 0, save = False, savepath = '.'):
    if not os.path.isfile(filepath):
        raise IOError("Image not found!")
        return []
    
    slide = open_slide(filepath)
    filename = re.search("(?<=/)[0-9]+\.svs", filepath).group(0)[0:-4]
    slideImage = slide.read_region(location, levelToAnalyze, patchSize)
    if plotImage:
        plt.figure()
        plt.imshow(slideImage)
        plt.show()
        
    if save:
        savename = os.path.join(savepath, str(filename)+'_'+str(location[0])+'_'+str(location[1])+'.png')
        misc.imsave(savename, slideImage)
        print("Writed to "+savename)
    return slideImage
    
def extractPatchByTissueArea(filePath, nPatch=0, maxPatch=10, filename=None, savePath=None, displayProgress=False):
    '''Input: slide
       Output: image patches'''
    if filename is None:
        filename = re.search("(?<=/)[0-9]+\.svs", filePath).group(0)
    if savePath is None:
        savePath = '/home/swan15/python/brainTumor/sample_patches/'
    bwMask, slideImageCV = getMaskForSlideImage(filePath, displayProgress=displayProgress)
    slide = open_slide(filePath)
    levelDims = slide.level_dimensions
    #find magnitude
    for i in range(0, len(levelDims)):
        if bwMask.shape[0] == levelDims[i][1]:
            magnitude = levelDims[0][1]/levelDims[i][1]
            break
    nCol = int(math.ceil(levelDims[0][1]/patchSize))
    nRow = int(math.ceil(levelDims[0][0]/patchSize))
    #get contour
    _, contours, _ = cv2.findContours(bwMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for nContours in range(0, len(contours)):
        print(nContours)
        # i is the y axis in the image
        for i in range(0, nRow):
            minRow = i*patchSize/magnitude
            maxRow = (i+1)*patchSize/magnitude
            matches = [x for x in range(0, len(contours[nContours][:, 0, 0]))
                       if (contours[nContours][x, 0, 1] > minRow and contours[nContours][x, 0, 1] < maxRow)]
            try:
                print [min(contours[nContours][matches, 0, 0]), max(contours[nContours][matches, 0, 0])]
                
                #save image
                minCol = min(contours[nContours][matches, 0, 0])*magnitude
                maxCol = max(contours[nContours][matches, 0, 0])*magnitude
                minColInt = int(math.floor(minCol/patchSize))
                maxColInt = int(math.ceil(maxCol/patchSize))
                
                for j in range(minColInt, maxColInt):
                    startCol = j*patchSize
                    startRow = i*patchSize
                    patch = slide.read_region((startCol, startRow), desiredLevel, (patchSize, patchSize))
                    patchCV = np.array(patch)
                    patchCV = patchCV[:, :, 0:3]
                    
                    fname = os.path.join(savePath, filename+'_'+str(i)+'_'+str(j)+'.png')
                    
                    if not os.path.isfile(fname):
                        misc.imsave(fname, patchCV)
                        nPatch = nPatch + 1
                        print(nPatch)
                    
                    if nPatch >= maxPatch:
                        break
            except ValueError:
                continue      
            if nPatch >= maxPatch:
                break   
        if nPatch >= maxPatch:
            break
            
def parseXML(xmlFile, pattern):
    """
    Parse XML File and returns an object containing all the vertices 
    Verticies: (dict)
         pattern: (list) of dicts, each with 'X' and 'Y' key 
                [{ 'X': [1,2,3], 
                   'Y': [1,2,3]  }]
    """
    
    tree = ET.parse(xmlFile) # Convert XML file into tree representation
    root = tree.getroot()

    regions = root.iter('Region') # Extract all Regions
    vertices = {pattern: []} # Store all vertices in a dictionary

    for region in regions: 
        label = region.get('Text') # label either as 'ROI' or 'normal'
        if label == pattern:
            vertices[label].append({'X':[], 'Y':[]})

            for vertex in region.iter('Vertex'): 
                X = float(vertex.get('X'))
                Y = float(vertex.get('Y'))

                vertices[label][-1]['X'].append(X)
                vertices[label][-1]['Y'].append(Y)

    return vertices

def calculateRatio(levelDims):
    """ Calculates the ratio between the highest resolution image and lowest resolution image.
    Returns the ratio as a tuple (Xratio, Yratio). 
    """
    highestReso = np.asarray(levelDims[0])
    lowestReso = np.asarray(levelDims[-1])
    Xratio, Yratio = highestReso/lowestReso
    return (Xratio, Yratio)

def createMask(levelDims, vertices, pattern):
    """
    Input: levelDims (nested list): dimensions of each layer of the slide.
           vertices (dict object as describe above)
    Output: (tuple) mask
            numpy nd array of 0/1, where 1 indicates inside the region
            and 0 is outside the region
    """
    # Down scale the XML region to create a low reso image mask, and then 
    # rescale the image to retain reso of image mask to save memory and time 
    Xratio, Yratio = calculateRatio(levelDims)

    nRows, nCols = levelDims[-1]
    mask = np.zeros((nRows, nCols), dtype=np.uint8)

    for i in range(len(vertices[pattern])):
        lowX = np.array(vertices[pattern][i]['X'])/Xratio
        lowY = np.array(vertices[pattern][i]['Y'])/Yratio
        rr, cc = polygon(lowX, lowY, (nRows, nCols))
        mask[rr, cc] = 1

    return mask

def getMask(xmlFile, svsFile, pattern):
    """ Parses XML File to get mask vertices and returns matrix masks 
    where 1 indicates the pixel is inside the mask, and 0 indicates outside the mask.

    @param: {string} xmlFile: name of xml file that contains annotation vertices outlining the mask. 
    @param: {string} svsFile: name of svs file that contains the slide image.
    @param: {pattern} string: name of the xml labeling
    Returns: slide - openslide slide Object 
             mask - matrix mask of pattern
    """
    vertices = parseXML(xmlFile, pattern) # Parse XML to get vertices of mask
    
    if not len(vertices[pattern]):
        slide = 0
        mask = 0
        return slide, mask

    slide = open_slide(svsFile)
    levelDims = slide.level_dimensions
    mask = createMask(levelDims, vertices, pattern)

    return slide, mask

def plotMask(mask):
    fig, ax1 = plt.subplots(nrows=1, figsize=(6,10))
    ax1.imshow(mask)
    plt.show()

def chooseRandPixel(mask):
    """ Returns [x,y] numpy array of random pixel.
    @param {numpy matrix} mask from which to choose random pixel.
    """
    array = np.transpose(np.nonzero(mask)) # Get the indices of nonzero elements of mask.
    index = random.randint(0,len(array)-1) # Select a random index
    return array[index]

def plotImage(image):
    plt.imshow(image)
    plt.show()
    
def checkWhiteSlide(image):
    im = np.array(image.convert(mode='RGB'))
    pixels = np.ravel(im)
    mean = np.mean(pixels)
    return mean >= 230

def getPatches(slide, mask, numPatches=0, dims=(0,0), dirPath='', slideNum='', plot=False, plotMask=False):
    # extractPatchByXMLLabeling
    """ Generates and saves 'numPatches' patches with dimension 'dims' from image 'slide' contained within 'mask'.
    @param {Openslide Slide obj} slide: image object
    @param {numpy matrix} mask: where 0 is outside region of interest and 1 indicates within 
    @param {int} numPatches
    @param {tuple} dims: (w,h) dimensions of patches
    @param {string} dirPath: directory in which to save patches
    @param {string} slideNum: slide number 
    Saves patches in directory specified by dirPath as [slideNum]_[patchNum]_[Xpixel]x[Ypixel].png
    """ 
    w,h = dims 
    levelDims = slide.level_dimensions
    Xratio, Yratio = calculateRatio(levelDims)

    i = 0
    while i < numPatches:
        firstLoop = True # Boolean to ensure while loop runs at least once. 

        while firstLoop: # or not mask[rr,cc].all(): # True if it is the first loop or if all pixels are in the mask 
            firstLoop = False
            x, y = chooseRandPixel(mask) # Get random top left pixel of patch. 
            xVertices = np.array([x, x+(w/Xratio), x+(w/Xratio), x, x])
            yVertices = np.array([y, y, y-(h/Yratio), y-(h/Yratio), y])
            rr, cc = polygon(xVertices, yVertices)

        image = slide.read_region((x*Xratio, y*Yratio), 0, (w,h))
        
        isWhite = checkWhiteSlide(image)
        newPath = 'other' if isWhite else dirPath
        if not isWhite: i += 1

        slideName = '_'.join([slideNum, 'x'.join([str(x*Xratio),str(y*Yratio)])])
        image.save(os.path.join(newPath, slideName+".png"))

        if plot: 
            plotImage(image)
        if plotMask: mask[rr,cc] = 0

    if plotMask:
        plotImage(mask)
        
'''Example codes for getting patches from labeled svs files:
#define the patterns
patterns = ['small_acinar', 
            'large_acinar',
            'tubular',
            'trabecular', 
            'aveolar', 
            'solid', 
            'pseudopapillary', 
            'rhabdoid',
            'sarcomatoid',
            'necrosis', 
            'normal', 
            'other']
#create folders
for pattern in patterns:
    if not os.path.exists(pattern):
        os.makedirs(pattern)
#define parameters
patchSize = 500
numPatches = 50
dirName = '/home/swan15/kidney/ccRCC/slides' 
annotatedSlides = 'slide_region_of_interests.txt'

f = open(annotatedSlides, 'r+')
slides = [re.search('.*(?=\.svs)', line).group(0) for line in f 
          if re.search('.*(?=\.svs)', line) is not None]
print slides
f.close()
for slideID in slides:
    print('Start '+slideID)
    try: 
        xmlFile = slideID+'.xml'
        svsFile = slideID+'.svs'

        xmlFile = os.path.join(dirName, xmlFile)
        svsFile = os.path.join(dirName, svsFile)
        
        if not os.path.isfile(xmlFile):
            print xmlFile+' not exist'
            continue
        
        for pattern in patterns:
            
            numPatchesGenerated = len([files for files in os.listdir(pattern)
                                      if re.search(slideID+'_.+\.png', files) is not None])
            if numPatchesGenerated >= numPatches:
                print(pattern+' existed')
                continue
            else:
                numPatchesTemp = numPatches - numPatchesGenerated
                
            slide, mask = getMask(xmlFile, svsFile, pattern)
            
            if not slide:
                #print(pattern+' not detected.')
                continue
            
            getPatches(slide, mask, numPatches = numPatchesTemp, dims = (patchSize, patchSize), 
                       dirPath = pattern+'/', slideNum = slideID, plotMask = False)  # Get Patches
            print(pattern+' done.')

        print('Done with ' + slideID)
        print('----------------------')

    except:
        print('Error with ' + slideID)
'''

def extractPatchByLocation(filepath, location, patchSize = (500, 500), 
                           plotImage = False, levelToAnalyze = 0, save = False, savepath = '.'):
    if not os.path.isfile(filepath):
        raise IOError("Image not found!")
        return []
    
    slide = open_slide(filepath)
    filename = re.search("(?<=/)[0-9]+\.svs", filepath).group(0)[0:-4]
    slideImage = slide.read_region(location, levelToAnalyze, patchSize)
    if plotImage:
        plt.figure()
        plt.imshow(slideImage)
        plt.show()
        
    if save:
        savename = os.path.join(savepath, str(filename)+'_'+str(location[0])+'_'+str(location[1])+'.png')
        misc.imsave(savename, slideImage)
        print("Writed to "+savename)
    return slideImage
    
##################################################################
#Some functions relating to RGB color processing
##################################################################

# convert RGBA image to RGB
def convertRGBA(RGBA_img):
    if np.shape(RGBA_img)[2] == 4:
        RGB_img = np.zeros((np.shape(RGBA_img)[0], np.shape(RGBA_img)[1], 3))
        RGB_img[RGBA_img[:, :, 3] == 0] = [255, 255, 255]
        RGB_img[RGBA_img[:, :, 3] == 255] = RGBA_img[RGBA_img[:, :, 3] == 255, 0:3]
        return RGB_img
    else:
        print("Not an RGBA image")
        return RGBA_img
        
# Convert RGB mask to one-channel mask
def RGBToIndex(RGB_img, RGBmarkers = None):
    '''
    RGBmarkers: start from background (marked as 0); 
    Example format:
        [[255, 255, 255],
        [160, 255, 0]]
    '''
    if np.shape(RGB_img)[2] is not 3:
        print("Not an RGB image")
        return RGB_img
    else:
        if RGBmarkers == None:
            RGBmarkers = [[255, 255, 255]]
        maskIndex = np.zeros((np.shape(RGB_img)[0], np.shape(RGB_img)[1]))
        for i in range(np.shape(RGBmarkers)[0]):
            maskIndex[np.all(RGB_img==RGBmarkers[i],axis=2)] = i
    return maskIndex