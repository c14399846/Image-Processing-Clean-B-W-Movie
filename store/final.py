## C14399846 DIT COMPUTER SCIENCE
# (C) OLEG PETCOV

# import the necessary packages:
#
# Need to pip install:
# [imutils, easygui, cv2]

from __future__ import division
import sys
import math
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
from collections import deque
import time


# Sharpening kernels
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)

# Used for dilation and erosion
largeMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
smallMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

videofile = easygui.fileopenbox()

video = cv2.VideoCapture(videofile)
#video = cv2.VideoCapture("../../Zorro.mp4")

# Counter for timekeeping purposes
startTime = time.time()


############################################################
# NOTE: NOT FULLY MY CODE
# TOOK "INSPIRATION" FROM THIS CODE
# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
# https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
#
# Purpose:
#	Take a pair of images (e.g frame 1 and 2).
#	Compare any similarities (mean and ssim).
#	Store those values and frame number in array.
#
#	Was going to be used fo scene identification:
#		See files: [mean.png, ssim.png, mean_inv.png]
#
#		The largest spikes correspond to change in scenes,
#		and the heavy movement towards the end.
############################################################
#
#		Can take frames up until the spikes in mean shift / ssim shift.
#
#		Store those frames as an array, and put that array into a list.
#		Use each array as a 'scene', and compute contrast, noise, colourspace, etc
#		uniquely for each 'scene' in the list of arrayed scenes.
#
############################################################
'''
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, framenum):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# Add an array of the computed data
	array.append([framenum,m,s])
'''
#############################################################


# Inpainting
# Uses first frame to get a mask image
# relies on backgorund of that frame being black
video.set(0,0)
(grabbed, I) = video.read()	
frame = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

width, height, channels = I.shape

# Thresh is better clarity here, rather than 0 
ret, frame_mask = cv2.threshold(frame, thresh = 10, maxval = 255, type = cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(frame_mask)

# Array used to store frames from video
Image_data = []
#Image_queue = deque()


# reset starting frame to 0
video.set(0,0)

# This is used to get rid of the watermark in the corner,
# and converts the images to gray. 
# Stored in the 'Image_data' array
while (video.isOpened()):

	(grabbed, I) = video.read()	

	if grabbed is not False:
		inpaint = cv2.inpaint(I,mask_inv,1,cv2.INPAINT_TELEA)
		gray = cv2.cvtColor(inpaint, cv2.COLOR_BGR2GRAY)
		Image_data.append(gray)
	else:
		break

print("FINISHED REMOVING WATERMARK\n")

video.release()

# creates videofile to write to
# video is saved at 24 fps, which looks a bit better (not as fast looking)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
newVideo = cv2.VideoWriter('./frames/ZorroClean2.avi',fourcc, 24.0, (width,height))

#Image_data
framepos = 0
totalframes = len(Image_data)
framecount = totalframes - 1
#print(totalframes)

if totalframes == 0:
	print ("No frames read in.\n")
	exit()

cleanedFrames = []
	
while (framepos <= framecount):

	#######################
	# NOT MINE
	#
	# It shows percentage of progress as the program is executing
	# Uses amount of frames in Image_data, compared to 'framepos'
	#
	# http://blog.montmere.com/2013/04/17/how-to-show-a-percent-complete-in-python/
	#######################
	sys.stdout.write('\r')
	sys.stdout.write('%.2f%% complete' % (framepos / totalframes * 100))
	sys.stdout.flush()
	

	frame = Image_data[framepos]
	
	# Remnants of SSIM / mean code
	#frame = Image_data[framepos]
	#if grabbed is False:
	#	break
	#frame2 = Image_data[framepos+1]
	#if grabbed2 is False:
	#	break
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	#compare_images(frame,frame2, framepos)

	#inpaint = cv2.inpaint(frame,mask_inv,1,cv2.INPAINT_TELEA)
	
	
	# Used to preserve edges of objects within the scene, while smoothing it
	# Very impolrtant for the walls and other edged surfaces (doors and tables)
	#
	# Sharpened to 'pop' some of the details
	
	bilat = cv2.bilateralFilter(frame, 7, 75, 75)
	sharpenBilat = cv2.filter2D(bilat, ddepth = -1, kernel = kernelSharp)


	# further smoothing of the image
	# While helping to (lightly) fill in any blotches
	
	dilated = cv2.dilate(sharpenBilat,smallMorph)
	eroded = cv2.erode(dilated,largeMorph)
	dilatedFinal = cv2.dilate(eroded,smallMorph)

	
	# removes noise from blur, sharpen and morphs.
	# A very light denoise pass here
	den = cv2.fastNlMeansDenoising(dilatedFinal,None,3,3,9)

	# Further sharpen to preserve feeature erosion form smoothing
	sharpenNoise = cv2.filter2D(den, ddepth = -1, kernel = kernelSharp)

	# Fills in more minor blotches,
	# and helps to smooth wall and background surfaces
	#
	# Unfortunately does 'brighten' some whitepoints in frames,
	# May need work on contrasting fixes to counteract.
	sharpDil = cv2.dilate(sharpenNoise,smallMorph)
	sharpEro = cv2.erode(sharpDil,largeMorph)
	sharpDilFinal = cv2.dilate(sharpEro,smallMorph)
	
	# Final blur (Gaussian) and heavier denoise is applied here
	# edge preservation less important here
	# detail is lost in denser areas 
	# (look at cloak / blanket in fight scene and on stairs)
	gBlurred = cv2.GaussianBlur(sharpDilFinal,(3,3),0)
	denoisedGray = cv2.fastNlMeansDenoising(gBlurred,None,8,7,21)
	
	

	# Converts back to BGR for formatting
	# Adds frame to videofile
	# Updates position of frame in array
	output = cv2.cvtColor(denoisedGray, cv2.COLOR_GRAY2BGR)
	#cv2.imshow('output',output)	
	cleanedFrames.append(output)
	
	#newVideo.write(output)
	
	framepos+=1
	
	key = cv2.waitKey(1)
	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

		
print (len(cleanedFrames))
for fr in range(framecount):
	newVideo.write(cleanedFrames[fr])

newVideo.release()



# Part of SSIm and mean code
# Used with matplotlib to show the change ranges of frames
# again, look at [mean.png, mean_inv.png, ssim.png]

#diffArr = np.diff(array[:,1])
'''
arrS = []
arrM = []

for x in range(framecount):
	#f = array[x][0]
	m = array[x][1]
	s = array[x][2]
	#m = diffArr[x][1]
	#print("\nDiff:" + str(diffArr[x]))
	
	arrS.append(s)
	arrM.append(m)
		
	#print (f)
	#print (m)
	#print (s)
	
	# Gets difference in second row, the mean values
	#diffArr = numpy.diff(a[:,1])
	#a[:,1]

#SSIM	
plt.plot(arrS)
plt.show()

#mean
plt.plot(arrM)
plt.show()
'''

# Trying out peak value detection tools.
# Iffy.
#
#peaks = peakdetect(arr, lookahead=100)
#indexes = find_peaks_cwt(arr, np.arange(1, max(arr))) # Better for positive peaks	
#indexes2 = peakutils.indexes(arr, thres=0.02/max(arr), min_dist=100)	
	
endTime = time.time()
print('\n')
print(endTime - startTime)
#cv2.destroyAllWindows()