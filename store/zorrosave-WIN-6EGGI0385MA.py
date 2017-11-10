# C14399846 DIT COMPUTER SCIENCE
# (C) OLEG PETCOV

# import the necessary packages:
from __future__ import division
from collections import deque
from matplotlib import pyplot as plt
from matplotlib import image as image
import numpy as np
import os, errno
import sys
import math
import cv2
import imutils
import easygui
import time

#args = argparse.ArgumentParser()
#args.add_argument("-f", "--frame", required=True, help="starting frame (starts at 0)")

#args.add_argument('frame', action="store", type=int)

# Opening an image using a File Open dialog:
videofile = easygui.fileopenbox()


#Capturing an image from a webcam:
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)


kernel2 = np.ones((5,5),np.uint8)
largeMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
smallMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))






video = cv2.VideoCapture(videofile)
(grabbed, I) = video.read()


startTime = time.time()


#I = imutils.resize(I, width=640)

# Video Capture:
grabbed = True

# Using this to count frames
fr = 0;
font = cv2.FONT_HERSHEY_SIMPLEX

#height, width, channels = I.shape
height, width, _ = I.shape
	
	
###################################
# Inpainting
# Uses first frame
video.set(0,0)
(grabbed, I) = video.read()	
I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

# Thresh is better clarity here, rather than 0 
ret, mask = cv2.threshold(I, thresh = 20, maxval = 255, type = cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)


# Set to false, using image frames instead of video files
makingVideo = False
directory = "./cleaned/"

if not os.path.exists(directory):
    os.makedirs(directory)

if makingVideo is True:		
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	newVideo = cv2.VideoWriter(directory + 'ZorroClean.avi',fourcc, 24.0, (width,height))


#Image_data
#i = 1
#framecount = len(Image_data) - 1

frameNum = 0 # counting frames
startFrame = 250 # start at this frame (0 is first)
frameStop = 10 # stop after 150 frames

totalFrames = frameStop + startFrame

video.set(0,startFrame)
#while (i < framecount):
#while (Image_queue):
#while (grabbed):
while ( grabbed and frameNum < frameStop):
	
	#######################
	# NOT MINE
	#
	# It shows percentage of progress as the program is executing
	# Uses amount of frames in Image_data, compared to 'framepos'
	#
	# http://blog.montmere.com/2013/04/17/how-to-show-a-percent-complete-in-python/
	#######################
	sys.stdout.write('\r')
	sys.stdout.write('%.2f%% complete' % (frameNum / frameStop * 100))
	sys.stdout.flush()
	
	currentFrame = startFrame + frameNum # current frame position

	(grabbed, I) = video.read()	

	
	if grabbed is False:
		break
	
	out = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

	out = cv2.inpaint(out,mask_inv,1,cv2.INPAINT_TELEA)
	
	bil = cv2.bilateralFilter(out, 7, 75, 75)
	d = cv2.filter2D(bil, ddepth = -1, kernel = kernelSharp)
	

	# CHANGE TO bil HERE
	E = cv2.dilate(d,smallMorph)
	erdcla3 = cv2.erode(E,largeMorph)
	dilcla3 = cv2.dilate(erdcla3,smallMorph)
	
	dencla = cv2.fastNlMeansDenoising(dilcla3,None,3,3,9)

	sharpenedblt = cv2.filter2D(dencla, ddepth = -1, kernel = kernelSharp)

	D = cv2.dilate(sharpenedblt,smallMorph)
	EE = cv2.erode(D,largeMorph)
	DD = cv2.dilate(EE,smallMorph)
	
	cla4 = cv2.GaussianBlur(DD,(3,3),0)
	denoise = cv2.fastNlMeansDenoising(cla4,None,8,7,21)

	output = cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR)
	
	if makingVideo is True:	
		# save to video
		newVideo.write(output)
	
	if makingVideo is False:
		imagename = directory + "newframe" + str(currentFrame) + ".png"
		cv2.imwrite(imagename, I)
	
	frameNum+=1
	
	key = cv2.waitKey(1)

	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

video.release()

if makingVideo is True:
	newVideo.release()

endTime = time.time()
print(endTime - startTime)
