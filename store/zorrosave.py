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


'''
HOW TO USE:
	The user will be able to choose a video file from directory
	
	A folder 'cleaned' is created if not exists already, from launch place of script.
	
	Bools:
		makingVideo
			Set to 'True' if making a video file
			Set to 'False' if making individual frames

		forceFrame
			Set to 'True' if starting from a specific frame,
			and ending after 'frameStop' amount of frames
		
			Set to 'False' if creating from frame 0,
			and computing the whole video
'''

'''
THE PROGRAM:
	
	The purpose of this python script is to improve the quality of old film-based clips.
	
	It will employ a range of functions, to clean up a video frame-by-frame.

	
	###############################################################################################################
	1)
	
	Initially the user will choose a video file from directory.

	
	###############################################################################################################
	2)
	
	The first frame will be used to take out any watermarks within the image,
	as long as the first frame is a largely black background image, with the watermark overlayed onto it.
	
	This is done through 'inpaint()' and an inverse mask.
	
	Reset video to frame 0 again

	
	###############################################################################################################
	3)
	
	The whole video is parsed to get rid of the watermark,
	and converted to gray.
	
	This is useful for a few reasons. 
	
	The main one is for processing using OpenCV functions,
	which require a gray / 2d channeled image.
	
	The other reason is performance.
	If you attempt to run some functions (like fastNLDenoise),
	You will hit HEAVY bottlenecks because the image is passed as a BGR, multi-channeled image.
	
	This is far harder to compute when using the fastNLDenoise for gray images.
	
	The computed image is stored in an array 'Image_data'.
	
	
	###############################################################################################################
	4)
	
	A folder 'cleaned' is created, if it doesn't already exist.
	This is used to store processed images and video files.
	
	
	The situational Bools are set here:
	
		If you want to process video, set Bool 'makingVideo' to True
		If you want to process frame-by-frame images, set 'MakingVideo' to False
	
		If you'd like to process only specific frames, e.g 105 to 150, set Bool 'forceFrame' to True
		Else, set it to False in order to process the entire video
	
	frame details are initialised:
		- totalFrames: length of the array 'Image_data'
		- startFrame: starting at this frame, default is 0, for beginnning frame
		- stopFrame: stop running after this many frames
		- endFrame: last frame, (totalFrames - 1)
		- frameNum: counting iterations of frames

		
	###############################################################################################################
	5)
	
														ALGO
	
	
	outputs percentage as you run the program.
	The percentage code is not mine, highlighted in the code itself.
	
	
	An image is grabbed from 'Image_data'
	
	
	###############################################################################################################
	i)
	
	Attributes: [bilat, sharpenBil]
	
	
	It is passed into a bilateral filter and a custom sharpen filter.
	
	This is done to preserve edges and details,
	and to not push any actors too heavily into the foreground,
	as can be seen in the walls if the sharpen is too heavy.
	
	Sharpen helps to 'pop' details after bilateral is applied.
	
	
	###############################################################################################################
	ii)
	
	Attributes: [dilated,eroded,dilatedFinal]
	
	
	A dilate, erode, dilate.
	
	Used to smooth out details after the sharpen,
	While (lightly) filling in some of the smaller blotches.
	
	They make use of a 5x5 Matrix kernel, and a 3x3 Matrix kernel.
	
	dilates use 3x3 [smallMorph], erode uses [largeMorph]
	
	
	###############################################################################################################
	iii)
	
	Attributes: [fastDen, sharpNoise]
	
	
	Removes noise from bilateral blur, sharpen and morphs.
	Quick sharpen to preserve edges again.
	
	A very light denoise pass here,
	to preserve detail for lower functions.
	Also to improve performance.

	Re-Sharpen to preserve feeature erosion form smoothing

	
	###############################################################################################################
	iv)
	
	Attributes: [sharpDil,sharpEro,sharpDilFinal]
	
	
	Fills in more minor blotches,
	and helps to smooth wall and background surfaces
	
	Unfortunately does 'brighten' some whitepoints in frames.
	This is due to the double dilates, in both parts of the algo.
	
	A custom equalise would be nice here.
	Clahe and the regular equalise fucntion return a non-ideal image.
	
	
	###############################################################################################################
	v)
	
	Attributes: [gBlurred,denoisedGray]
	
	
	Final blur (Gaussian) and heaviest denoise is applied here.
	
	Edge preservation is less important here, as compared to initial image, 
	where more preservation is ideal, when you need to do many passes of smoothing, etc. 
	
	Detail is lost in denser areas.
	Look at cloak / blanket in fight scene, and on stairs beside Zorro. 
	The top of them is lightly melded together.
	
	Although, walls and background are a lot cleaner now.
	The candles are a bit clearer too.
	
	
	###############################################################################################################
	vi)
	
	Attributes: [output,frameNum]
	
	Converts back to BGR for correct video format.
	
	Adds frame to videofile / creates imagefile in specified directory
	
	[frameNum] used to iterate through the array
	
	
	###############################################################################################################
	vii)
	
	Attributes: [time]
	
	Outputs the time to compute algo in seconds
	
	
	###############################################################################################################
'''


'''
A nifty tool to use for easier comparison is 'combineVideos.py'
	This script will combine two video files into one stacked video.
	
	Input the video filenames inside the sscript, no arguments used.
	
	Unfortunately they are a bit 'squashed', because of the stacking,
	But it's better than nothing.
	
	It's fairly fast to run too.
	
	Note: **** They must be the same resolution ****
	
'''

# 1)
# Opening an image using a File Open dialog:
videofile = easygui.fileopenbox()

# NOT USED, IT MESSES UP WALL SHADOWS
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Sharpening kernels
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)

# Morph kernels
largeMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
smallMorph = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Capture video, resolution, FPS
video = cv2.VideoCapture(videofile)
(grabbed, I) = video.read()
height, width, _ = I.shape
fps = 24.0

# start timer for execution
startTime = time.time()	



# 2)
# Inpainting mask
video.set(0,0)
(grabbed, I) = video.read()	
I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(I, thresh = 10, maxval = 255, type = cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
video.set(0,0)



# 3)
# Remove watermark
print("STARTED REMOVING WATERMARK\n")

Image_data = []


while (video.isOpened()):

	(grabbed, I) = video.read()	

	if grabbed is not False:
		inpaint = cv2.inpaint(I,mask_inv,1,cv2.INPAINT_TELEA)
		gray = cv2.cvtColor(inpaint, cv2.COLOR_BGR2GRAY)
		Image_data.append(gray)
	else:
		break

print("FINISHED REMOVING WATERMARK\n\n")



# 4)
# Imagefile names and directory
directory = "./cleaned/"
imageExtension = ".png"
imagePrefix = "newframe"

if not os.path.exists(directory):
    os.makedirs(directory)

	
# Clean video
makingVideo = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
newvideo = cv2.VideoWriter()

if makingVideo is True:		
	newVideo = cv2.VideoWriter(directory + 'ZorroClean.avi',fourcc, fps, (width,height))


# Frames
forceFrame = False
totalFrames = len(Image_data)
startFrame = 0 
stopFrame = 0 
frameNum = 0 

if forceFrame is True:
	startFrame = 105 
	stopFrame = 15
	endFrame = stopFrame + startFrame
	frameNum = 0
else:
	startFrame = 0 
	stopFrame = totalFrames - 1
	endFrame = totalFrames - 1
	frameNum = 0


if makingVideo is True:
	print("STARTED PROCESSING VIDEO\n")
else:
	print("STARTED PROCESSING FRAMES\n")

if forceFrame is True:
	print("GETTING FRAMES FROM " + str(startFrame) + " TO " + str(endFrame) + "\n")
else:
	print("GETTING ALL FRAMES\n")
	


	
##################################################################################
# NOT FULLY MINE
#
# It shows percentage of progress as the program is executing
# Uses amount of frames in Image_data, compared to 'framepos'
#
# http://blog.montmere.com/2013/04/17/how-to-show-a-percent-complete-in-python/
##################################################################################
def percentage(fNum,fStop):
	
	sys.stdout.write('\r')
	sys.stdout.write('%.2f%% complete' % (fNum / fStop * 100))
	sys.stdout.flush()
	

# 5)
# processing algo
while (frameNum <= stopFrame):
	
	percentage(frameNum,stopFrame)
	
	currentFrame = startFrame + frameNum # current frame position
	
	if currentFrame > endFrame:
		break
	
	frame = Image_data[currentFrame]
	
	bilat = cv2.bilateralFilter(frame, 7, 75, 75)
	sharpenBil = cv2.filter2D(bilat, ddepth = -1, kernel = kernelSharp)
	
	
	dilated = cv2.dilate(sharpenBil,smallMorph)
	eroded = cv2.erode(dilated,largeMorph)
	dilatedFinal = cv2.dilate(eroded,smallMorph)

	
	fastDen = cv2.fastNlMeansDenoising(dilatedFinal,None,3,3,9)
	sharpNoise = cv2.filter2D(fastDen, ddepth = -1, kernel = kernelSharp)

	
	sharpDil = cv2.dilate(sharpNoise,smallMorph)
	sharpEro = cv2.erode(sharpDil,largeMorph)
	sharpDilFinal = cv2.dilate(sharpEro,smallMorph)

	
	gBlurred = cv2.GaussianBlur(sharpDilFinal,(3,3),0)
	denoisedGray = cv2.fastNlMeansDenoising(gBlurred,None,8,7,21)


	output = cv2.cvtColor(denoisedGray, cv2.COLOR_GRAY2BGR)
	
	
	if makingVideo is True:	
		newVideo.write(output)
	elif makingVideo is False:
		imagename = directory + imagePrefix + str(currentFrame+1) + imageExtension
		cv2.imwrite(imagename, output)
	
	frameNum += 1
	
	key = cv2.waitKey(1)

	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

		

video.release()

if makingVideo is True:
	newVideo.release()


endTime = time.time()
print ("\n")
time = str("%.2f" % round(endTime - startTime,2))
print(time + " Seconds\n")


cv2.destroyAllWindows()