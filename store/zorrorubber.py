# C14399846 DIT COMPUTER SCIENCE
# (C) OLEG PETCOV

# import the necessary packages:
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


#Capturing an image from a webcam:
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
kernel2 = np.ones((5,5),np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Tiling is important for correct contrast mappings
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Change to your zorro file location
video = cv2.VideoCapture("../../Zorro.mp4")
(grabbed, I) = video.read()

# Video Capture:
grabbed = True

# Using this to count frames
#fr = 0;
#font = cv2.FONT_HERSHEY_SIMPLEX

height, width, _ = I.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ZorroRubberChicken2.wmv',fourcc, 30.0, (854,480))

while (video.isOpened()):
	
	#fr += 1
	#frtxt = str(fr)
	
	#print (frtxt + "\n")
	
	# Hard sets the frame to frame 114
	# Nicer for testing purposes
	#video.set(1,103)
	
	(grabbed, I) = video.read()
	
	# convert to gray
	I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	
	# get rid of holes / smooths out the image
	Dil = cv2.dilate(I,element2)
	Ero = cv2.erode(Dil,element)
	Dil2 = cv2.dilate(Ero,element2)
	
	
	# remove any noise, 
	# then apply a sharpening filter to pop detials, I think
	dencla = cv2.fastNlMeansDenoising(Dil2,None,3,3,9)
	sharpen = cv2.filter2D(dencla, ddepth = -1, kernel = kernelSharp)

	# Smoothing sharpen artifacts
	sharpDil = cv2.dilate(sharpen,element2)
	sharpEro = cv2.erode(sharpDil,element)
	sharpDil2 = cv2.dilate(sharpEro,element2)
	
	# Blur out remaining sharpened artifacts + morphs.
	# Apply a final denoise for clarity.
	#
	# This will remove quite a bit of detail, 
	# But it is nicer to look at now
	blur = cv2.GaussianBlur(sharpDil2,(3,3),0)
	final = cv2.fastNlMeansDenoising(blur,None,8,7,21)
	
	# Revert back to BGR for video saving
	output = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
	
	out.write(output)
	
	key = cv2.waitKey(1)

	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

video.release()
out.release()

#cv2.waitKey(0)