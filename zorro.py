# C14399846 DIT COMPUTER SCIENCE
# (C) OLEG PETCOV

# import the necessary packages:
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

# Opening an image from a file:
#I = cv2.imread("colours.jpg")

# Opening an image using a File Open dialog:
#f = easygui.fileopenbox()
#I = cv2.imread(f)

# Getting the size of the image:
# size = np.shape(I)

#Capturing an image from a webcam:
k = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
kernel2 = np.ones((3,3),np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Tiling is important for correct contrast mappings
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

video = cv2.VideoCapture("Zorro.mp4")
(grabbed, I) = video.read()
#I = imutils.resize(I, width=640)

# Video Capture:
grabbed = True

# Using this to count frames
fr = 0;
font = cv2.FONT_HERSHEY_SIMPLEX

#height, width, channels = I.shape
height, width, _ = I.shape


#fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ZorroErodedNewOrderUnCropped.wmv',fourcc, 30.0, (854,480))

while (video.isOpened()):
	
	#fr += 1
	#frtxt = str(fr)
	
	#print (frtxt + "\n")
	
	# Hard sets the frame to frame 114
	# Nicer for testing purposes
	#camera.set(1,114)
	#camera.set(1,103)
	
	(grabbed, I) = video.read()
	
	
	# Width is now 638
	#cropped = I[0:480, 108:746]
	
	
	# THIS IS VERY IMPORTANT FOR PERFORMANCE
	# THE zorro.mp4 IS NORMALLY TREATED AS COLOUR IMAGE,
	# WHICH IS SLOWER TO LOAD*************
	I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	
	cla = clahe.apply(I)
	
	#cv2.imshow("cla", cla)
	#cla2 = clahe.apply(F)
	#I2 = cv2.fastNlMeansDenoisingColored(I,None,5,5,7,21) 
	
	
	
	
	# CHANGED THIS FROM DOING A 
	# BLUR, THEN DENOISE, THEN SHARPENING
	F2 = cv2.filter2D(cla, ddepth = -1, kernel = k)
	I2 = cv2.GaussianBlur(F2,(3,3),0)
	denoised = cv2.fastNlMeansDenoising(I2,None,5,7,21)
	
	
	#denoised = cv2.fastNlMeansDenoising(I,None,5,7,21)
	#I2 = cv2.GaussianBlur(denoised,(3,3),0)
	
	#F = cv2.filter2D(I, ddepth = -1, kernel = k)
	
	#F2 = cv2.filter2D(I2, ddepth = -1, kernel = k)
	
	#F2 = cv2.cvtColor(F2, cv2.COLOR_GRAY2BGR)
	
	
	
	# OUTPUT TEXT FRAMES ON IMAGE
	#cv2.putText(F2, frtxt ,(10,height-50), font, 4,(255,255,255),2,cv2.LINE_AA)
	
	#F3 = cv2.fastNlMeansDenoising(I2,None,10,7,21)
	
	#Fcla = cv2.filter2D(cla, ddepth = -1, kernel = k)

	#erosion = cv2.erode(F2,element)
	#dilation = cv2.dilate(F3,kernel2)
	#dilation2 = cv2.dilate(erosion,element)
	
	dilation = cv2.dilate(denoised,element)
	erosion = cv2.erode(dilation,element)
	
	
	#cv2.imshow("imageSharpBlur", F2)

	#cv2.imshow("image", I)
	#cv2.imshow("imageBlur", I2)
	
	#cv2.imshow("imageCLA", cla)
	
	#cv2.imshow("imageSharp", F)
	#cv2.imshow("imageSharpBlur", F2)
	#cv2.imshow("imageFastSharpBlur", F3)
	
	#cv2.imshow("imageSharpCLA", Fcla)	
	#cv2.imshow("erosion", erosion)
	#cv2.imshow("dilation", dilation)
	#cv2.imshow("both", dilation2)
	
	output = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
	
	out.write(output)
	
	key = cv2.waitKey(1)

	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

video.release()
out.release()

#cv2.waitKey(0)






















# Writing an image:
# cv2.imwrite("image.jpg",I)

# Showing an image on the screen (OpenCV):
#cv2.imshow("LMAO", I)
#key = cv2.waitKey(0)

# Showing an image on the screen (MatPlotLib):
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# plt.imshow(I) 
# plt.show() 

# Converting to different colour spaces:
#RGB = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
#YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
# G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

#cv2.imshow("LMAORGB", RGB)
#cv2.imshow("LMAOHSV", HSV)
#cv2.imshow("LMAOYUV", YUV)
#key = cv2.waitKey(0)

# Keeping a copy:
# Original = I.copy() 

# # Drawing a line:
# cv2.line(img = I, pt1 = (200,200), pt2 = (500,600), color = (255,255,255), thickness = 5) 

# # Drawing a circle:
# cv2.circle(img = I, center = (800,400), radius = 50, color = (0,0,255), thickness = -1)

# # Drawing a rectangle:
# cv2.rectangle(img = I, pt1 = (500,100), pt2 = (800,300), color = (255,0,255), thickness = 10)

# # Accessing a pixel's value:
# B = I[400,800,0]
# BGR = I[400,800]
# print B
# print BGR

# Setting a pixel's value:
# I[400,800,0] = 255
# cv2.imshow("image", I)
# key = cv2.waitKey(0)

# I[400,800] = (255,0,0)
# cv2.imshow("image", I)
# key = cv2.waitKey(0)

# Using the colon operator:
# I[390:410,790:810] = (255,0,0)
# cv2.imshow("image", I)
# key = cv2.waitKey(0)

# I[:,:,2] = 0
# cv2.imshow("image", I)
# key = cv2.waitKey(0)

# Capturing user input:
# def draw(event,x,y,flags,param): 
	# if event == cv2.EVENT_LBUTTONDOWN: 
		# cv2.circle(img = I, center = (x,y),radius = 5, color = (255,255,255), thickness = -1) 
		# cv2.imshow("image", I) 
			
# cv2.namedWindow("image") 
# cv2.setMouseCallback("image", draw) 
# cv2.imshow("image", I)
# key = cv2.waitKey(0)

# A handy way to use the waitkey....

# while True:
	# cv2.imshow("image", I)
	# key = cv2.waitKey(0)

	# # if the 'r' key is pressed, reset the image:
	# if key == ord("r"):
		# I = Original.copy()

	# # if the 'q' key is pressed, quit:
	# elif key == ord("q"):
		# break

