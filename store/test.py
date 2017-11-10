# This joins two video files together in a vertical stack.
# Useful for comparing videos side by side.

# Can pause and go back easily, 
# don't have to re-run the creation script


# import the necessary packages:
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

video = cv2.VideoCapture("ZorroVSharpContrastAfter.wmv")
video2 = cv2.VideoCapture("ZorroVSharpContrastBefore.wmv")

(grabbed, I) = video.read()
#I = imutils.resize(I, width=640)

# Video Capture:
grabbed = True

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ZCompContrast.wmv',fourcc, 30.0, (1708,960))

fr = 0;
font = cv2.FONT_HERSHEY_SIMPLEX

while (video.isOpened()):

	(grabbed, I) = video.read()
	(grabbed2, I2) = video2.read()
	
	if I is not None:
		I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
		
		'''
		h1, w1 = I.shape[:2]
		h2, w2 = I2.shape[:2]
		nWidth = w1+w2
		nHeight = max(h1, h2)
		hdif = (h1-h2)/2
		newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
		newimg[hdif:hdif+h2, :w2] = I
		newimg[:h1, w2:w1+w2] = img
		
		both = np.hstack((I,I2))
		'''
		#imgBoth = np.dstack((I,I2))
		#imgBoth.shape      # prints (480,640,2)
		#print (imgBoth.shape )
		
		vis = np.concatenate((I, I2), axis=0)
		output = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
		
		output = cv2.resize(output, (1708,960))
		
		#height, width, _ = output.shape
		
		#height = str(height)
		#width = str(width)
	
		#print ("HL:" + height + " W:" + width + "\n")	
		
		#cv2.imshow("I2",output)
		#cv2.waitKey(0)
		
		out.write(output)
		#cv2.imshow("I2",newimg)
		
	else:
		break
	
	'''
	fr += 1
	frtxt = str(fr)
	
	print (frtxt + "\n")
	'''
	
	#cv2.imshow("I",I)
	#cv2.imshow("I2",I2)
	#cv2.waitKey(0)
	
	
	
	key = cv2.waitKey(1)
	if key == ord("q"):
		break
	
video.release()
video2.release()
out.release()