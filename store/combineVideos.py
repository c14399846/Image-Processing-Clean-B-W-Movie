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

vidName = "../../Zorro.mp4"
vidName2 = "./cleaned/ZorroClean.avi"

video = cv2.VideoCapture(vidName)
video2 = cv2.VideoCapture(vidName2)

(grabbed, I) = video.read()
(grabbed2, I2) = video2.read()
#I = imutils.resize(I, width=640)

# Video Capture:
grabbed = True


height, width, _ = I.shape
height2, width2, _2 = I.shape

if ( (height != height2) or (width != width2)):
	print ("NOT THE SAME RESOLUTIONS\n")
	exit()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('comparedVideo.wmv',fourcc, 24.0, (width*2,height*2))

fr = 0;
font = cv2.FONT_HERSHEY_SIMPLEX

video.set(0,0)
video2.set(0,0)

while (video.isOpened()):

	(grabbed, I) = video.read()
	(grabbed2, I2) = video2.read()
	
	if I is not None:
		I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
		
		# Put fileNames in the corner of the videos
		cv2.putText(I,vidName,(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
		cv2.putText(I2,vidName2,(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
		
		# Adds images together
		vis = np.concatenate((I, I2), axis=0)
		
		output = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
		
		output = cv2.resize(output, (1708,960))

		out.write(output)

		
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