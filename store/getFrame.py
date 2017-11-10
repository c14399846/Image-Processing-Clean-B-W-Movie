from skimage.measure import structural_similarity as ssim

from threading import Thread
import sys

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

framenum = 664

#video = cv2.VideoCapture("../../Zorro.mp4")
video = cv2.VideoCapture("ZorroRubberChicken.wmv")

#(grabbed, I) = video.read()

#out = cv2.VideoWriter('ZorroRubberChicken5.wmv',fourcc, 30.0, (854,480))
#key = 'e'

#J = 34

#while (key is not "q"):

'''
frtxt = str(J)

print (frtxt + "\n")

if J > 470:
	J = 420

video.set(1,J)

J+=1
'''

video.set(1,framenum)
(grabbed, I) = video.read()

blur = cv2.Laplacian(I, cv2.CV_64F).var()

cv2.putText(I, "{}: {:.2f}".format("Blur", blur), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

#cv2.imshow("J", I)

imagename = "./frames/newframe" + str(framenum) + ".png"

cv2.imwrite(imagename, I)

#cv2.waitKey(0)