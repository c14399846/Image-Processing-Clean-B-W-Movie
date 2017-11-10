# C14399846 DIT COMPUTER SCIENCE
# (C) OLEG PETCOV

# import the necessary packages:
import math
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
from collections import deque
import time

#from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim
import peakutils

#from peakdetect import peakdetect #https://blog.ytotech.com/2015/11/01/findpeaks-in-python/

from scipy.signal import find_peaks_cwt

#Capturing an image from a webcam:
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)

kernelSharpTest = np.array( [[ 0, -1, 0], [ -1, 4, -1], [ 0, -1, 0]], dtype = float)

kernel2 = np.ones((5,5),np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Tiling is important for correct contrast mappings
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

video = cv2.VideoCapture("Zorro.mp4")

font = cv2.FONT_HERSHEY_SIMPLEX


###################################################
# NOTE: NOT FULLY MY CODE
# TOOK "INSPIRATION" FROM THIS CODE
# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
# https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
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

	array.append([framenum,m,s])


video.set(0,0)
(grabbed, I) = video.read()	
I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)


ret, mask = cv2.threshold(I, thresh = 20, maxval = 255, type = cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

grabbed = True

Image_data = []


start = time.time()


while (grabbed):

	(grabbed, I) = video.read()	
	

	if grabbed is not False:
		dst = cv2.inpaint(I,mask_inv,1,cv2.INPAINT_TELEA)
		gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

		Image_data.append(gray)


print("DONE ARRAY\n")

i = 0
framecount = len(Image_data) - 1
print(str(framecount))


frametest = i

video.release()



array = []
	
	
	
while (i < framecount):

	
	#print (str(i))
	
	# frame1
	fr = frametest
	frtxt = str(fr)
	
	#frame2
	fr2 = fr+1
	fr2txt = str(fr2)

	out = Image_data[i]

	out2 = Image_data[i+1]
	

	cv2.putText(out,frtxt,(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
	

	cv2.putText(out2,fr2txt,(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
	
	compare_images(out,out2, frametest)	
	
	i += 1
	frametest += 1
	
	key = cv2.waitKey(1)

	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

print("FINISHED SSIM\n")

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
	
	
	'''
	print (f)
	print (m)
	print (s)
	'''
	# Gets difference in second row, the mean values
	#diffArr = numpy.diff(a[:,1])
	#a[:,1]

	
#peaks = peakdetect(arr, lookahead=100)
#indexes = find_peaks_cwt(arr, np.arange(1, max(arr))) # Better for positive peaks
#indexes2 = peakutils.indexes(arr, thres=0.02/max(arr), min_dist=100)	

end = time.time()
print(end - start)

plt.plot(arrS)

plt.show()

plt.plot(arrM)

plt.show()

