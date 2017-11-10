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


#Capturing an image from a webcam:
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
kernelVerySharp = np.array( [[ -1, -1, -1], [ -1, 9, -1], [ -1, -1, -1]], dtype = float)

kernelSharpTest = np.array( [[ 0, -1, 0], [ -1, 4, -1], [ 0, -1, 0]], dtype = float)

kernel2 = np.ones((5,5),np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Tiling is important for correct contrast mappings
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))

video = cv2.VideoCapture("../../Zorro.mp4")
#(grabbed, I) = video.read()
#I = imutils.resize(I, width=640)

# Video Capture:
#grabbed = True

# Using this to count frames
#fr = 0;
font = cv2.FONT_HERSHEY_SIMPLEX

#height, width, channels = I.shape
#height, width, _ = I.shape



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
	
	'''
	print ("WAITING")
	
	
	print ("\n\n\n")
	print ("FRAMENUM:" + str(framenum) + "\n")
	
	print ("Mean:")
	print (m)
	print ("\n")
	print ("SSIM:")
	print (s)
	print ("\n")
	print("******************\n")
	'''
	
	
	#cv2.waitKey(0)
	
	'''
	if ( (m < ) and (s < )):
		append to a queue
	


	'''
	'''
	else:
		cut off that clip, 
		create another queue,
		start filling items
		

	'''


	
	
###################################
# Inpainting
# Uses first frame
#video.set(0,0)
#(grabbed, I) = video.read()	
#I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

# Thresh is better clarity here, rather than 0 
#ret, mask = cv2.threshold(I, thresh = 20, maxval = 255, type = cv2.THRESH_BINARY_INV)

#mask_inv = cv2.bitwise_not(mask)


#cv2.imshow("mask",mask)
#cv2.imshow("maskI",I)
#cv2.imshow("maskinv",mask_inv)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#inpaintTEMP = cv2.VideoWriter('ZorroTEMP.wmv',fourcc, 24.0, (854,480))

grabbed = True

Image_data = []
#Image_queue = deque()

start = time.time()

#video.set(0,0)
while (grabbed):

	#fr+=1
	#strfr = str(fr)
	#print (strfr + "\n")

	(grabbed, I) = video.read()	
	
	#print (grabbed)
	
	if grabbed is not False:
		#dst = cv2.inpaint(I,mask_inv,1,cv2.INPAINT_TELEA)
		gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		#dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
		
		#output = cv2.cvtColor(cla4, cv2.COLOR_GRAY2BGR)
		#Image_queue.append(gray)
		Image_data.append(gray)
		#inpaintTEMP.write(dst)

print("DONE ARRAY\n")
#end = time.time()
#print(end - start)

#video.release()
#video = cv2.VideoCapture("ZorroTEMP.wmv")
#(grabbed, I) = video.read()	
#cv2.waitKey(0)

#fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#newVideo = cv2.VideoWriter('ZorroNOMASK.avi',fourcc, 24.0, (854,480))

'''
list(im.getdata())

for pixel in iter(im.getdata()):
    print pixel
'''
#Image_data
i = 0
framecount = len(Image_data) - 1
print(str(framecount))

# Original framething
#video.set(0,0)

frametest = i
#video.set(0,frametest)
video.release()



array = []


'''
#while (Image_queue):
#while (grabbed):
	#i += 1
	#fr += 1
	#fr = frametest
	#frtxt = str(fr)
	
	#print (frtxt + "\n")
	
	# Hard sets the frame to frame 114
	# Nicer for testing purposes
	#video.set(1,114)
	#$video.set(1,126)
	
	
	if frametest > 140:
		#frametest = 103
		#video.set(1,frametest)
		break
	
'''	
	
	
	
while (i < framecount):

	
	#print (str(i))
	
	# frame1
	fr = frametest
	frtxt = str(fr)
	
	#frame2
	fr2 = fr+1
	fr2txt = str(fr2)
	
	#(grabbed, I) = video.read()
	out = Image_data[i]
	
	#if grabbed is False:
	#	break
	
	#frametest += 1
	
	#(grabbed2, I2) = video.read()	
	out2 = Image_data[i+1]
	
	#if grabbed2 is False:
	#	break
	
	#out = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	cv2.putText(out,frtxt,(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
	
	#out2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	cv2.putText(out2,fr2txt,(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
	
	compare_images(out,out2, frametest)
	
	#cv2.imshow("I", out)
	#cv2.imshow("I2", out2)
	
	#cv2.waitKey(0)

	
	'''
	
	out = cv2.inpaint(out,mask_inv,1,cv2.INPAINT_TELEA)
	
	bil = cv2.bilateralFilter(out, 7, 75, 75)
	d = cv2.filter2D(bil, ddepth = -1, kernel = kernelSharp)


	# CHANGE TO bil HERE
	E = cv2.dilate(d,element2)
	erdcla3 = cv2.erode(E,element)
	dilcla3 = cv2.dilate(erdcla3,element2)

	#Original
	dencla = cv2.fastNlMeansDenoising(dilcla3,None,3,3,9)

	sharpenedblt = cv2.filter2D(dencla, ddepth = -1, kernel = kernelSharp)

	D = cv2.dilate(sharpenedblt,element2)
	EE = cv2.erode(D,element)
	DD = cv2.dilate(EE,element2)

	
	denoise = cv2.fastNlMeansDenoising(DD,None,8,7,21)
	cla4 = cv2.GaussianBlur(denoise,(3,3),0)

	output = cv2.cvtColor(cla4, cv2.COLOR_GRAY2BGR)

	newVideo.write(output)
	'''
	
	i += 1
	frametest += 1
	
	key = cv2.waitKey(1)

	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

print("FINISHED SSIM\n")
#video.release()
#newVideo.release()

#diffArr = np.diff(array[:,1])

arr = []

for x in range(framecount):
	f = array[x][0]
	m = array[x][1]
	s = array[x][2]
	#m = diffArr[x][1]
	#print("\nDiff:" + str(diffArr[x]))
	
	arr.append(m)
	
	
	'''
	print (f)
	print (m)
	print (s)
	'''
	# Gets difference in second row, the mean values
	#diffArr = numpy.diff(a[:,1])
	#a[:,1]

plt.plot(arr)
plt.ylabel('some numbers')

plt.axis([0, (framecount/24), 0, max(arr)])

plt.show()
	
end = time.time()
print(end - start)

cv2.waitKey(0)






















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

