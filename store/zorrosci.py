# C14399846 DIT COMPUTER SCIENCE
# (C) OLEG PETCOV

# import the necessary packages:
from skimage.measure import structural_similarity as ssim

from threading import Thread
import sys

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


#############################################################################################

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()


# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("images/jp_gates_original.png")
contrast = cv2.imread("images/jp_gates_contrast.png")
shopped = cv2.imread("images/jp_gates_photoshopped.png")
 
# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
 
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
 
# show the figure
plt.show()
 
# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, contrast, "Original vs. Contrast")
compare_images(original, shopped, "Original vs. Photoshopped")

##########################################################################################
		
		
#Capturing an image from a webcam:
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Tiling is important for correct contrast mappings
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))

video = cv2.VideoCapture("../../Zorro.mp4")
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
out = cv2.VideoWriter('ZorroRubberChicken5.wmv',fourcc, 30.0, (854,480))

while (video.isOpened()):

	
	(grabbed, I) = video.read()

	I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	

	
	E = cv2.dilate(I,element2)
	erdcla3 = cv2.erode(E,element)
	dilcla3 = cv2.dilate(erdcla3,element2)
	
	dencla = cv2.fastNlMeansDenoising(dilcla3,None,3,3,9)

	sharpenedblt = cv2.filter2D(dencla, ddepth = -1, kernel = kernelSharp)
	
	D = cv2.dilate(sharpenedblt,element2)
	EE = cv2.erode(D,element)
	DD = cv2.dilate(EE,element2)
	
	cla4 = cv2.fastNlMeansDenoising(DD,None,8,7,21)
	cla4 = cv2.GaussianBlur(cla4,(3,3),0)


	output = cv2.cvtColor(cla4, cv2.COLOR_GRAY2BGR)
	
	out.write(output)
	
	key = cv2.waitKey(1)

	#if the 'q' key is pressed, quit:
	if key == ord("q"):
		break

video.release()
out.release()

#cv2.waitKey(0)