#

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import threading
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())
 
 
'''
#Capturing an image from a webcam:
kernelSharp = np.array( [[ 0, -1, 0], [ -1, 5, -1], [ 0, -1, 0]], dtype = float)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
element2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Tiling is important for correct contrast mappings
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ZorroRubberChickenFast.wmv',fourcc, 30.0, (854,480)) 
 '''
# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
 
# start the FPS timer
fps = FPS().start()

fr = 0;
font = cv2.FONT_HERSHEY_SIMPLEX



# loop over frames from the video file stream
while fvs.more():
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale (while still retaining 3
	# channels)
	frame = fvs.read()	
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	frame2 = fvs.read()	
	
	num_threads = threading.activeCount()
	
	print (num_threads)
	'''
	fr += 1
	frtxt = str(fr)
	
	print (frtxt + "\n")
	
	
	E = cv2.dilate(frame,element2)
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
	
	'''
	
	
	
	
	
	frame = np.dstack([frame, frame, frame])
	
	# display the size of the queue on the frame
	cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
	
	
	
	
	
	
	# show the frame and update the FPS counter
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)
	fps.update()
	
	
	
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()