import numpy as np 
import cv2 

if __name__ == "__main__":

	#=====[ Step 1: load in images	]=====
	img1_raw = cv2.imread ('../data/game1/01.jpg')
	img2_raw = cv2.imread ('../data/game1/02.jpg')

	#=====[ Step 2: convert to grayscale	]====
	img1_gray = cv2.cvtColor (img1_raw, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor (img2_raw, cv2.COLOR_BGR2GRAY)	

	#=====[ Step 3: downsample to 1/16th the original size	]=====
	img1 = cv2.pyrDown (cv2.pyrDown (img1_gray))
	img2 = cv2.pyrDown (cv2.pyrDown (img2_gray))	

	#=====[ Step 4: compute difference ]=====
	difference = img1 - img2

	#=====[ Step 6: erode, dilate	]=====
	kernel 	= np.ones ((4, 4), np.uint8)
	erosion = cv2.erode (difference, kernel)
	dilation = cv2.dilate (erosion, kernel)
	img1_large = cv2.pyrUp(cv2.pyrUp(dilation))

	cv2.namedWindow ('test')
	cv2.imshow ('test', difference)
	



