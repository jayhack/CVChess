import cv2
import numpy as np
from util import *


if __name__ == "__main__":

	#=====[ Step 1: read in image	]=====
	im_raw = cv2.imread ('../data/ic1/4.jpg')
	im_size = im_raw.shape[0]
	im_raw = im_raw[int(im_size/2):, :]

	#=====[ Step 2: blur with 5x5 kernel	]=====
	im_blur = cv2.blur (im_raw, (5, 5))
	kernel 	= np.ones ((4, 4), np.uint8)
	
	dilation = cv2.dilate (im_blur, kernel)
	erosion = cv2.erode (dilation, kernel)

	# erosion = cv2.erode (im_blur, kernel)
	# dilation = cv2.dilate (erosion, kernel)


	#=====[ Step 5: apply canny edge detection	]=====
	im_canny = cv2.Canny (dilation, 30, 80)
	# canny_blur = cv2.blur (im_canny, (4,4))

	# kernel 	= np.ones ((4, 4), np.uint8)
	# erosion = cv2.erode (canny_blur, kernel)
	# dilation = cv2.dilate (erosion, kernel)

	#=====[ Step 5: apply hough transform	]=====
	# houghLines = cv2.HoughLines (im_canny, 3, np.pi/180, 400)[0]


	# im_blur = draw_lines_rho_theta (im_raw, houghLines)
	cv2.namedWindow ('lines')
	cv2.imshow ('lines', dilation)
	key = 0
	while key != 27:
		key = cv2.waitKey (30)


