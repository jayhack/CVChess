#!/opt/local/bin/python
import os, sys
import numpy as np
import cv, cv2
import CVAnalysis
from Board import Board
from util import print_message, print_status


#==========[ Global Data	]==========
corner_keypoints 		= []
corner_image_points 	= []
corner_board_points 	= []
corner_sift_desc		= []


def get_closest_keypoint (image_point, keypoints):
	"""
		Function: get_closest_keypoint
		------------------------------
		given a point in the image, this returns the closest 
		corresponding point in keypoints
	"""
	def euc_dist (ip, kp):
		return (ip[0] - kp.pt[0])**2 + (ip[1] - kp.pt[1])**2

	distances = [euc_dist(image_point, kp) for kp in keypoints]
	return keypoints.pop (np.argmin(distances))


def on_mouse(event, x, y, flags, param):
	"""
		Function: on_mouse
		------------------
		callback for clicking the image;
		has you input the board coords of the point you just clicked
		stores results in corner_board_points and corner_image_points
	"""
	#=====[ Step 1: only accept button down events	]=====
	if not event == cv2.EVENT_LBUTTONDOWN:
		return

	#=====[ Step 2: get corresponding points	]=====
	print "=====[ Enter board coordinates: ]====="
	# board_x = int(raw_input ('>>> x: '))
	# board_y = int(raw_input ('>>> y: '))
	# board_point = (board_x, board_y)
	keypoint = get_closest_keypoint ((x, y), param)
	image_point = keypoint.pt
	print "Stored as: "
	# print "	- board_point: ", board_point
	print "	- image_point: ", image_point
	# corner_board_points.append (board_point)
	corner_image_points.append (image_point)
	corner_keypoints.append (keypoint)
	print_message ("ESC to exit")




if __name__ == "__main__":

	#==========[ Step 1: sanitize input 	]==========
	if not len(sys.argv) == 2:
		filename = '../data/micahboard/1.jpg'
	else:
		filename 	= sys.argv[1]
	image_name 	= os.path.split (filename)[1]
	if not os.path.isfile (filename):
		raise StandardError ("Couldn't find the file you passed in: " + image_name)


	#==========[ Step 2: get/convert image	]==========
	image = cv2.imread(filename)
	gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)


	#==========[ Step 3: get corners	]==========
	harris_corners = CVAnalysis.get_harris_corners (image)
	disp_image = cv2.drawKeypoints (image, harris_corners, color=(0, 0, 255))


	#==========[ Step 4: draw image	]==========
	cv2.namedWindow ('DISPLAY')
	cv.SetMouseCallback ('DISPLAY', on_mouse, param=harris_corners)


	#==========[ Step 4: have user mark keypoints ]==========
	while True:

		disp_image = cv2.drawKeypoints (image, harris_corners, color=(0, 0, 255))
		disp_image = cv2.drawKeypoints (disp_image, corner_keypoints, color=(255, 0, 0))

		cv2.imshow ('DISPLAY', disp_image)
		key = cv2.waitKey(30)
		if key == 27:
			break

	
	#==========[ Step 5: get descriptors for each corner point	]==========
	print_status ('MarkImage', 'getting SIFT descriptors for clicked corners')
	kpts, desc = CVAnalysis.get_sift_descriptors (image, corner_keypoints)
	corner_sift_desc = desc


	#==========[ Step 6: construct BoardImage	]==========
	print_status ('MarkImage', 'constructing BoardImage object')	
	board = Board 	(	
						image=image, 
						name=image_name, 
						board_points=corner_board_points, 
						image_points=corner_image_points,
						sift_desc = corner_sift_desc
					)