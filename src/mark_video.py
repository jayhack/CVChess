from copy import deepcopy
import os
import cv2
import numpy as np
import sklearn
import pickle
import CVAnalysis
from Board import Board
from util import *


def preprocess_frame (frame):
	return frame [(frame.shape[0]/2):, :]

def get_next_frame (vc):
	rval, frame = vc.read ()
	if not rval:
		raise Exception("Couldn't read the frame correctly for some reason...")
	return preprocess_frame (frame)


if __name__ == '__main__':

	#=====[ Step 1: setup videocapture	]=====
	video_filename = '../data/videos/3.mov'
	vc = cv2.VideoCapture (video_filename)


	#=====[ Step 2: initialize the board	]=====
	# corner_classifier_filename = '../data/classifiers/corner_classifier.clf'
	corner_classifier_filename = '../data/videos/3.clf'
	corner_classifier = pickle.load (open(corner_classifier_filename, 'r'))	#more data
	board = Board(corner_classifier=corner_classifier)


	#=====[ Step 3: initialize board plane ]=====
	frame_empty = get_next_frame (vc)
	board.initialize_board_plane (frame_empty)	
	####[ DEBUG: verify BIH is correct	]#####
	# cv2.imwrite ('../data/videos/3_ic.jpg', frame_empty)
	# frame_ic = board.draw_vertices(frame_empty)
	# cv2.imshow ('BIH MARKED', frame_empty)
	# key = 0
	# while not key in [27, ord('Q'), ord('q')]: 
	# 	key = cv2.waitKey (30)
	# cv2.destroyAllWindows ()


	#=====[ Step 4: initialize game 	]=====
	num_frames = 1
	while (num_frames < 507):
		frame_ic = get_next_frame (vc)
		num_frames += 1
	board.initialize_game (frame_ic)
	####[ DEBUG: display board with initial config	]#####
	# cv2.imshow ('INITIAL CONFIG', frame_ic)
	# key = 0
	# while not key in [27, ord('Q'), ord('q')]: 
	# 	key = cv2.waitKey (30)
	# cv2.destroyAllWindows ()


	# add_frames = [470, 516, 550, 589, 648, 709, 819, 878, 932] #1.mov
	add_frames = 		[	531, 566, 652, 689, 744, 801, 		#3.mov
							1005, 1069, 1501, 1558, 1610, 
							1642, 1673, 1695, 1922, 1963, 
							2050, 2211, 2359, 2768, 2812, 
							2972, 3219, 3290, 3484, 3576, 
							3781, 4000, 4502, 4933, 5377, 
							5645]
	while True:

		#=====[ Step 1: get/preprocess frame	]=====
		frame = get_next_frame (vc)

		#=====[ Step 2: display and wait indefinitely on key	]=====
		print "frame #: ", num_frames
		if num_frames > 531:
			disp_frame = board.draw_last_move (deepcopy(frame))
		else:
			disp_frame = frame
		cv2.imshow ('FRAME', disp_frame)

		#=====[ Case: space bar -> add frame	]=====
		if num_frames in add_frames:
			print "===[ ADDING FRAME #" + str(num_frames) + " ]==="
			# cv2.imwrite ('../data/videos/' + str(num_frames) +'.jpg', frame)
			board.add_move (frame)
			board.display_movement_heatmaps ()			


		######[ DEBUG: for marking pieces	]#####
		key = cv2.waitKey (5)
		# while key != ord('d'):
			# key = cv2.waitKey (5)

		#=====[ Case: exit -> escape	]=====
		if key in [27, ord('Q'), ord('q')]: 
			break

		num_frames += 1




