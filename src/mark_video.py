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

if __name__ == '__main__':

	#=====[ Step 1: setup videocapture	]=====
	video_filename = '../data/videos/1.mov'
	vc = cv2.VideoCapture (video_filename)

	#=====[ Step 2: initialize the board	]=====
	corner_classifier_filename = '../data/classifiers/corner_classifier.clf'
	corner_classifier = pickle.load (open(corner_classifier_filename, 'r'))	#more data
	board = Board(corner_classifier=corner_classifier)

	#=====[ Step 3: get initial configuration	]=====
	success, frame_ic = vc.read ()
	assert success
	print frame_ic.shape
	frame_ic = preprocess_frame(frame_ic)
	board.add_frame (frame_ic)
	
	#####[ DEBUG: show initial config	]#####
	frame_ic = board.draw_vertices(frame_ic)
	cv2.imshow ('test', frame_ic)
	key = 0
	while not key in [27, ord('Q'), ord('q')]: 
		key = cv2.waitKey (30)



	# num_frames = 0
	# while (True):

	# 	#=====[ Step 1: get/preprocess frame	]=====
	# 	rval, frame = webcam.read ()
	# 	half_height = frame.shape[0]/2
	# 	frame = frame[half_height:, :]

	# 	key = cv2.waitKey(20)
	# 	#=====[ Case: space bar -> add frame	]=====
	# 	if key == ord(' '):
	# 		print "===[ SPACE PRESSED ]==="
	# 		print "num_frames: ", num_frames
	# 		board.add_frame (frame)

	# 		#####[ DEMO: show BIH ]#####
	# 		if num_frames == 0:
	# 			board.draw_squares (frame)
	# 			key2 = 0
	# 			while key2 != 27:
	# 				key2 = cv2.waitKey (30)
	# 			cv2.destroyWindow ("board.draw_squares")
	# 		#####[ END DEMO: show BIH ]#####
	# 		if num_frames >= 1:
	# 			board.get_occlusion_changes ()

	# 		if num_frames >= 2:
	# 			board.update_game ()
	# 			board.display_occlusion_changes ()				


	# 		num_frames += 1

	# 	#=====[ Case: exit -> escape	]=====
	# 	elif key in [27, ord('Q'), ord('q')]: 
	# 		break

	# 	if num_frames >= 3:
	# 		frame = board.draw_last_move (frame)
	# 	cv2.imshow ("LIVE FEED", frame)





	#=====[ ITERATE THROUGH ALL FRAMES	]=====
	# for index, frame in enumerate(frames[1:]):
	# 	cv2.imwrite ('posterpics/move2_raw.png', frame)
	# 	print_header ("ADDING FRAME: " + str(index))

	# 	#=====[ Step 3: add frame ]=====
	# 	board.add_frame (frame)

	# 	#=====[ Step 4: get occlusion change	]=====
	# 	board.get_occlusion_changes ()
	# 	if index >= 1:
	# 		board.update_game ()
	# 		# board.display_occlusion_changes ()
	# 		frame = board.draw_square_an (('F', 6), board.current_frame, color=(0, 255, 0))
	# 		cv2.imshow ('test', frame)
	# 		cv2.imwrite ('move2_detected.png', frame)



			# key = 0
			# while key != 27:
				# key = cv2.waitKey (30)

