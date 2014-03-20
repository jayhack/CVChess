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
	video_filename = '../data/videos/1.mov'
	vc = cv2.VideoCapture (video_filename)


	#=====[ Step 2: initialize the board	]=====
	corner_classifier_filename = '../data/classifiers/corner_classifier.clf'
	corner_classifier = pickle.load (open(corner_classifier_filename, 'r'))	#more data
	board = Board(corner_classifier=corner_classifier)


	#=====[ Step 3: initialize board plane ]=====
	frame_empty = get_next_frame (vc)
	board.initialize_board_plane (frame_empty)	
	# cv2.imwrite ('../data/videos/0.jpg', frame_ic)
	#####[ DEBUG: verify BIH is correct	]#####
	frame_ic = board.draw_vertices(frame_empty)
	cv2.imshow ('BIH MARKED', frame_empty)
	key = 0
	while not key in [27, ord('Q'), ord('q')]: 
		key = cv2.waitKey (30)
	cv2.destroyAllWindows ()


	#=====[ Step 4: initialize game 	]=====
	num_frames = 1
	while (num_frames < 420):
		frame_ic = get_next_frame (vc)
		num_frames += 1
	board.initialize_game (frame_ic)
	#####[ DEBUG: display board with initial config	]#####
	frame_ic = board.draw_vertices(frame_empty)
	cv2.imshow ('INITIAL CONFIG', frame_empty)
	key = 0
	while not key in [27, ord('Q'), ord('q')]: 
		key = cv2.waitKey (30)
	cv2.destroyAllWindows ()


	add_frames = [470, 516, 550, 589, 648, 709, 819, 878]
	while True:
		#=====[ Step 1: get/preprocess frame	]=====
		frame = get_next_frame (vc)

		#=====[ Step 2: display and wait indefinitely on key	]=====
		print "frame #: ", num_frames
		cv2.imshow ('FRAME', frame)

		#=====[ Case: space bar -> add frame	]=====
		if num_frames in add_frames:
			print "===[ ADDING FRAME #" + str(num_frames) + " ]==="
			board.add_frame (frame)
			board.get_occlusion_changes ()
			cv2.imwrite ('../data/videos/' + str(num_frames) +'.jpg', frame)
			if num_frames >= 470:
				board.update_game ()
				board.display_occlusion_changes ()			


		#=====[ Case: exit -> escape	]=====
		key = cv2.waitKey (5)
		if key in [27, ord('Q'), ord('q')]: 
			break

		num_frames += 1


	# key = 0
	# num_frames = 1
	# while True:

	# 	#=====[ Step 1: get/preprocess frame	]=====
	# 	frame = get_next_frame (vc)

	# 	#=====[ Step 2: display and wait indefinitely on key	]=====
	# 	print "frame #: ", num_frames
	# 	cv2.imshow ('FRAME', frame)
	# 	key = cv2.waitKey ()

	# 	#=====[ Case: space bar -> add frame	]=====
	# 	if key == ord(' '):
			
	# 		print "===[ ADDING FRAME #", + num_frames + " ]==="
	# 		board.add_frame (frame)
	# 		if num_frames >= 1:
	# 			board.get_occlusion_changes ()
	# 		if num_frames >= 2:
	# 			board.update_game ()
	# 			board.display_occlusion_changes ()			

	# 	#=====[ Case: exit -> escape	]=====
	# 	elif key in [27, ord('Q'), ord('q')]: 
	# 		break

	# 	num_frames += 1

		# if num_frames >= 3:
			# frame = board.draw_last_move (frame)
		# cv2.imshow ("LIVE FEED", frame)



