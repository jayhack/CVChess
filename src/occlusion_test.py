from copy import deepcopy
import os
import cv2
import numpy as np
import sklearn
import pickle
import CVAnalysis
from Board import Board
from util import *




if __name__ == '__main__':

	#=====[ Step 1: get all frames	]=====
	game_dir = '../data/game3'
	frame_filenames = [os.path.join(game_dir, f) for f in os.listdir(game_dir)]
	frames = [cv2.imread (frame_filename) for frame_filename in frame_filenames]
	im_size = frames[0].shape[0]
	frames = [f[int(im_size/2):, :] for f in frames]

	#=====[ Step 2: initialize the board	]=====
	corner_classifier = pickle.load (open('./corner_data/corner_classifier.obj', 'r'))	#more data
	board = Board(corner_classifier=corner_classifier)
	board.add_frame (frames[0])

	#####[ DEBUG: check BIH via vertices	]#####
	# board.draw_vertices (deepcopy(board.current_frame))
	# key = 0
	# while key != 27:
	# 	key = cv2.waitKey (30)


	#=====[ ITERATE THROUGH ALL FRAMES	]=====
	for index, frame in enumerate(frames[1:]):
		print_header ("ADDING FRAME: " + str(index))

		#=====[ Step 3: add frame ]=====
		board.add_frame (frame)

		#=====[ Step 4: get occlusion change	]=====
		board.get_occlusion_changes ()
		if index >= 1:
			board.update_game ()
			board.display_occlusion_changes ()



			key = 0
			while key != 27:
				key = cv2.waitKey (30)

