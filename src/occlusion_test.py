from copy import deepcopy
import cv2
import numpy as np
import sklearn
import pickle
import CVAnalysis
from Board import Board
from util import *




if __name__ == '__main__':

	#=====[ Step 1: get frames ]=====
	frame_0 = cv2.imread ('../data/game3/00.jpg')	
	frame_1 = cv2.imread ('../data/game3/01.jpg')
	frame_2 = cv2.imread ('../data/game3/02.jpg')	
	frame_3 = cv2.imread ('../data/game3/03.jpg')
	frame_4 = cv2.imread ('../data/game3/04.jpg')	
	im_size = frame_0.shape[0]
	frame_0 = frame_0[int(im_size/2):, :]
	frame_1 = frame_1[int(im_size/2):, :]
	frame_2 = frame_2[int(im_size/2):, :]
	frame_3 = frame_3[int(im_size/2):, :]	
	frame_4 = frame_4[int(im_size/2):, :]		

	#=====[ Step 2: initialize the board	]=====
	corner_classifier = pickle.load (open('./corner_data/corner_classifier.obj', 'r'))	#more data
	board = Board(corner_classifier=corner_classifier)

	#=====[ Step 3: add frames	]=====

	print_header ("ADDING FRAME: 0")
	board.add_frame (frame_0)

	print_header ("ADDING FRAME: 1")
	board.add_frame (frame_1)
	board.get_occlusion_changes ()

	print_header ("ADDING FRAME: 2")
	board.add_frame (frame_2)
	board.get_occlusion_changes ()
	# board.show_square_edges ()
	# board.draw_top_occlusion_changes (deepcopy(board.current_frame), num_squares=5)	

	print_header ("ADDING FRAME: 3")
	board.add_frame (frame_3)
	board.get_occlusion_changes ()
	# board.draw_top_occlusion_changes (deepcopy(board.current_frame), num_squares=6)	

	print_header ("ADDING FRAME: 4")
	board.add_frame (frame_4)
	board.get_occlusion_changes ()
	# board.draw_top_occlusion_changes (deepcopy(board.current_frame), num_squares=2)	



