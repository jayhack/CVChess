#=====[ standard ]=====
import os

#=====[ PIL	]=====
import matplotlib.pyplot as plt
import Image, ImageDraw

#=====[ our modules	]=====
from BoardImage import BoardImage
from CVAnalyzer import CVAnalyzer
from Board import Board
from util import print_welcome, print_message, print_status

#=====[ globals ]=====
board_image_dir = '../data/marked'


if __name__ == "__main__":
	print_welcome ()

	#==========[ Step 1: get board_image ]==========
	print_status ("Main", "loading board image")
	bi_filename = os.path.join (board_image_dir, 'micah1.bi')
	# bi_filename = os.path.join (board_image_dir, 'above.bi')
	board_image = BoardImage (filename=bi_filename)

	#==========[ Step 2: construct cv_analyzer, get BIH ]==========
	print_status ("Main", "creating cv_analyzer")
	cv_analyzer = CVAnalyzer ()
	print_status ("Main", "finding BIH (board-image homography)")
	BIH	= cv_analyzer.find_board_image_homography (board_image)

	#==========[ Step 3: construct the board	]==========
	print_status ("Main", "constructing the board")
	board = Board (BIH)

	#==========[ Step 4: draw squares on image	]==========
	print_status ("Main", "drawing the board")
	pil_image = Image.fromarray (board_image.image)
	drawer = ImageDraw.Draw (pil_image)
	board.draw_vertices (drawer)

	#==========[ Step 6: display image	]==========
	pil_image.show ()






	