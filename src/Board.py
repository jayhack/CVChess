import numpy as np
from Square import Square

def homogenize (x):
	"""
		Function: homogenize
		--------------------
		given a tuple, this will return a numpy array representing it in
		homogenous coordinates
	"""
	return np.transpose(np.matrix ((x[0], x[1], 1)))


def dehomogenize (x):
	"""
		Function: dehomogenize
		----------------------
		given a homogenous vector, this returns a tuple
	"""
	x = list(np.array(x).reshape(-1,))
	return (x[0]/x[2], x[1]/x[2])


class Board:
	"""
		Class: Board
		------------
		class to hold and represent the entire board
	"""

	def __init__ (self, _H):
		"""
			PUBLIC: Constructor
			-------------------
			given _H, the homography relating world coordinates to image coordinates, this creates
			a representation of the board.
		"""
		#==========[ Step 1: set parameters	]==========
		self.H = _H

		#==========[ Step 2: construct squares	]==========
		self.squares = self.construct_squares ()
		for i in range(8):
			for j in range(8):
				print self.squares[i][j]


	def board_to_image_coords (self, board_coords):
		"""
			PRIVATE: board_to_image_coords
			------------------------------
			given a homography matrix and a point on the board in board coordinates, 
			this will return its (homogenous) coordinates in the image 
		"""
		return dehomogenize(np.dot (self.H, homogenize(board_coords)))


	def square_index_to_image_coords (self, square_index):
		"""
			PRIVATE: square_index_to_image_coords
			-------------------------------------
			given a square's index (i.e. top left corner = (0, 0)), this
			will return a list of the image coordinates of its vertices in clockwise
			fashion, starting from the top left
		"""
		x, y 					= square_index[0], square_index[1]
		vertex_board_coords 	= [(x,y), (x+1, y), (x+1, y+1), (x, y+1)]
		vertex_image_coords 	= [self.board_to_image_coords (c) for c in vertex_board_coords]
		return vertex_image_coords


	def construct_squares (self):
		"""
			PRIVATE: construct_squares
			--------------------------
			returns a 2d-array where the (i, j)th element is a Square object
		"""
		#=====[ Step 1: initialize self.squares	]=====
		squares = [[]]*8;

		#=====[ Step 2: construct each square	]=====
		for i in range (8):
			for j in range(8):

				index = (i, j)
				vertex_image_coords = self.square_index_to_image_coords (index)
				new_square = Square (index, vertex_image_coords)
				squares[i].append (new_square)

		return squares




