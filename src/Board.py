from util import iter_algebraic_notations
from Square import Square

class Board:
	"""
		Class: Board
		------------
		class to hold and represent the entire board
	"""

	def __init__ (self, BIH):
		"""
			PUBLIC: Constructor
			-------------------
			BIH: board to image projective transformation
		"""
		#==========[ Step 1: set parameters	]==========
		self.BIH = BIH

		#==========[ Step 2: construct squares	]==========
		self.squares = self.construct_squares ()


	####################################################################################################
	##############################[ --- UTILITIES --- ]#################################################
	####################################################################################################

	def iter_squares (self):
		""" 
			PRIVATE: iter_squares
			---------------------
			iterates over all squares in this board
		"""
		for i in range(8):
			for j in range(8):
				yield self.squares [i][j]



	####################################################################################################
	##############################[ --- CONSTRUCTION --- ]##############################################
	####################################################################################################

	def construct_squares (self):
		"""
			PRIVATE: construct_squares
			--------------------------
			returns a 2d-array where the (i, j)th element is a Square object
		"""
		#=====[ Step 1: initialize self.squares to empty 8x8 grid	]=====
		squares = [[None for i in range(8)] for j in range(8)]

		#=====[ Step 2: create a square for each algebraic notation ]=====
		for square_an in iter_algebraic_notations ():

				new_square = Square (self.BIH, square_an)
				square_location = new_square.vertices_bc[0]
				squares [square_location[0]][square_location[1]] = new_square

		return squares



	##################################################################################################
	##############################[ --- INTERFACE --- ]#################################################
	####################################################################################################

	def __str__ (self):
		"""
			PUBLIC: __str__
			---------------
			prints out the current representation of the board
		"""
		for i in range(8):
			for j in range(8):
				print self.squares[i][j].an
			print "\n"


	def draw_squares (self, draw):
		"""
			PUBLIC: draw_squares
			--------------------
			given a draw, this will draw each of the squares in self.squares
		"""
		for square in self.iter_squares ():
			square.draw_surface (draw)


	def draw_vertices (self, draw):
		"""
			PUBLIC: draw_squares
			--------------------
			given a draw, this will draw each of the squares in self.squares
		"""
		for square in self.iter_squares ():
			square.draw_vertices (draw)



