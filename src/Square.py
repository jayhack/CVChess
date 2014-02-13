import numpy as np
from parameters import display_parameters
from util import board_to_image_coords, image_to_board_coords, algebraic_notation_to_board_coords
from util import iter_algebraic_notations, iter_board_vertices


class Square:
	"""
		Class: Square
		-------------
		class to hold and represent all data relevant to a single
		board square 
	"""

	def __init__ (self, BIH, algebraic_notation):
		"""
			PRIVATE: Constructor
			--------------------
			_index: this square's 'index' on the board. (x, y). (i.e. top left = (0, 0))
			_vertext_corners: list of image coordinates of this square's coordinates in 
								clockwise order
		"""
		#==========[ Step 1: set parameters	]==========
		self.an 			= algebraic_notation
		self.vertices_bc	= algebraic_notation_to_board_coords (algebraic_notation)
		self.vertices_ic	= [board_to_image_coords (BIH, bc) for bc in self.vertices_bc]
		self.color 			= self.get_square_color (self.an)

	
	def __str__ (self):
		"""
			Function: __str__
			-----------------
			prints out a string representation of the square
		"""
		title = "=====[ 	Square: " + str(self.index) + " ]====="
		top_left = "top left: " + str(self.vertex_coords[0])
		top_right = "top right: " + str(self.vertex_coords[1])
		bottom_right = "bottom right " + str(self.vertex_coords[2])
		bottom_left = "bottom left: " + str(self.vertex_coords[3])
		return '\n'.join ([title, top_left, top_right, bottom_right, bottom_left])




	####################################################################################################
	##############################[ --- UTILITIES --- ]#################################################
	####################################################################################################

	def get_square_color (self, algebraic_notation):
		"""
			Function: get_square_color
			--------------------------
			given a square's algebraic notation, this returns its binary color
			Note: 0 -> white, 1 -> colored
		"""
		tl = algebraic_notation_to_board_coords (algebraic_notation)[0]
		return (tl[0] + tl[1]) % 2 





	####################################################################################################
	##############################[ --- DRAWING --- ]###################################################
	####################################################################################################

	def draw_surface (self, draw):
		"""
			PUBLIC: draw_surface
			--------------------
			draws the surface of this square on the image
		"""
		draw.polygon (self.vertex_coords, fill=self.color)


	def draw_vertices (self, draw):
		"""
			PUBLIC: draw_vertices
			---------------------
			draws the verties of this square on the image
		"""
		for vertex in self.vertices_ic:
			top_left = (vertex[0] - 8, vertex[1] - 8)
			bottom_right = (vertex[0] + 8, vertex[1] + 8)
			draw.rectangle ([top_left, bottom_right], fill=(0, 0, 255))






