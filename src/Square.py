import numpy as np
from parameters import display_parameters

def get_square_color (square_index):
	"""
		Function: get_square_color
		--------------------------
		given a square index, returns its color as a tuple: (R, G, B)
	"""
	return display_parameters['square_colors'][sum(square_index) % 2]


class Square:
	"""
		Class: Square
		-------------
		class to hold and represent all data relevant to a single
		board square 
	"""

	def __init__ (self, _index, _vertex_coords):
		"""
			PRIVATE: Constructor
			--------------------
			_index: this square's 'index' on the board. (x, y). (i.e. top left = (0, 0))
			_vertext_corners: list of image coordinates of this square's coordinates in 
								clockwise order
		"""
		#==========[ Step 1: set parameters	]==========
		self.index = _index
		self.vertex_coords = _vertex_coords
		self.color = get_square_color (_index)
		print self


	
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
	##############################[ --- DRAWING --- ]###################################################
	####################################################################################################

	def draw_surface (self, draw):
		"""
			PUBLIC: draw_surface
			--------------------
			draws the surface of this square on the image
		"""
		draw.polygon (self.vertex_coords, fill=self.color)


	# def draw_vertices (self, draw):
	# 	"""
	# 		PUBLIC: draw_vertices
	# 		---------------------
	# 		draws the verties of this square on the image
	# 	"""
	# 	for vertex in self.vertex_coords:
	# 		top_left = (vertex[0])
	# 		draw.rectangle 






