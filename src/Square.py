import numpy as np
import cv2
from parameters import display_parameters
from util import board_to_image_coords, image_to_board_coords, algebraic_notation_to_board_coords
from util import iter_algebraic_notations, iter_board_vertices


class Square:
	"""
		Class: Square
		-------------
		class to hold and represent all data relevant to a single
		board square 
		Paramters:
			- image: copy of the image this square is found in
			- BIH: board-image homography
			- algebraic notation:
	"""

	def __init__ (self, image, BIH, algebraic_notation):
		"""
			PRIVATE: Constructor
			--------------------
			_index: this square's 'index' on the board. (x, y). (i.e. top left = (0, 0))
			_vertext_corners: list of image coordinates of this square's coordinates in 
								clockwise order
		"""
		#=====[ Step 1: algebraic notation/draw color	]=====
		self.an = algebraic_notation
		self.get_draw_color (self.an)

		#=====[ Step 2: get vertices in board/image coords	]=====
		self.get_vertices (BIH)

		#=====[ Step 3: extract image region corresponding to this square	]=====
		self.get_image_region (image)

		#=====[ Step 3: find a histogram of the content sof self.image_region ]=====
		self.get_contents_histogram ()

	
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
	##############################[ --- EXTRACTING INFO FROM IMAGE --- ]################################
	####################################################################################################

	def get_draw_color (self, algebraic_notation):
		"""
			Function: get_square_color
			--------------------------
			given a square's algebraic notation, this returns its binary color
			Note: 0 -> white, 1 -> colored
		"""
		tl = algebraic_notation_to_board_coords (algebraic_notation)[0]
		if (tl[0] + tl[1]) % 2 == 1:
			self.draw_color = (255, 0, 0)
		else:
			self.draw_color = (0, 0, 255)


	def get_vertices (self, BIH):
		"""
			PRIVATE: get_vertices
			---------------------
			fills self.board_vertices, self.image_vertices 
		"""
		self.board_vertices	= algebraic_notation_to_board_coords (self.an)
		self.image_vertices	= [board_to_image_coords (BIH, bv) for bv in self.board_vertices]


	def get_image_region (self, image):
		"""
			PRIVATE: get_image_region
			-------------------------
			given an image, sets self.image_region with a the region of the image
			that corresponds to this square; additionally, applies a mask to the 
			areas outside of the square 
		"""
		#=====[ Step 1: get bounding box coordinates	]=====
		iv = np.array (self.image_vertices).astype(int)
		x_min, x_max = min(iv[:, 0]), max(iv[:, 0])
		y_min, y_max = min(iv[:, 1]), max(iv[:, 1])

		#=====[ Step 2: extract image region 	]=====
		self.image_region = image[y_min:(y_max+1), x_min:(x_max + 1)]

		#=====[ Step 3: create mask	]=====
		self.image_region_mask = np.zeros (self.image_region.shape[:2])
		mask_coords = [(i[0] - x_min, i[1] - y_min) for i in self.image_vertices]
		cv2.fillConvexPoly (self.image_region_mask, np.array(mask_coords).astype(int), 1)

		#=====[ Step 4: apply mask (turns outside pixels to black)	]=====
		idx = (self.image_region_mask == 0)
		self.image_region[idx] = 0


	def get_contents_histogram (self):
		"""
			PRIVATE: get_contents_histogram
			-------------------------------
			sets self.get_contents_histogram with a histogram of the pixel values
			that occur within the *masked* self.image_region
		"""
		#=====[ Step 1: get histograms for each	]=====
		num_buckets = [16]
		self.b_hist = cv2.calcHist(self.image_region, [0], None, num_buckets, [0, 256])
		self.g_hist = cv2.calcHist(self.image_region, [1], None, num_buckets, [0, 256])
		self.r_hist = cv2.calcHist(self.image_region, [2], None, num_buckets, [0, 256])		

		#=====[ Step 2: account for all the black in images due to mask	]=====
		self.b_hist[0] = 0
		self.g_hist[0] = 0
		self.r_hist[0] = 0

		#=====[ Step 3: concatenate to get hist_features	]=====
		self.contents_histogram = np.concatenate ([self.b_hist, self.g_hist, self.r_hist], 0)





	####################################################################################################
	##############################[ --- DRAWING --- ]###################################################
	####################################################################################################

	def draw_surface (self, image):
		"""
			PUBLIC: draw_surface
			--------------------
			draws the surface of this square on the image
		"""
		cv2.fillConvexPoly(image, np.array(self.image_vertices).astype(int), self.draw_color)
		return image


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






