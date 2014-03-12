from copy import deepcopy
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
		#=====[ Step 1: algebraic notation/colors	]=====
		self.an = algebraic_notation
		self.get_colors (self.an)
		self.get_vertices (BIH)

		#=====[ Step 2: init image regions	]=====
		self.image_region = None
		self.last_image_region = None

		#=====[ Step 3: init image regions normalized	]=====
		self.image_region_normalized = None
		self.last_image_region_normalized = None

		#=====[ Step 3: init contents histograms ]=====
		self.contents_histogram = None
		self.last_contents_histogram = None

		#=====[ Step 3: init edges	]=====
		self.edges = None
		self.last_edges = None


	
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
	##############################[ --- INITIAL IMAGE ANALYSIS --- ]####################################
	####################################################################################################

	def get_colors (self, algebraic_notation):
		"""
			Function: get_square_color
			--------------------------
			given a square's algebraic notation, this returns its binary color
			Note: 0 -> white, 1 -> colored
		"""
		tl = algebraic_notation_to_board_coords (algebraic_notation)[0]
		if (tl[0] + tl[1]) % 2 == 1:
			self.color = 1
			self.draw_color = (255, 0, 0)
		else:
			self.color = 0
			self.draw_color = (0, 0, 255)


	def get_vertices (self, BIH):
		"""
			PRIVATE: get_vertices
			---------------------
			fills self.board_vertices, self.image_vertices 
		"""
		self.board_vertices	= algebraic_notation_to_board_coords (self.an)
		self.image_vertices	= [board_to_image_coords (BIH, bv) for bv in self.board_vertices]






	####################################################################################################
	##############################[ --- UPDATING WITH FRAMES --- ]######################################
	####################################################################################################

	def update_image_region (self, image):
		"""
			PRIVATE: update_image_region
			----------------------------
			given an image, sets self.image_region with a the region of the image
			that corresponds to this square; additionally, applies a mask to the 
			areas outside of the square 
		"""
		self.last_image_region = self.image_region

		#=====[ Step 1: get bounding box coordinates	]=====
		iv = np.array (self.image_vertices).astype(int)
		x_min, x_max = min(iv[:, 0]), max(iv[:, 0])
		y_min, y_max = min(iv[:, 1]), max(iv[:, 1])

		#=====[ Step 2: extract image region 	]=====
		self.image_region = deepcopy(image[y_min:(y_max+1), x_min:(x_max + 1)])

		#=====[ Step 3: create mask	]=====
		self.image_region_mask = np.zeros (self.image_region.shape[:2], dtype=np.uint8)
		mask_coords = [(i[0] - x_min, i[1] - y_min) for i in self.image_vertices]
		cv2.fillConvexPoly (self.image_region_mask, np.array(mask_coords).astype(int), 1)

		#=====[ Step 4: apply mask (turns outside pixels to black)	]=====
		idx = (self.image_region_mask == 0)
		self.image_region[idx] = 0

		#=====[ Step 5: get intensity normalized image region	]=====
		self.last_image_region_normalized = self.image_region_normalized
		totals = np.sum (self.image_region, 2, dtype=np.float)
		self.image_region_normalized = self.image_region.astype(np.float)
		self.image_region_normalized[:, :, 0] = np.divide (self.image_region_normalized[:, :, 0], totals) * 255
		self.image_region_normalized[:, :, 1] = np.divide (self.image_region_normalized[:, :, 1], totals) * 255
		self.image_region_normalized[:, :, 2] = np.divide (self.image_region_normalized[:, :, 2], totals) * 255
		self.image_region_normalized = self.image_region_normalized.astype (np.uint8);


	def update_contents_histogram (self):
		"""
			PRIVATE: update_contents_histogram
			----------------------------------
			sets self.contents_histogram with a histogram of the colors in self.image_region,
			also sets up self.last_contents_histogram to the previous one.
		"""
		self.last_contents_histogram = self.contents_histogram

		#=====[ Step 1: get histograms for each	]=====
		num_buckets = [32]
		self.b_hist = cv2.calcHist(self.image_region_normalized, [0], None, num_buckets, [1, 256])
		self.g_hist = cv2.calcHist(self.image_region_normalized, [1], None, num_buckets, [1, 256])
		self.r_hist = cv2.calcHist(self.image_region_normalized, [2], None, num_buckets, [1, 256])

		#=====[ Step 3: update contents_histogram, last_contents_histogram	]=====
		self.contents_histogram = np.concatenate ([self.b_hist, self.g_hist, self.r_hist], 0).flatten ()


	def update_edges (self):
		"""
			PRIVATE: update_contents_edges
			------------------------------
			sets self.contents_edges as an image containing a canny
			edge detector applied to it
		"""
		self.last_edges = self.edges
		gray = cv2.cvtColor (self.image_region, cv2.COLOR_BGR2GRAY)
		self.edges = cv2.Canny (gray, 50, 40)



	def add_frame (self, image):
		"""
			PUBLIC: add_frame
			-----------------
			call this function to update all info about the square when a new frame 
			arrives
		"""
		#=====[ Step 1: update image_region	]=====
		self.update_image_region (image)

		#=====[ Step 2: update contents histograms	]=====
		self.update_contents_histogram ()

		#=====[ Step 3: update edges	]=====
		self.update_edges ()




	####################################################################################################
	##############################[ --- EXTRACTING INFO FROM IMAGE --- ]################################
	####################################################################################################

	def get_occlusion_change_features (self):
		"""
			PUBLIC: get_occlusion_change_features
			-------------------------------------
			returns a numpy array representing this square, optimized
			for discerning occlusion 
		"""
		# return np.sum(np.abs(self.image_region_normalized - self.last_image_region_normalized))
		return cv2.compareHist (self.contents_histogram, self.last_contents_histogram, 3)


	def get_occlusion_change (self):
		"""
			PUBLIC: get_occlusion_change
			----------------------------
			sets self.occlusion in response to the current contents 
			of the square 
		"""
		#=====[ Step 1: get occlusion change features	]=====
		features = self.get_occlusion_change_features ()
		return features



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


	def draw_vertices (self, image):
		"""
			PUBLIC: draw_vertices
			---------------------
			draws the verties of this square on the image
		"""
		for vertex in self.image_vertices:
			cv2.circle (image, (int(vertex[0]), int(vertex[1])), 5, (0, 0, 255))


	def show_image_region_normalized (self):

		print self.an
		cv2.imshow ('IMAGE_REGION_NORMALIZED', self.image_region_normalized)
		key = 0
		while key != 27:
			key = cv2.waitKey (30)


	def show_edges (self):

		# diff = (self.image_region_normalized - self.last_image_region_normalized)
		# gray = cv2.cvtColor ()
		# cv2.imshow ('NORMALIZED', cv2.pyrUp(cv2.pyrUp(np.abs(self.image_region_normalized - self.last_image_region_normalized))))

		cv2.imshow ('EDGES', cv2.pyrUp(cv2.pyrUp(self.edges)))
		cv2.imshow ('REGION', cv2.pyrUp(cv2.pyrUp(self.image_region)))

		key = 0
		while key != 27:
			key = cv2.waitKey (30)



