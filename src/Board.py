from time import time
import pickle
from util import iter_algebraic_notations
from Square import Square

class Board:
	"""
		Class: Board
		------------
		class for representing everything about the board

		Member Variables:
			- image: numpy ndarray representing the image 
			- corners: list of point correspondances between board coordinates
						and image coordinates. (board_coords, image_coords, descriptor)
			- squares: list of SquareImage 

	"""

	def __init__ (self, name=None, image=None, board_points=None, image_points=None, sift_desc=None, filename=None):
		"""
			PRIVATE: Constructor
			--------------------
			constructs a BoardImage from it's constituent data
			or the filename of a saved one
		"""
		#=====[ Step 1: file/not file ]=====
		if filename:
			self.load (filename)
			return
	
		#=====[ Step 2: check arguments	]=====
		if None in [name, image, corner_board_points, corner_image_points]:
				raise StandardError ("Must enter all data arguments or a filename")

		#=====[ Step 3: set name	]=====
		if not name:
			self.name = str(time ())
		else:
			self.name = name

		#=====[ Step 4: set corners	]=====
		assert len(board_points) == len(image_points)
		assert len(board_points) == len(sift_desc)
		self.board_points = board_points
		self.image_points = image_points
		self.sift_desc = sift_desc

		#=====[ Step 5: get cv_analyzer, BIH, squares	]=====
		self.cv_analyzer = CVAnalyzer ()
		self.get_BIH ()
		self.construct_squares ()









	####################################################################################################
	##############################[ --- LOADING/SAVING --- ]############################################
	####################################################################################################	

	def save (self, filename):
		"""
			PUBLIC: save
			------------
			saves this object to disk
		"""
		pickle.dump (	{	'name':self.name,
							'image':self.image,
							'board_points': self.board_points,
							'image_points': self.image_points,
							'sift_desc':self.sift_desc
						}, 
						open(filename, 'w'))

	
	def load (self, filename):
		"""
			PUBLIC: load
			------------
			loads a past BoardImage from a file 
		"""
		save_file 	= open(filename, 'r')
		saved_dict 	= pickle.load (save_file)
		self.name 	= saved_dict['name']
		self.image 	= saved_dict['image']
		self.board_points = saved_dict['board_points']
		self.image_points = saved_dict['image_points']
		self.sift_desc = saved_dict ['sift_desc']








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
	##############################[ --- CV TASKS --- ]##################################################
	####################################################################################################

	def get_BIH (self):
		"""
			PRIVATE: get_BIH
			----------------
			finds the board-image homography 
		"""
		self.BIH = self.cv_analyzer.find_board_image_homography (self.board_points, self.image_points)


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


	def print_correspondences (self):
		"""
			PUBLIC: print_correspondences
			-----------------------------
			prints out a summary of all the point correspondences 
			that we have on hand
		"""
		title 			= "==========[ 	BoardImage: " + self.name + " ]=========="
		point_count		= "##### " + str(len(self.board_points)) + " point correspondances: #####"
		point_corr 		= '\n'.join(['	' + str(bp) + '->' + str(ip) for bp, ip in zip (self.board_points, self.image_points)])
		return '\n'.join ([title, point_count, point_corr])


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






