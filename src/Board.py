import pickle
from time import time
from copy import deepcopy
import cv2
import numpy as np
import Chessnut
import CVAnalysis
from Square import Square
from util import iter_algebraic_notations

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
	def __init__ (self, corner_classifier=None):
		"""
			PUBLIC: Constructor 
			-------------------
			initializes all variables 
		"""
		self.game 				= Chessnut.Game ()
		self.last_frame 		= None
		self.current_frame		= None
		self.num_frames 		= 0
		self.corner_classifier	= corner_classifier












	####################################################################################################
	##############################[ --- GAME MANAGEMENT --- ]###########################################
	####################################################################################################	

	def update_frames (self, new_frame):
		"""
			PRIVATE: update_frames 
			----------------------
			updates self.last_frame, self.current_frame
		"""
		#=====[ Step 1: rotate out frames	]=====
		self.last_frame = self.current_frame
		self.current_frame = new_frame
		self.num_frames += 1



	def add_frame (self, new_frame):
		"""
			PRIVATE: add_frame
			------------------
			adds a frame to the current game, from which 
			it will try to discern what move occurred 
		"""
		#=====[ Step 1: update frames	]=====
		self.update_frames (new_frame)

		#=====[ CASE: first frame ]=====
		if self.num_frames == 1:

			self.find_BIH ()
			self.construct_squares ()

		#=====[ CASE: subsequent frames	]=====
		else:
			pass
			#=====[ Step 2: get changes in occlusion	]=====
			# occlusion_changes = self.get_occlusion_changes ()

			#=====[ Step 3: infer the most likely move	]=====
			# self.infer_move (occlusion_changes)


	def get_occlusion_changes (self):
		"""
			PRIVATE: get_occlusion_changes
			------------------------------
			for each square, gets the change in probability
		"""
		raise NotImplementedError


	def get_occupation_probabilities (self, image):
		"""
			PRIVATE: get_occupation_probabilities
			-------------------------------------
			given an image, calculates and returns the probability of occupation 
			for each square
		"""
		raise NotImplementedError


	def is_valid_move (self, move):
		"""
			PRIVATE: is_valid_move
			----------------------
			given a move, returns true if it is valid on the current 
			game state 
		"""
		raise NotImplementedError








	####################################################################################################
	##############################[ --- CV TASKS --- ]##################################################
	####################################################################################################

	def find_BIH (self):
		"""
			PRIVATE: find_BIH
			----------------
			finds the board-image homography from self.current_frame, which is assumed 
			to be the first frame
		"""
		assert self.corner_classifier 
		self.BIH = CVAnalysis.find_board_image_homography (self.current_frame, self.corner_classifier)


	def construct_squares (self):
		"""
			PRIVATE: construct_squares
			--------------------------
			sets self.squares
		"""
		#=====[ Step 1: initialize self.squares to empty 8x8 grid	]=====
		self.squares = [[None for i in range(8)] for j in range(8)]

		#=====[ Step 2: create a square for each algebraic notation ]=====
		for square_an in iter_algebraic_notations ():

				new_square = Square (self.current_frame, self.BIH, square_an)
				square_location = new_square.board_vertices[0]
				self.squares [square_location[0]][square_location[1]] = new_square








	####################################################################################################
	##############################[ --- DATA MANAGEMENT --- ]###########################################
	####################################################################################################	

	def parse_occlusions (self, filename):
		"""
			PRIVATE: parse_occlusions
			-------------------------
			given a filename containing occlusions, returns it in 
			list format 
		"""
		return [line.strip().split(' ') for line in open(filename, 'r').readlines ()]


	def add_occlusions (self, filename):
		"""
			PUBLIC: add_occlusions
			----------------------
			given the name of a file containing occlusions, this will 
			add them to each of the squares 
		"""
		#=====[ Step 1: parse occlusions	]=====
		occlusions = self.parse_occlusions (filename)

		#=====[ Step 2: add them one-by-one	]=====
		for i in range(8):
			for j in range(8):
				self.squares [i][j].add_occlusion (occlusions[i][j])


	def get_occlusion_features (self):
		"""
			PUBLIC: get_occlusions
			----------------------
			returns X, y
			X: np.mat where each row is a feature vector representing a square
			y: list of labels for X
		"""
		X = [s.get_occlusion_features () for s in self.iter_squares ()]
		y = [s.occlusion for s in self.iter_squares ()]
		return np.array (X), np.array(y)







	####################################################################################################
	##############################[ --- CONSTRUCTING FROM OBJECTS --- ]#################################
	####################################################################################################	

	def construct_from_file (self, filename):
		"""
			PRIVATE: construct_from_file
			----------------------------
			loads a previously-saved BoardImage
		"""
		self.load (filename)


	def construct_from_points (self, name, image, board_points, image_points, sift_desc):
		"""
			PRIVATE: construct_from_points
			------------------------------
			fills out this BoardImage based on passed in fields 
		"""
		#=====[ Step 1: set name	]=====
		if not name:
			self.name = str(time ())
		else:
			self.name = name

		#=====[ Step 2: set image	]=====
		self.image = image

		#=====[ Step 3: set corners	]=====
		assert len(board_points) == len(image_points)
		assert len(board_points) == len(sift_desc)
		self.board_points = board_points
		self.image_points = image_points
		self.sift_desc = sift_desc

		#=====[ Step 4: get BIH, squares ]=====
		self.get_BIH ()
		self.construct_squares ()


	def construct_from_BIH (self, name, image, BIH):
		"""
			PRIVATE: construct_from_BIH
			---------------------------
			fills out this BoardImage based on a BIH, assuming the image 
			that you computed it from came from the same chessboard, same 
			pose 
		"""
		#=====[ Step 1: set name	]=====
		if not name:
			self.name = str(time ())
		else:
			self.name = name

		#=====[ Step 1: set BIH/image	]=====
		self.BIH = BIH
		self.image = image

		#=====[ Step 2: set squares	]=====
		self.construct_squares ()

		#=====[ Step 3: set everything else	]=====
		self.board_points = None
		self.image_points = None 
		self.sift_desc = None



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
							'sift_desc':self.sift_desc,
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
		self.get_BIH ()
		self.construct_squares ()








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


	def draw_squares (self, image):
		"""
			PUBLIC: draw_squares
			--------------------
			call this function to display a version of the image with square 
			outlines marked out
		"""	
		#=====[ Step 1: fill in all of the squares	]=====
		for square in self.iter_squares():
			image = square.draw_surface (image)

		#=====[ Step 2: draw them to the screen	]=====
		cv2.namedWindow ('board.draw_squares')
		cv2.imshow ('board.draw_squares', image)


	def draw_vertices (self, image):
		"""
			PUBLIC: draw_squares
			--------------------
			given a draw, this will draw each of the squares in self.squares
		"""
		#=====[ Step 1: draw all vertices	]=====
		for square in self.iter_squares ():
			square.draw_vertices (image)

		#=====[ Step 2: draw them to the screen	]=====
		cv2.namedWindow ('board.draw_vertices')
		cv2.imshow ('board.draw_vertices', image)




if __name__ == "__main__":
	
	board = Board (filename='test.bi')
	board.add_occlusions ('test.oc')
	X, y = board.get_occlusions ()



