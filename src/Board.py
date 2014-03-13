import pickle
from time import time
from copy import deepcopy
import cv2
import numpy as np
import Chessnut
import matplotlib.pyplot as plt
import CVAnalysis
from Square import Square
from util import *

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

		#=====[ Step 2: find params on first frame ]=====
		if self.num_frames == 1:
			self.find_BIH ()
			self.construct_squares ()

		#=====[ Step 3: update squares	]=====
		for square in self.iter_squares ():
			square.add_frame (new_frame)


	def display_occlusion_changes (self):
		"""
			PUBLIC: display_occlusion_changes
			---------------------------------
			shows add_mat and sub_mat as heatmaps via 
			matplotlib
		"""
				#=====[ BOARD TOP 3	]=====
		self.added_t3 = sorted([(key, value) for key, value in self.squares_entered.items ()], key=lambda x: x[1], reverse=True)[:3]
		print "ADDED: ", self.added_t3
		self.subtracted_t3 = sorted([(key, value) for key, value in self.squares_exited.items ()], key=lambda x: x[1], reverse=True)[:3]
		print "SUBTRACTED: ", self.subtracted_t3
		added_img = deepcopy(self.current_frame)
		
		for an, value in self.added_t3:
			added_img = self.draw_square_an (an, added_img)
		cv2.imshow ("ADDED", added_img)

		sub_img = deepcopy(self.current_frame)
		for an, value in self.subtracted_t3:
			sub_img = self.draw_square_an (an, sub_img)
		cv2.imshow ("SUBTRACTED", sub_img)

		#=====[ OCCLUSIONS HEATMAP	]=====
		column_labels = list('ABCDEFGH')
		row_labels = list('12345678')
		fig, ax1 = plt.subplots()
		heatmap = ax1.pcolor(self.enter_occlusions, cmap=plt.cm.Blues)
		ax1.xaxis.tick_top()

		column_labels = list('ABCDEFGH')
		row_labels = list('12345678')
		fig2, ax2 = plt.subplots()
		heatmap = ax2.pcolor(self.exit_occlusions, cmap=plt.cm.Reds)
		ax2.xaxis.tick_top()
		plt.show ()


	def get_occlusion_changes (self):
		"""
			PRIVATE: get_occlusion_changes
			------------------------------
			for each square, gets the probability that it is now occluded
		"""
		#=====[ Step 1: get add_mat, sub_mat ]=====
		self.enter_occlusions = np.zeros ((8, 8))
		self.exit_occlusions = np.zeros ((8, 8))
		for i, j, square in self.iter_squares_index ():
			
			(oc, ocd) = square.get_occlusion_change ()
			if ocd == 'entered':
				self.enter_occlusions[i, j] = oc
			elif ocd == 'exited':
				self.exit_occlusions[i, j] = oc

		#=====[ Step 2: get a matrix of all squares and their occlusions 	]=====
		self.squares_entered = {square.an:square.occlusion_change for square in self.iter_squares() if square.occlusion_change_direction == 'added'}
		self.squares_exited = {square.an:square.occlusion_change for square in self.iter_squares() if square.occlusion_change_direction == 'subtracted'}		


	def get_enter_moves (self, square_an):
		"""
			PRIVATE: get_enter_moves
			------------------------
			given a square, gets all valid moves where the piece 
			moves into this square 
		"""
		suffix = square_an[0].lower() + str(square_an[1])
		return [move for move in self.game.get_moves () if move[-2:] == suffix]


	def get_exit_moves (self, square_an):
		"""
			PRIVATE: get_exit_moves
			-----------------------
			given a square, gets all valid moves where the piece 
			moves out of this square
		"""
		prefix = square_an[0].lower() + str(square_an[1])
		return [move for move in self.game.get_moves () if move[:2] == prefix]


	def get_move_score (self, move):
		"""
			PRIVATE: get_move_score
			-----------------------
			given a move in Chessnut move notation, returns its total score.
		"""		


		#=====[ Step 1: get move indices	]=====
		exit_an, enter_an = split_move_notation (move)
		enter_ix, exit_ix = an_to_index (enter_an), an_to_index (exit_an)
		# print "=====[ 	GET MOVE SCORE ]====="
		# print "move: ", move
		# print "enter, exit_an", enter_an, exit_an
		# print "enter, exit_ix", enter_ix, exit_ix
		# print "enter occlusion: ",  self.enter_occlusions[enter_ix[0], enter_ix[1]]
		# print "exit occlusion: ",  self.exit_occlusions[exit_ix[0], exit_ix[1]]

		#=====[ Step 2: sum and return	]=====
		return self.enter_occlusions[enter_ix[0], enter_ix[1]] + self.exit_occlusions[exit_ix[0], exit_ix[1]]


	def infer_move (self):
		"""
			PRIVATE: infer_move
			-------------------
			operates on self.enter_occlusions and self.exit_occlusions
			in order to find the most likely move 
		"""
		#=====[ Step 1: get enter, exit coordinates	]=====
		enter_index = np.unravel_index(np.argmax(self.enter_occlusions), self.enter_occlusions.shape)
		exit_index = np.unravel_index(np.argmax(self.exit_occlusions), self.exit_occlusions.shape)		

		#=====[ Step 2: convert to algebraic notation	]=====
		enter_an = index_to_an (enter_index)
		exit_an = index_to_an (exit_index)
		# print "Enter index, algebraic: ", enter_index, enter_an
		# print "Exit index, algebraic: ", exit_index, exit_an		

		#=====[ Step 3: get enter, exit moves	]=====
		enter_moves = self.get_enter_moves (enter_an)
		exit_moves = self.get_exit_moves (exit_an)
		all_moves = enter_moves + exit_moves
		# print '=====[ ENTER MOVES ]====='
		# print enter_moves
		# print '=====[ EXIT MOVES ]====='
		# print exit_moves

		#=====[ Step 4: score each	]=====
		print "ALL MOVES: ", all_moves
		scores = np.array([self.get_move_score (move) for move in all_moves])
		print "ALL SCORES: ", list(scores)
		best_index = np.argmax(scores)
		best_move = all_moves[best_index]
		print "BEST MOVE: ", best_move
		return best_move


	def update_game (self):
		"""
			PRIVATE: update_game
			--------------------
			determines the move, then updates self.game
		"""
		move = self.infer_move ()
		self.game.apply_move (move)




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


	def iter_squares_an (self):
		"""
			PRIVATE: iter_squares_an
			------------------------
			iterates over all Squares in self.squares
			returns (alegebraic_notation, square)
		"""
		for i in range (8):
			for j in range (8):
				yield (self.squares[i][j].an, self.squares[i][j])


	def iter_squares_index (self):
		"""
			PRIVATE: iter_squares_index
			------------------------
			iterates over all Squares in self.squares
			returns (i, j, square), where i, j are indices into 
			self.squares
		"""
		for i in range (8):
			for j in range (8):
				yield (i, j , self.squares[i][j])




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
			PUBLIC: draw_vertices
			---------------------
			draws all square vertices on the image
		"""
		#=====[ Step 1: draw all vertices	]=====
		for square in self.iter_squares ():
			square.draw_vertices (image)

		#=====[ Step 2: draw them to the screen	]=====
		cv2.namedWindow ('board.draw_vertices')
		cv2.imshow ('board.draw_vertices', image)


	def draw_square_an (self, an, image):
		"""
			PUBLIC: draw_square_an
			----------------------
			given a square's algebraic notation, this function
			will draw it on the provided image, then return the image 
		"""
		board_coords = algebraic_notation_to_board_coords (an)[0]
		square = self.squares[board_coords[0]][board_coords[1]]
		image = square.draw_surface (image)
		return image 



	def draw_top_occlusion_changes (self, image, num_squares=5):
		"""
			PUBLIC: draw_top_occlusion_changes
			----------------------------------
			draws the surfaces of the top 'num_squares' squares in terms of 
			their occlusion changes 
		"""
		#=====[ Step 1: get threshold	]=====
		occlusion_changes = [square.get_occlusion_change () for square in self.iter_squares ()]
		occlusion_changes.sort ()
		threshold = occlusion_changes[-num_squares]
		

		#=====[ Step 2: draw qualifying squares with 'added' ]=====
		add_img = deepcopy (image)
		for square in self.iter_squares ():
			if square.get_occlusion_change () >= threshold:
				if square.piece_added_or_subtracted () == 'added':
					print square.get_occlusion_change ()
					add_img = square.draw_surface (add_img)
		cv2.namedWindow ('PIECES ADDED')
		cv2.imshow ('PIECES ADDED', add_img)


		#=====[ Step 3: draw qualifying squares with 'subtracted'	]=====
		sub_img = deepcopy (image)
		for square in self.iter_squares ():
			if square.get_occlusion_change () >= threshold:
				if square.piece_added_or_subtracted () == 'subtracted':
					print square.get_occlusion_change ()
					sub_img = square.draw_surface (sub_img)
		cv2.namedWindow ('PIECES SUBTRACTED')
		cv2.imshow ('PIECES SUBTRACTED', sub_img)

		key = 0
		while key != 27:
			key = cv2.waitKey (30)


	def show_square_edges (self):

		for square in self.iter_squares ():
			square.show_edges ()




if __name__ == "__main__":
	
	board = Board (filename='test.bi')
	board.add_occlusions ('test.oc')
	X, y = board.get_occlusions ()



