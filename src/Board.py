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
		self.last_frame 		= None
		self.current_frame		= None
		self.color_kmeans 		= None
		self.corner_classifier	= corner_classifier









	####################################################################################################
	##############################[ --- MANAGING SQUARES --- ]##########################################
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


	def construct_squares (self, empty_board_frame):
		"""
			PRIVATE: construct_squares
			--------------------------
			given a frame containing the empty board, this function 
			fills self.squares with properly initialized squares
		"""
		#=====[ Step 1: initialize self.squares to empty 8x8 grid	]=====
		self.squares = [[None for i in range(8)] for j in range(8)]

		#=====[ Step 2: create a square for each algebraic notation ]=====
		for square_an in iter_algebraic_notations ():
				new_square = Square (empty_board_frame, self.BIH, square_an)
				ix = new_square.board_vertices[0]
				self.squares [ix[0]][ix[1]] = new_square


	def update_square_image_regions (self, frame):
		"""
			PRIVATE: update_square_image_regions
			------------------------------------
			given a frame, updates square.image_region for each square 
		"""
		for square in self.iter_squares ():
			square.update_image_region (frame)


	def update_square_color_hists (self):
		"""
			PRIVATE: update_square_colors
			-----------------------------
			applies self.color_kmeans to each square to get 
			its color histogram 
		"""
		assert self.color_kmeans
		for square in self.iter_squares ():
			square.update_color_hist (self.color_kmeans)











	####################################################################################################
	##############################[ --- INITIALIZE BOARD PLANE --- ]####################################
	####################################################################################################

	def find_BIH (self, empty_board_frame):
		"""
			PRIVATE: find_BIH
			-----------------
			given a frame containing the empty board, this function 
			finds the homography relating board coordinates to 
			image coordinates. (sets self.BIH)
		"""
		assert self.corner_classifier 
		self.BIH = CVAnalysis.find_board_image_homography (empty_board_frame, self.corner_classifier)


	def initialize_board_plane (self, empty_board_frame):
		"""
			PUBLIC: initialize_board_plane
			-------------------------------
			given a frame of the empty board, this function
			will:
				- find the BIH 
				- construct self.squares
		"""
		#=====[ Step 1: find the BIH	]=====
		self.find_BIH (empty_board_frame)

		#=====[ Step 2: construct squares	]=====
		self.construct_squares (empty_board_frame)










	####################################################################################################
	##############################[ --- INITIALIZE GAME --- ]###########################################
	####################################################################################################

	def get_color_kmeans (self):
		"""
			PRIVATE: get_color_kmeans
			-------------------------
			runs kmeans on all square image regions to get self.color_kmeans
		"""
		#=====[ Step 1: get all data for kmeans	]=====
		s_reg = [s.image_region for s in board.iter_squares ()]
		reshaped = [s.reshape ((s.shape[0]*s.shape[1], 3)) for s in s_reg]
		all_pixels = np.concatenate (reshaped, 0)

		#=====[ Step 2: fit self.color_kmeans	]=====
		self.color_kmeans = KMeans (n_clusters=4)
		self.color_kmeans.fit (all_pixels)


	def initialize_game (self, start_config_frame):
		"""
			PUBLIC: initialize_game
			-----------------------
			given a frame of the board with all pieces in their 
			start locations, this will initialize the game 
		"""
		#=====[ Step 1: construct game ]=====
		self.game = Chessnut.Game ()
		self.num_moves = 0

		#=====[ Step 2: get image regions for all squares	]=====
		for square in self.iter_squares ():
			square.update_image_region (start_config_frame)

		#=====[ Step 3: get self.color_kmeans	]=====
		self.get_color_kmeans ()

		#=====[ Step 4: set colors for each square	]=====
		self.update_square_colors ()











	####################################################################################################
	##############################[ --- SUBSEQUENT MOVES --- ]##########################################
	####################################################################################################

	def add_move (self, frame):
		"""
			PRIVATE: add_move
			-----------------
			adds a frame to the current game, assuming that a move
			has occurred from last frame
		"""
		#=====[ Step 1: update frames	]=====
		self.last_frame = self.current_frame
		self.current_frame = frame 
		self.num_moves += 1

		#=====[ Step 2: update square image regions, color hists	]=====
		self.update_square_image_regions (frame)
		self.update_square_color_hists ()

		#=====[ Step 3: infer move	]=====
		# self.infer_move ()





	####################################################################################################
	##############################[ --- INFERRING MOVES --- ]###########################################
	####################################################################################################

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
		self.squares_entered = {square.an:square.occlusion_change for square in self.iter_squares() if square.occlusion_change_direction == 'entered'}
		self.squares_exited = {square.an:square.occlusion_change for square in self.iter_squares() if square.occlusion_change_direction == 'exited'}		


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


	def add_scores (self, occ_map, ix):
		"""
			PRIVATE: add_scores
			-------------------
			given a map of occlusions (either enter or exit) and the index 
			being moved to, this will return a score for it 
		"""
		#=====[ NAIVE: just score of square at ix	]=====
		# return occ_map[ix[0], ix[1]]

		#=====[ NAIVE 2: score of square at ix, as well as 2 above, normalized	]=====
		v_ix = range(ix[0], 9)[:3]
		return np.sum([occ_map[v_ix[i]][ix[1]] for i in range(len(v_ix))])


	def get_move_score (self, move):
		"""
			PRIVATE: get_move_score
			-----------------------
			given a move in Chessnut move notation, returns its total score.
		"""		
		#=====[ Step 1: get move indices	]=====
		exit_an, enter_an = split_move_notation (move)
		enter_ix, exit_ix = an_to_index (enter_an), an_to_index (exit_an)
		print "=====[ 	GET MOVE SCORE ]====="
		print "move: ", move
		print "exit_an, enter_an", exit_an, enter_an
		print "exit_ix, enter_ix", exit_ix, enter_ix
		print "exit occlusion: ",  self.exit_occlusions[exit_ix[0], exit_ix[1]]
		print "enter occlusion: ",  self.enter_occlusions[enter_ix[0], enter_ix[1]]


		#=====[ Step 2: sum and return	]=====
		return self.add_scores(self.enter_occlusions, enter_ix) + self.add_scores(self.exit_occlusions, exit_ix)
		# return self.enter_occlusions[enter_ix[0], enter_ix[1]] + self.exit_occlusions[exit_ix[0], exit_ix[1]]


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
		all_moves = list(set(enter_moves + exit_moves))
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
		self.last_move = self.infer_move ()
		self.game.apply_move (self.last_move)




	def is_valid_move (self, move):
		"""
			PRIVATE: is_valid_move
			----------------------
			given a move, returns true if it is valid on the current 
			game state 
		"""
		raise NotImplementedError
























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
		for square in self.iter_squares():
			image = square.draw_surface (image)	
		return image


	def draw_vertices (self, image):
		"""
			PUBLIC: draw_vertices
			---------------------
			draws all square vertices on the image
		"""
		for square in self.iter_squares ():
			square.draw_vertices (image)
		return image


	def draw_square_an (self, an, image, color=(255, 0, 0)):
		"""
			PUBLIC: draw_square_an
			----------------------
			given a square's algebraic notation, this function
			will draw it on the provided image, then return the image 
		"""
		board_coords = algebraic_notation_to_board_coords (an)[0]
		square = self.squares[board_coords[0]][board_coords[1]]
		image = square.draw_surface (image, color)
		return image 


	def display_occlusion_changes (self):
		"""
			PUBLIC: display_occlusion_changes
			---------------------------------
			shows add_mat and sub_mat as heatmaps via 
			matplotlib
		"""
		#=====[ BOARD TOP 3	]=====
		self.entered_t2 = sorted([(key, value) for key, value in self.squares_entered.items ()], key=lambda x: x[1], reverse=True)[:2]
		self.exited_t2 = sorted([(key, value) for key, value in self.squares_exited.items ()], key=lambda x: x[1], reverse=True)[:2]

		disp_img = deepcopy(self.current_frame)		
		for an, value in self.entered_t2:
			disp_img = self.draw_square_an (an, disp_img, color=(255, 0, 0))
		for an, value in self.exited_t2:
			disp_img = self.draw_square_an (an, disp_img, color=(0, 0, 255))
		cv2.imshow ('Board', disp_img)


		#=====[ OCCLUSIONS HEATMAP	]=====
		column_labels = list('HGFEDCBA')
		row_labels = list('87654321')
		fig, ax1 = plt.subplots()
		heatmap = ax1.pcolor(self.enter_occlusions, cmap=plt.cm.Blues)
		ax1.xaxis.tick_top()
		ax1.set_xticks(np.arange(self.enter_occlusions.shape[0])+0.5, minor=False)
		ax1.set_yticks(np.arange(self.enter_occlusions.shape[1])+0.5, minor=False)
		ax1.xaxis.tick_top()
		ax1.set_xticklabels(row_labels, minor=False)
		ax1.set_yticklabels(column_labels, minor=False)
		# plt.title ('Positive Increases in Occlusion by Square')


		fig2, ax2 = plt.subplots()
		heatmap = ax2.pcolor(self.exit_occlusions, cmap=plt.cm.Reds)
		ax2.xaxis.tick_top()
		ax2.set_xticks(np.arange(self.enter_occlusions.shape[0])+0.5, minor=False)
		ax2.set_yticks(np.arange(self.enter_occlusions.shape[1])+0.5, minor=False)
		ax2.set_xticklabels(row_labels, minor=False)
		ax2.set_yticklabels(column_labels, minor=False)
		plt.show ()



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


	def draw_last_move (self, frame):
		"""
			PUBLIC: draw_last_move
			----------------------
			draws self.last_move onto the board
		"""
		exit_an, enter_an = split_move_notation (self.last_move)
		frame = self.draw_square_an (enter_an)
		return frame


	def show_square_edges (self):

		for square in self.iter_squares ():
			square.show_edges ()




if __name__ == "__main__":
	
	board = Board (filename='test.bi')
	board.add_occlusions ('test.oc')
	X, y = board.get_occlusions ()



