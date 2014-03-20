import pickle
from time import time
from copy import deepcopy
from collections import Counter
import cv2
import numpy as np
import scipy as sp
import Chessnut
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
			PRIVATE: update_square_color_hists
			----------------------------------
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
		img_regs = [s.image_region for s in self.iter_squares ()]
		reshaped = [ir.reshape ((ir.shape[0]*ir.shape[1], 3)) for ir in img_regs]
		all_pixels = np.concatenate (reshaped, 0)

		#=====[ Step 2: fit self.color_kmeans	]=====
		self.color_kmeans = KMeans (n_clusters=4)
		self.color_kmeans.fit (all_pixels)


	def get_piece_color_indices (self):
		"""
			PRIVATE: get_piece_color_indices
			--------------------------------
			sets self.piece_color_indices to a dict mapping {'black', 'white'}
			to their respective indices in self.color_kmeans
		"""
		#=====[ Step 1: get sums of left, right sides of board	]=====
		b_hist = Counter ({})
		w_hist = Counter ({})
		for i in range(8):
			for j in range(2):
				b_hist += self.squares[i][j].color_hist 
		for i in range (8):
			for j in range (6, 8):
				w_hist += self.squares[i][j].color_hist 

		#=====[ Step 2: normalize histograms ]=====
		b_hist = np.array(list(b_hist.values())).astype(np.float)
		b_hist = b_hist / np.sum(b_hist)
		w_hist = np.array(list(w_hist.values())).astype(np.float)
		w_hist = w_hist / np.sum (w_hist)

		#=====[ Step 3: take differnce and construct	]=====
		diff = b_hist - w_hist
		self.piece_color_indices = {'black':np.argmax(diff), 'white':np.argmin(diff)}


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
		self.update_square_image_regions (start_config_frame)

		#=====[ Step 3: get self.color_kmeans	]=====
		self.get_color_kmeans ()

		#=====[ Step 4: set colors for each square	]=====
		self.update_square_color_hists ()

		#=====[ Step 5: get piece colors	]=====
		self.get_piece_color_indices ()











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

		#=====[ Step 3: update movement heatmaps	]=====
		self.update_movement_heatmaps ()
		
		#=====[ Step 4: infer move/update game	]=====
		self.update_game ()











	####################################################################################################
	##############################[ --- MOVEMENT HEATMAPS --- ]#########################################
	####################################################################################################

	def get_turn (self):
		"""
			PRIVATE: get_turn
			-----------------
			returns 'white' or 'black' corresponding to the person 
			who just moved (i.e. result of move is in this frame)
		"""
		assert self.num_moves > 0
		if (self.num_moves) % 2 == 1:
			return 'white'
		else:
			return 'black'


	def get_moving_piece_color_index (self):
		"""
			PRIVATE: get_moving_piece_color_index
			-------------------------------------
			returns the kmeans index of the color of the piece that 
			just moved 
		"""
		return self.piece_color_indices[self.get_turn ()]


	def update_movement_heatmaps (self):
		"""
			PRIVATE: update_movement_heatmaps
			---------------------------------
			gets self.enter_heatmap and self.exit_heatmap based on square 
			colors
		"""
		#=====[ Step 1: get color of moving piece	]=====
		piece_color_index = self.get_moving_piece_color_index ()

		#=====[ Step 2: initialize matrices	]=====
		self.enter_heatmap = np.zeros ((8, 8))
		self.exit_heatmap = np.zeros ((8, 8))

		#=====[ FILL HEATMAPS	]=====
		for i, j, square in self.iter_squares_index ():

			#=====[ Step 3: get normalized histogram change	]=====
			color_hist_change = square.get_color_hist_change (piece_color_index)
			if color_hist_change > 0.0:
				self.enter_heatmap[i][j] = color_hist_change
			else:
				self.exit_heatmap[i][j] = abs(color_hist_change)









	####################################################################################################
	##############################[ --- MOVEMENT HEATMAPS -> MOVES --- ]################################
	####################################################################################################

	def place_on_heatmap (self, hm, ix, piece_height=4):
		"""
			PRIVATE: place_on_heatmap
			-------------------------
			given the index of a piece and the heatmap to place it on,
			this updates the heatmap and returns it 
		"""
		x_coord, y_coord = ix[0], ix[1]
		xs = range(x_coord, 8)[:piece_height]
		for x in xs:
			hm[x][y_coord] = 1
		return hm


	def get_expected_heatmaps (self, move):
		"""
			PRIVATE: get_expected_heatmaps
			------------------------------
			given a move in chessnut notation, this returns 
			(exit_hm_exp, enter_hm_exp), corresponding to what we 
			would expect to see 
		"""
		#=====[ Step 1: get enter, exit square in an	]=====
		exit_an, enter_an = split_move_notation (move)

		#=====[ Step 2: get enter, exit square in ix	]=====
		exit_ix, enter_ix = an_to_index (exit_an), an_to_index (enter_an)

		#=====[ Step 3: create heatmaps	]=====
		exit_hm_exp = np.zeros ((8,8))
		enter_hm_exp = np.zeros ((8,8))
		exit_hm_exp = self.place_on_heatmap (exit_hm_exp, exit_ix)
		enter_hm_exp = self.place_on_heatmap (enter_hm_exp, enter_ix)		

		return exit_hm_exp, enter_hm_exp


	def score_move (self, move, v_obs):
		"""
			PRIVATE: score_move
			-------------------
			given a move in chessnut notation, returns a score for 
			it based on empirical observations 
		"""
		#=====[ Step 1: get/normalize heatmaps	]=====
		exit_hm_exp, enter_hm_exp = self.get_expected_heatmaps (move)
		exit_hm_exp = exit_hm_exp.flatten() / np.sum(exit_hm_exp.flatten())
		enter_hm_exp = enter_hm_exp.flatten() / np.sum(enter_hm_exp.flatten())		

		#=====[ Step 2: convert to vector form	]=====
		v_exp = np.concatenate([exit_hm_exp, enter_hm_exp], 0)		


		#=====[ Step 3: compute and return dot product	]=====
		# return np.dot (v_obs, v_exp) #screws up on second knight...
		return sp.spatial.distance.cosine (v_obs, v_exp) # works on knight, second bishop (other one didnt...)




	def get_observed_heatmap_vec (self):
		"""
			PRIVATE: get_observed_heatmap_vec
			---------------------------------
			returns a vector representing the observed heatmaps.
			this involves normalization -> concatenation of self.exit_heatmap and 
			self.enter_heatmap
		"""
		#=====[ Step 1: get/normalize vectors	]=====
		exit_hm_obs = self.exit_heatmap.flatten().astype(np.float)
		exit_hm_obs = exit_hm_obs / np.sum(exit_hm_obs)
		enter_hm_obs = self.enter_heatmap.flatten().astype(np.float)
		enter_hm_obs = enter_hm_obs / np.sum(enter_hm_obs)

		#=====[ Step 2: concatenate and return	]=====
		return np.concatenate ([exit_hm_obs, enter_hm_obs], 0)


	def infer_move (self):
		"""
			PRIVATE: infer_move
			-------------------
			operates on self.enter_heatmap and self.exit_heatmap to infer 
			the most likely move to have taken place. returns in Chessnut 
			notation
		"""
		#=====[ Step 1: get all moves	]=====
		moves = self.game.get_moves()

		#=====[ Step 2: get observed heatmap vector	]=====
		v_obs = self.get_observed_heatmap_vec ()

		#=====[ Step 2: score all moves	]=====
		scores = [self.score_move (m, v_obs) for m in moves]

		#=====[ Step 3: get and return max	]=====
		min_ix = np.argmin(scores)
		best_move = moves[min_ix]
		print "best move: ", best_move

		#####[ DEBUG: Print out heatmaps 	]#####
		# x_hm_exp, e_hm_exp = self.get_expected_heatmaps ('e2e4')
		# print "=====[ EXIT_HM_OBS	]====="
		# print np.around(self.exit_heatmap, decimals=3)
		# print "=====[ ENTER_HM_OBS	]====="
		# print np.around(self.enter_heatmap, decimals=3)
		# print "=====[ EXIT_HM_EXP	]====="
		# print x_hm_exp
		# print "=====[ EXIT_HM_EXP	]====="
		# print e_hm_exp


		return best_move


	def update_game (self):
		"""
			PRIVATE: update_game
			--------------------
			determines the move, then updates self.game
		"""
		self.last_move = self.infer_move ()
		self.game.apply_move (self.last_move)


























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


	def display_movement_heatmaps (self):
		"""
			PUBLIC: display_movement_heatmaps
			---------------------------------
			shows self.enter_heatmap and self.exit_heatmap via matplotlib
		"""
		# #=====[ BOARD TOP 3	]=====
		# self.entered_t2 = sorted([(key, value) for key, value in self.squares_entered.items ()], key=lambda x: x[1], reverse=True)[:2]
		# self.exited_t2 = sorted([(key, value) for key, value in self.squares_exited.items ()], key=lambda x: x[1], reverse=True)[:2]
		# disp_img = deepcopy(self.current_frame)		
		# for an, value in self.entered_t2:
		# 	disp_img = self.draw_square_an (an, disp_img, color=(255, 0, 0))
		# for an, value in self.exited_t2:
		# 	disp_img = self.draw_square_an (an, disp_img, color=(0, 0, 255))
		# cv2.imshow ('Board', disp_img)


		#=====[ OCCLUSIONS HEATMAP	]=====
		column_labels = list('HGFEDCBA')
		row_labels = list('87654321')
		fig, ax1 = plt.subplots()
		heatmap = ax1.pcolor(self.enter_heatmap, cmap=plt.cm.Blues)
		ax1.xaxis.tick_top()
		ax1.set_xticks(np.arange(self.enter_heatmap.shape[0])+0.5, minor=False)
		ax1.set_yticks(np.arange(self.enter_heatmap.shape[1])+0.5, minor=False)
		ax1.xaxis.tick_top()
		ax1.set_xticklabels(row_labels, minor=False)
		ax1.set_yticklabels(column_labels, minor=False)
		# plt.title ('Positive Increases in Piece color by Square')

		fig2, ax2 = plt.subplots()
		heatmap = ax2.pcolor(self.exit_heatmap, cmap=plt.cm.Reds)
		ax2.xaxis.tick_top()
		ax2.set_xticks(np.arange(self.exit_heatmap.shape[0])+0.5, minor=False)
		ax2.set_yticks(np.arange(self.exit_heatmap.shape[1])+0.5, minor=False)
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
		frame = self.draw_square_an (exit_an, frame, color=(0, 0, 255))
		frame = self.draw_square_an (enter_an, frame, color=(0, 255, 0))
		return frame


	def show_square_edges (self):

		for square in self.iter_squares ():
			square.show_edges ()




if __name__ == "__main__":
	
	board = Board (filename='test.bi')
	board.add_occlusions ('test.oc')
	X, y = board.get_occlusions ()



