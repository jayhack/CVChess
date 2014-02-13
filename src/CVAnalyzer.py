from numpy import matrix, concatenate
from numpy.linalg import inv, pinv
from util import homogenize
from BoardImage import BoardImage

class CVAnalyzer:
	"""
		Class: CVAnalyzer
		-----------------
		class encapsulating all CV modules, including:
			- getting images in from camera
			- finding board homography
	"""

	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
			board_image: BoardImage object, the first frame
		"""
		pass




	####################################################################################################
	##############################[ --- FINDING BOARD_IMAGE HOMOGRAPHY --- ]############################
	####################################################################################################	

	def get_BP_rows (self, bp_j):
		"""
			Function: get_BP_rows
			---------------------
			bp_j: point in board coordinaes
			returns: matrix BP_j
				BP_j = 	[x y z 0 0 0 0 0 0]
						[0 0 0 x y z 0 0 0]
						[0 0 0 0 0 0 x y z]
		"""
		x, y, z = bp_j[0], bp_j[1], 1
		return matrix 	([	
								[x, y, z, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, x, y, z, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, x, y, z]
							])


	def get_BP (self, board_points):
		"""
			Function: get_BP_matrix
			-----------------------
			board_points: list of points in board coordinates
			returns: matrix BP such that
				BP * bih = IP
				bih = columnized version of BIH
				IP 	= columnized version of image points (see get_IP)
		"""
		return concatenate ([self.get_BP_rows (board_point) for board_point in board_points], 0)


	def get_IP (self, image_points):
		"""
			Function: get_IP
			----------------
			image points: list of points in image coordinates
			returns: matrix IP such that
				IP[i*3][0] 		= ip_i.x
				IP[i*3 + 1][0]	= ip_i.y
				IP[i*3 + 2][0]	= ip_i.z
		"""
		return concatenate([homogenize(image_point) for image_point in image_points], 0)


	def solve_homogenous_system (self, BP, IP):
		"""			
			Function: solve_homogenous_system
			---------------------------------
			solves for x in the following equation:
			BP*x = IP
		"""
		return pinv(BP) * IP


	def get_BIH_least_squares (self, BP, IP):
		"""
			Function: get_BIH_least_squares
			-------------------------------
			finds BIH using a least-squares approach
		"""
		x_ls = self.solve_homogenous_system (BP, IP)
		return concatenate ([x_ls[0:3].T, x_ls[3:6].T, x_ls[6:9].T], 0);


	def find_board_image_homography (self, board_image):
		"""
			Function: find_board_homography
			-------------------------------
			board_image: a BoardImage object *containing point correspondences*
			returns: matrix BIH such that
				BIH * pb = pi
				pb = point in (homogenized) board coordinates 
				pw = point in (homogenized) image coordinates
		"""
		#=====[ Step 1: ensure correct number of correspondances ]=====
		board_points, image_points = board_image.board_points, board_image.image_points
		assert (len (board_points) == len (image_points)) and (len(board_points) >= 4)

		#=====[ Step 2: get BP and IP ]=====
		BP = self.get_BP (board_points)
		IP = self.get_IP (image_points)

		#=====[ Step 3: get BIH ]===
		BIH = self.get_BIH_least_squares (BP, IP)
		return BIH


