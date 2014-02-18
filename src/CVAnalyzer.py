from numpy import matrix, concatenate
from numpy.linalg import inv, pinv, svd
from util import homogenize, print_message, board_to_image_coords
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
	##############################[ --- FIND BOARD CORNER CORRESPONDENCES --- ]#########################
	####################################################################################################

	def get_harris_corners (self, image):
		"""
			Function: get_harris_corners
			----------------------------
			given an image, returns a list of image coordinates 
			of all harris corners
		"""
		raise NotImplementedError 


	def get_sift_representation (self, image, image_point):
		"""
			Function: get_sift_representation
			---------------------------------
			given an image and point within it, this returns a sift representation
		"""
		raise NotImplementedError


	def get_corner_prob (self, harris_corner_sift):
		"""
			Function: get_corner_prob
			-------------------------
			given a sift representation of a harris corner, this returns the prob.
			that it corresponds to a board corner.
		"""
		raise NotImplementedError


	def filter_harris_corners (self, image, harris_corners, prob_threshold=0.8):
		"""
			Function: filter_harris_corners 
			-------------------------------
			given an image and a list of image coordinates corresponding to harris 
			corners, this returns a 'filtered' list of (harris corner, corner prob)
		"""
		#=====[ Step 1: get sift rep of corners	]=====
		harris_corners_sift = [self.get_sift_representation(image, c) for c in corners_ic]

		#=====[ Step 2: get prob. of each	]=====
		corner_probs = [self.get_corner_prob (hcs) for hcs in harris_corners_sift]

		#=====[ Step 3: filter and return	]=====
		return [(hc, prob) for hc, prob in zip(harris_corners_sift, corner_probs) if prob > prob_threshold]


	def get_board_corner_correspondences  (self, image):
		"""
			Function: get_board_corner_correspondences
			------------------------------------------
			given an image, this returns a list of point correspondences relating 
			board coordinates to image coordinates.
		"""
		raise NotImplementedError









		
	####################################################################################################
	##############################[ --- FINDING BOARD_IMAGE HOMOGRAPHY --- ]############################
	####################################################################################################	

	def get_P_rows (self, bp, ip):
			"""
			Function: get_P_rows
			--------------------
			bp: point in board coordinates
			ip: corresponding point in image coordinates
			returns: matrix P_j
				P_j = 	[x y z 0 0 0 0 0 0]
						[0 0 0 x y z 0 0 0]
			"""
			u, v = ip[0], ip[1]
			x, y, z = bp[0], bp[1], 1
			return matrix 	([
								[x, y, z, 0, 0, 0, -u*x, -u*y, -u],
								[0, 0, 0, x, y, z, -v*x, -v*y, -v],								

							])

	def get_P (self, board_points, image_points):
		"""
			Function: get_P
			---------------
			board_points: list of points from board 
			image_points: list of points from image
			returns: matrix P such that 
				P * M = 0
				M = columnized version of BIH

			P is 2n * 9, where n = number of correspondences
		"""
		return concatenate ([self.get_P_rows (bp, ip) for bp, ip in zip(board_points, image_points)], 0)


	def assemble_BIH (self, V):
		"""
			Function: assemble_BIH
			----------------------
			given V, assembles a matrix from it 
		"""
		vt = V.T[:, -1]
		return concatenate ([vt[0:3].T, vt[3:6].T, vt[6:9].T], 0);


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
		# return matrix 	([	
		# 						[x, y, z, 0, 0, 0, 0, 0, 0],
		# 						[0, 0, 0, x, y, z, 0, 0, 0],
		# 						[0, 0, 0, 0, 0, 0, x, y, z]
		# 					])

		#####[ Trying it without the bottom left...	]#####
		return matrix 	([	
								[x, y, z, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, x, y, z, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, x, y, 0]
							])


	def get_BP (self, board_points):
		"""
			Function: get_BP_matrix
			-----------------------
			board_points: list of points in board coordinates
			returns: matrix BP such that
				BP * BIH = IP
				bih = columnized version of BIH
				IP 	= columnized version of image points (see get_IP)
		"""
		return concatenate ([self.get_BP_rows (board_point) for board_point in board_points], 0)


	def alt_homogenize (self, image_point):
		c = homogenize (image_point)
		c[2][0] = -1
		return c

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
		# return concatenate([homogenize(image_point) for image_point in image_points], 0)
		return concatenate([self.alt_homogenize(image_point) for image_point in image_points], 0)		


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

		#=====[ Step 2: get P	]=====
		P = self.get_P (board_points, image_points)

		#=====[ Step 3: SVD on P ]=====
		U, S, V = svd (P)

		#=====[ Step 4: assemble BIH from V	]=====
		BIH = self.assemble_BIH (V)

		return BIH
	



