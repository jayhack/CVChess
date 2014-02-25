import cv2
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
		#=====[ Step 1: set up feature extractors	]=====
		self.corner_detector = cv2.FeatureDetector_create ('HARRIS')
		self.sift_descriptor = cv2.DescriptorExtractor_create('SIFT')










	####################################################################################################
	##############################[ --- FIND BOARD CORNER CORRESPONDENCES --- ]#########################
	####################################################################################################


	def get_harris_corners (self, image):
		"""
			Function: get_harris_corners
			----------------------------
			given an image, returns a list of cv2.KeyPoints representing
			the harris corners
		"""
		return self.corner_detector.detect (image)


	def get_sift_descriptors (self, image, kpts):
		"""
			Function: get_sift_descriptor
			-----------------------------
			given an image and a list of keypoints, this returns
			(keypoints, descriptors), each a list
		"""
		return self.sift_descriptor.compute (image, kpts)


	def get_corner_prob (self, harris_corner_sift):
		"""
			Function: get_corner_prob
			-------------------------
			given a sift representation of a harris corner, this returns the prob.
			that it corresponds to a board corner.
		"""
		raise NotImplementedError



	def get_board_corner_correspondences  (self, image):
		"""
			Function: get_board_corner_correspondences
			------------------------------------------
			given an image, this returns a list of point correspondences relating 
			board coordinates to image coordinates.
		"""
		#=====[ Step 1: get harris corners	]=====
		harris_corners = self.get_harris_corners (image)

		#=====[ Step 2: get SIFT descriptors for them	]=====
		harris_corners_sift = self.get_sift_descriptors (image, harris_corners)

		return harris_corners_sift











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


	def find_board_image_homography (self, board_points, image_points):
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
		assert (len (board_points) == len (image_points)) and (len(board_points) >= 4)

		#=====[ Step 2: get P	]=====
		P = self.get_P (board_points, image_points)

		#=====[ Step 3: SVD on P ]=====
		U, S, V = svd (P)

		#=====[ Step 4: assemble BIH from V	]=====
		BIH = self.assemble_BIH (V)

		return BIH
	



