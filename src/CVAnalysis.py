import os
from subprocess import call
from copy import deepcopy
from random import choice
import cv2
import numpy as np
from numpy import matrix, concatenate
from numpy.linalg import inv, pinv, svd
from sklearn.cluster import KMeans
from util import *



####################################################################################################
##############################[ --- IMAGE DIFFERENCE --- ]##########################################
####################################################################################################

def get_image_diff (img1, img2):
	"""
		Function: get_image_diff
		------------------------
		given two images, this finds the eroded/dilated difference 
		between them on a coarse grain.
		NOTE: assumes both are full-size, color
	"""
	#=====[ Step 1: convert to gray	]=====
	img1_gray = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor (img2, cv2.COLOR_BGR2GRAY)	

	#=====[ Step 2: downsample 	]=====
	img1_small = cv2.pyrDown(cv2.pyrDown(img1_gray))
	img2_small = cv2.pyrDown(cv2.pyrDown(img2_gray))	

	#=====[ Step 3: find differnece	]=====
	difference = img2_small - img1_small

	#=====[ Step 4: erode -> dilate	]=====
	kernel = np.ones ((4, 4), np.uint8)
	difference_ed = cv2.dilate(cv2.erode (difference, kernel), kernel)

	#=====[ Step 5: blow back up	]=====
	return cv2.pyrUp (cv2.pyrUp (difference_ed))






####################################################################################################
##############################[ --- CORNER DETECTION/DESCRIPTION--- ]###############################
####################################################################################################

def get_harris_corners (image):
	"""
		Function: get_harris_corners
		----------------------------
		given an image, returns a list of cv2.KeyPoints representing
		the harris corners
	"""
	corner_detector = cv2.FeatureDetector_create ('HARRIS')
	return corner_detector.detect (image)


def get_sift_descriptors (image, kpts):
	"""
		Function: get_sift_descriptor
		-----------------------------
		given an image and a list of keypoints, this returns
		(keypoints, descriptors), each a list
	"""
	sift_descriptor = cv2.DescriptorExtractor_create('SIFT')
	return sift_descriptor.compute (image, kpts)[1]







####################################################################################################
#################[ --- FINDING BOARD_IMAGE HOMOGRAPHY FROM POINTS CORRESPONDENCES --- ]#############
####################################################################################################	

def get_P_rows (bp, ip):
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


def get_P (board_points, image_points):
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
	return concatenate ([get_P_rows (bp, ip) for bp, ip in zip(board_points, image_points)], 0)


def assemble_BIH (V):
	"""
		Function: assemble_BIH
		----------------------
		given V, assembles a matrix from it 
	"""
	vt = V.T[:, -1]
	return concatenate ([vt[0:3].T, vt[3:6].T, vt[6:9].T], 0);


def point_correspondences_to_BIH (board_points, image_points):
	"""
		Function: point_correspondences_to_BIH
		--------------------------------------
		board_image: a BoardImage object *containing point correspondences*
		returns: matrix BIH such that
			BIH * pb = pi
			pb = point in (homogenized) board coordinates 
			pw = point in (homogenized) image coordinates
	"""
	#=====[ Step 1: ensure correct number of correspondances ]=====
	assert (len (board_points) == len (image_points)) and (len(board_points) >= 4)

	#=====[ Step 2: get P	]=====
	P = get_P (board_points, image_points)

	#=====[ Step 3: SVD on P ]=====
	U, S, V = svd (P)

	#=====[ Step 4: assemble BIH from V	]=====
	BIH = assemble_BIH (V)
	return BIH








####################################################################################################
#################[ --- FINDING BIH FROM IMAGE --- ]#################################################
####################################################################################################

def cluster_points (points, cluster_dist=7):
	"""
		Function: cluster_points
		------------------------
		given a list of points and the distance between them for a cluster,
		this will return a list of points with clusters compressed 
		to their centroid 
	"""
	#=====[ Step 1: init old/new points	]=====
	old_points = np.array (points)
	new_points = []

	#=====[ ITERATE OVER OLD_POINTS	]=====
	while len(old_points) > 1:
		p1 = old_points [0]
		distances = np.array([euclidean_distance (p1, p2) for p2 in old_points])
		idx = (distances < cluster_dist)
		points_cluster = old_points[idx]
		centroid = get_centroid (points_cluster)
		new_points.append (centroid)
		old_points = old_points[np.invert(idx)]

	return new_points


def get_chessboard_corner_candidates (image, corner_classifier):
	"""
		Function: get_chessboard_corner_candidates
		------------------------------------------
		given an image, returns a list of points (represented as tuples)
		that correspond to possible chessboard corners. Note: they have 
		been clustered on output to account for harris corners surrounding 
		actual chessboard corners
	"""
	#=====[ Step 1: get corners/sift descriptors	]=====
	hc = get_harris_corners(image)
	sd = get_sift_descriptors (image, hc)

	#=====[ Step 2: make predictions	]=====
	predictions = corner_classifier.predict (sd)
	idx = (predictions == 1)
	chessboard_corners = [c.pt for c, i in zip(hc, idx) if i]
	# chessboard_corners = [c.pt for c in hc] #no classifier - for demos

	#=====[ Step 3: cluster corners	]=====
	chessboard_corners = cluster_points (chessboard_corners)

	return chessboard_corners


def snap_points_to_lines (lines, points, max_distance=10):
	"""
		Function: snap_points_to_lines
		------------------------------
		given a list of lines and a list of points, this will
		return a 2d list where the i, jth element is the jth point 
		that falls on the ith line
		max_distance = maximum distance a point can be from a line 
		before it no longer is considered part of it.
	"""
	#=====[ Step 1: initialize points grid	]=====
	grid = [[] for line in lines]

	#=====[ Step 2: iterate through corners, adding to lines	]=====
	for point in points:
		distances = np.array([get_line_point_distance (line, point) for line in lines])
		if np.min (distances) < max_distance:
			line_index = np.argmin(distances)
			grid[line_index].append (point)

	#=====[ Step 3: sort each one by y coordinate	]=====
	for i in range(len(grid)):
		grid[i].sort (key=lambda x: x[0])

	return grid


def is_BIH_inlier (all_BIH_ip, corner, pix_dist=10):
	"""
		Function: is_BIH_inlier
		-----------------------
		given the set of BIH image points and a corner,
		returns true if the corner is within pix_dist
		of any BIH ip
	"""
	return any([(euclidean_distance(ip, corner) <= pix_dist) for ip in all_BIH_ip])


def compute_inliers (BIH, corners):
	"""
		Function: compute_inliers
		-------------------------
		given a board-image homography and a set of all corners,
		this will return the number that are inliers 
	"""
	#=====[ Step 1: get a set of all image points for vertices of board coords	]=====
	all_board_points = []
	for i in range(9):
		for j in range(9):
			all_board_points.append ((i, j))
	all_BIH_ip = [board_to_image_coords (BIH, bp) for bp in all_board_points]

	#=====[ Step 2: get booleans for each corner being an inlier	]=====
	num_inliers = sum ([is_BIH_inlier (all_BIH_ip, corner) for corner in corners])
	return num_inliers


def evaluate_homography (horz_indices, vert_indices, horz_points_grid, vert_points_grid, corners):
	"""
		Function: evaluate_homography
		-----------------------------
		given two point grids and the indexes (we think) they correspond to on the 
		image, this will find point correspondences, compute a homography and score it
		returns (BIH, score), where score = number of corners on chessboard that match
	"""
	board_points, image_points = [], []
	# print "#####[ 	EVALUATE HOMOGRAPHY 	]#####"
	# print "horz_indices: ", horz_indices
	# print "vert_indices: ", vert_indices


	#=====[ Step 1: find point correspondences	]=====
	for i in range (len(horz_points_grid)):
		for j in range(len(vert_points_grid)):

			#=====[ get horizontal/vertical coordinates	]=====
			board_x = horz_indices[i]
			board_y = vert_indices[j]

			#=====[ check if a point exists in intersection, add it	]=====
			intersection = list(set(horz_points_grid[i]).intersection (set(vert_points_grid[j])))
			if len(intersection) > 0:
				image_points.append (intersection[0])
				board_points.append ((board_x, board_y))


	#=====[ Step 2: compute a homography from it	]=====
	BIH = point_correspondences_to_BIH (board_points, image_points)

	#=====[ Step 3: compute score	]=====
	score = compute_inliers (BIH, corners)

	return BIH, score


def find_BIH (horz_points_grid, horz_indices, vert_points_grid, vert_indices, corners):
	"""
		Function: find_BIH
		------------------
		given two point grids and the indices that row corresponds to on the board
		(up to a shift), this will find the point correspondences and BIH
	"""

	#=====[ Step 1: shift both indices all the way to bottom left ]=====
	horz_indices = horz_indices - horz_indices[0] 
	vert_indices = vert_indices - vert_indices[0]

	#=====[ Step 2: initialize parameters	]=====
	BIH_score_list = []

	#=====[ ITERATE THROUGH ALL SHIFTS	]=====
	hi = deepcopy(horz_indices)
	while (hi[-1] < 9):
		# print "hi: ", hi
		vi = deepcopy (vert_indices)
		while (vi[-1] < 9):
			# print "	vi: ", vi

			#=====[ evaluate homography	]=====
			BIH, score = evaluate_homography (hi, vi, horz_points_grid, vert_points_grid, corners)
			BIH_score_list.append ((BIH, score))

			# print "		", score

			#=====[ shift vi ]=====
			vi += 1
		hi += 1

	BIH_score_list.sort (key=lambda x: x[1], reverse=True)
	return BIH_score_list


def get_chessboard_lines (corners, image):
	"""
		Function: get_chessboard_lines
		------------------------------
		given a list of corners represented as tuples, this returns 
		(horizontal_lines, vertical_lines) represented as (a, b, c)
		pairs 
	"""
	#=====[ Step 1: make an image for Matlab scripts ]=====
	corners_img = np.zeros (image.shape[:2], dtype=np.uint8)
	for corner in corners:
		corners_img[int(corner[1])][int(corner[0])] = 255									

	#=====[ Step 2: save to IPC	]=====
	cv2.imwrite ('./IPC/corners.png', corners_img)

	#=====[ Step 3: run matlab script	]=====
	os.chdir ('./autofind_lines')
	call(["/Applications/MATLAB_R2013b.app/bin/matlab", "-nojvm", "-nodisplay", "-nosplash", "-r ", "autofind_lines"])
	os.chdir ('../')

	#=====[ Step 4: get the lines back	]=====
	horz_lines_indexed = np.genfromtxt ('./IPC/horizontal_lines.csv', delimiter=',')
	vert_lines_indexed = np.genfromtxt ('./IPC/vertical_lines.csv', delimiter=',')
	horz_lines = zip(list(horz_lines_indexed[0, :]), list(horz_lines_indexed[1, :]))
	vert_lines = zip(list(vert_lines_indexed[0, :]), list(vert_lines_indexed[1, :]))
	horz_indices = horz_lines_indexed[2, :]
	vert_indices = vert_lines_indexed[2, :]

	#=====[ Step 3: snap points to grid ]===
	horz_points_grid = snap_points_to_lines (horz_lines, corners)
	vert_points_grid = snap_points_to_lines (vert_lines, corners)

	#=====[ Step 4: find homography	]=====
	BIH_score_list = find_BIH (horz_points_grid, horz_indices, vert_points_grid, vert_indices, corners)
	return BIH_score_list


def find_board_image_homography (image, corner_classifier):
	"""
		Function: find_board_image_homography
		-------------------------------------
		given an image presumed to be of a chessboard near the bottom
		of the screen, this function returns the homography relating 
		board coordinates to image coordinates
		corner_classifier is an sklearn classifier that maps points 
		to {0,1}, where 1 means it is a chessboard corner
	"""
	img_copy = deepcopy(image)

	#=====[ Step 1: get chessboard corners	]=====
	corners = get_chessboard_corner_candidates (image, corner_classifier)
	#####[ DEBUG: DRAW CORNERS	]#####
	# img_copy = draw_points_xy (img_copy, corners)
	# cv2.namedWindow ('corners')
	# cv2.imshow ('corners', img_copy)
	# key = 0
	# while key != 27:
	# 	key = cv2.waitKey (30)

	#=====[ Step 2: generate a list of BIH candidates	]=====
	BIH_score_list = get_chessboard_lines (corners, image)
	BIH = BIH_score_list[0][0]
	return BIH


	#=====[ Step 3: evaluate candidates until one is satisfactory	]=====
	# all_board_points = []
	# for i in range(9):
		# for j in range(9):
			# all_board_points.append ((i, j))

	# for BIH, score in BIH_score_list:

	# 	#=====[ Step 1: get image points	]=====
	# 	all_BIH_ip = [board_to_image_coords (BIH, bp) for bp in all_board_points]
	# 	all_BIH_kp = [cv2.KeyPoint(x=ip[0], y=ip[1], size=5) for ip in all_BIH_ip]

	# 	#=====[ Step 2: get SIFT descriptors for each one	]=====
	# 	sifd_desc = get_sift_descriptors(image, all_BIH_kp)

	# 	#=====[ Step 3: get probabilities for each	]=====
	# 	predictions = corner_classifier.predict_proba (sift_desc)
	# 	print shape(predictions)












####################################################################################################
##############################[ --- FUNCTIONS ON BOARD SQUARES --- ]################################
####################################################################################################

def extract_polygon_region (image, polygon):
	"""
		Function: extract_polygon_region
		--------------------------------
		given a polygon (list of point tuples), returns an image 
		masked with that polygon 
	"""
	#=====[ Step 1: create the mask	]=====
	mask = np.zeros ((image.shape[0], image.shape[1]))
	if not type(polygon) == type(np.array([])):
		polygon = np.array(polygon)
	cv2.fillConvexPoly(mask, polygon, 1)

	#=====[ Step 2: copy over the image	]=====
	polygon_region = np.zeros ((image.shape[0], image.shape[1]))
	idx = (mask != 0)
	polygon_region[idx] = image[idx]

	return polygon_region






















