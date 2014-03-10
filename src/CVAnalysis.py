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


def is_BIH_inlier (all_BIH_ip, corner, pix_dist=6):
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
	print "#####[ 	EVALUATE HOMOGRAPHY 	]#####"
	print "horz_indices: ", horz_indices
	print "vert_indices: ", vert_indices



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
	if (horz_indices[0] == 0) and (vert_indices[0] == 1):
		print "=====[ (this should be the correct one) ]====="
		for bp, ip in  zip(board_points, image_points):
			print bp, ": ", ip

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
	best_BIH = np.zeros ((3,3))
	best_score = -1

	#=====[ ITERATE THROUGH ALL SHIFTS	]=====
	hi = deepcopy(horz_indices)
	while (hi[-1] < 9):
		print "hi: ", hi
		vi = deepcopy (vert_indices)
		while (vi[-1] < 9):
			print "	vi: ", vi

			#=====[ evaluate homography	]=====
			BIH, score = evaluate_homography (hi, vi, horz_points_grid, vert_points_grid, corners)

			#=====[ update best	]=====
			if score > best_score:
				best_score = score
				best_BIH = BIH
				print '				=====[ 	NEW BEST SCORE (# inliers) ]====='
				print '				# inliers: ', best_score

			#=====[ shift vi	]=====
			vi += 1
		hi += 1

	return best_BIH





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
	BIH = find_BIH (horz_points_grid, horz_indices, vert_points_grid, vert_indices, corners)
	return BIH




def sort_point_grid (grid, sort_coord):
	"""
		Function: sort_point_grid
		-------------------------
		given a grid and a coordinate to sort by, this returns
		a sorted version
	"""
	#=====[ Step 1: get mean sort coords	]=====
	mean_sort_coords = []
	for i in range(len(grid)):
		mean_sort_coord = np.mean([x[sort_coord] for x in grid[i]])
		mean_sort_coords.append (mean_sort_coord)
	mean_sort_coords = np.array (mean_sort_coords)

	#=====[ Step 2: make sort grid	]=====
	sort_grid = [(g, m) for g, m in zip(grid, mean_sort_coords)]
	sort_grid.sort (key=lambda x: x[1])
	grid = [g for g, m in sort_grid]

	return grid


def get_line (point, grid):

	for index, line in enumerate(grid):
		if point in line:
			return index

	assert False


def extract_point_correspondences (horz_grid, vert_grid, grid_points):
	"""
		Function: extract_point_correspondences
		---------------------------------------
		given a horizontal grid and a vertical grid, this returns a list 
		of point correspondences
	"""
	image_points = []
	board_points = []
	for image_point in grid_points:
		board_x = 2 + get_line (image_point, vert_grid)
		board_y = get_line (image_point, horz_grid)

		image_points.append (image_point)
		board_points.append ((board_x, board_y))

	return image_points, board_points





####################################################################################################
#################[ --- MESSY RANSAC --- ]###########################################################
####################################################################################################

quadrants_bp = {
	
	'tl': 	[	
				(0, 2), (0, 3), (0, 4),
				(1, 2), (1, 3), (1, 4),
				(2, 2), (2, 3), (2, 4),
				(3, 2), (3, 3), (3, 4),
				(4, 2), (4, 3), (4, 4),
			],

	'bl':	[	
				(5, 2), (5, 3), (5, 4),
				(6, 2), (6, 3), (6, 4),
				(7, 2), (7, 3), (7, 4),
				(8, 2), (8, 3), (8, 4),
			],

	'tr': 	[
				(0, 5), (0, 6),
				(1, 5), (1, 6),
				(2, 5), (2, 6),
				(3, 5), (3, 6),
				(4, 5), (4, 6),
			],

	'br': 	[
				(5, 5), (5, 6),
				(6, 5), (6, 6),
				(7, 5), (7, 6),
				(8, 5), (8, 6),
			]

}
qbp= {
	
	'tl': 	[	
				(0, 2), (0, 3), #(0, 4),
				# (1, 2), (1, 3), #(1, 4),
				# (2, 2), (2, 3), (2, 4),
				# (3, 2), (3, 3), (3, 4),
				# (4, 2), (4, 3), (4, 4),
			],

	'bl':	[	
				# (5, 2), (5, 3), (5, 4),
				# (6, 2), (6, 3), (6, 4),
				# (7, 2), (7, 3), (7, 4),
				(8, 2), (8, 3), #(8, 4),
			],

	'tr': 	[
				# (0, 5), (0, 6),
				(1, 5), (1, 6),
				# (2, 5), (2, 6),
				# (3, 5), (3, 6),
				# (4, 5), (4, 6),
			],

	'br': 	[
				# (5, 5), (5, 6),
				# (6, 5), (6, 6),
				# (7, 5), (7, 6),
				(8, 5), (8, 6),
			]
}
all_board_points = (quadrants_bp['tl'] + quadrants_bp['tr'] + quadrants_bp['bl'] + quadrants_bp['br'])


def get_messy_point_grids (horz_lines, vert_lines, corners):
	"""
		Function: get_messy_point_grids
		-------------------------------
		given horizontal lines, vertical lines (in abc) and a set of corners,
		this will return two 'messy grids' that organize points on horizontal 
		lines and vertical lines.
	"""
	#=====[ Step 1: snap points to grid	]=====
	horz_lines = [abc_to_rho_theta(l) for l in horz_lines]
	vert_lines = [abc_to_rho_theta(l) for l in vert_lines]
	horz_grid = snap_points_to_lines (horz_lines, corners)
	vert_grid = snap_points_to_lines (vert_lines, corners)

	#=====[ Step 2: get all points that are on each grid	]=====
	all_horz_points = [p for h in horz_grid for p in h]
	all_vert_points = [p for v in vert_grid for p in v]
	grid_points = list(set(all_horz_points).intersection(set(all_vert_points)))

	#=====[ Step 3: create and sort grids ]=====
	messy_horz_grid = snap_points_to_lines (horz_lines, grid_points)
	messy_vert_grid = snap_points_to_lines (vert_lines, grid_points)
	messy_horz_grid = sort_point_grid (messy_horz_grid, 1)
	messy_vert_grid = sort_point_grid (messy_vert_grid, 0)
	#####[ DEBUG: print out grid lengths	]#####
	print '=====[ horz_grid lengths	]====='
	for g in horz_grid:
		print len(g)
	print '=====[ vert grid lengths]====='
	for g in vert_grid:
		print len(g)

	return messy_horz_grid, messy_vert_grid


def get_quadrants_ip (messy_horz_grid, messy_vert_grid):
	"""
		Function: get_quadrants_ip
		--------------------------
		returns quadrants_ip, which is a dict mapping quadrant 
		name to a list of points that fall within it 
	"""
	#=====[ Step 1: get top_half, bottom_half	]=====
	fourth 			= int(len(messy_horz_grid)/4)
	three_fourth 	= int(3*len(messy_horz_grid)/4)
	half 			= int(len(messy_horz_grid)/2)
	top_half 		= set([p for l in messy_horz_grid[:half] for p in l])
	bottom_half 	= set([p for l in messy_horz_grid[half:] for p in l])

	top_fourth		= set([p for l in messy_horz_grid[:fourth] for p in l])
	bottom_fourth	= set([p for l in messy_horz_grid[three_fourth:] for p in l])	

	#=====[ Step 2: get left half, right_half	]=====
	fourth 			= int(len(messy_vert_grid)/4)
	three_fourth 	= int(3*len(messy_vert_grid)/4)

	half = int(len(messy_vert_grid)/2)
	left_half 		= set([p for l in messy_vert_grid[:half] for p in l])
	right_half 		= set([p for l in messy_vert_grid[half:] for p in l])
	left_fourth		= set([p for l in messy_vert_grid[:fourth] for p in l])
	right_fourth 		= set([p for l in messy_vert_grid[three_fourth:] for p in l])

	#=====[ Step 3: assemble quadrants	]=====
	quadrants_ip = {

		'tl':list(top_half.intersection(left_half)),
		'bl':list(bottom_half.intersection(left_half)),
		'tr':list(top_half.intersection(right_half)),
		'br':list(bottom_half.intersection(right_half))

	}
	q_ip = {

		'tl':list(top_fourth.intersection(left_fourth)),
		'bl':list(bottom_fourth.intersection(left_fourth)),
		'tr':list(top_fourth.intersection(right_fourth)),
		'br':list(bottom_fourth.intersection(right_fourth))

	}
	corners = (quadrants_ip['tl'] + quadrants_ip['tr'] + quadrants_ip['bl'] + quadrants_ip['br'])
	return corners, q_ip


def sample_from_quadrants (quadrants):
	"""
		Function: sample_from_quadrants
		-------------------------------
		returns a randomly-chosen set of 4 points, one 
		from each quadrant
	"""
	return 	[	
				choice(quadrants['tl']), choice(quadrants['tr']), 
				choice(quadrants['bl']), choice(quadrants['br'])
			]

def is_BIH_inlier (all_BIH_ip, corner, pix_dist=6):
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
	all_BIH_ip = [board_to_image_coords (BIH, bp) for bp in all_board_points]

	#=====[ Step 2: get booleans for each corner being an inlier	]=====
	num_inliers = sum ([is_BIH_inlier (all_BIH_ip, corner) for corner in corners])
	return num_inliers


def messy_grid_ransac (messy_horz_grid, messy_vert_grid, all_corners, image, num_iterations=10000):
	"""
		Function: messy_grid_ransac
		---------------------------
		given messy horizontal/vertical grids, this runs ransac in 
		order to find a point correspondance that best matches 
		the corners that we originally found 
		all_corners = the set of all corners, not just those that snapped 
		to lines
	"""

	#=====[ Step 1: divide image points into grids	]=====
	corners, quadrants_ip = get_quadrants_ip (messy_horz_grid, messy_vert_grid)
	#####[ DEBUG: draw the quadrants	]#####
	cv2.namedWindow ('quadrants_ip')
	image = draw_points_xy (image, quadrants_ip['tl'])
	cv2.imshow ('quadrants_ip', image)
	key = 0
	while key != 27:
		key = cv2.waitKey (30)
	image = draw_points_xy (image, quadrants_ip['tr'])
	cv2.imshow ('quadrants_ip', image)
	key = 0
	while key != 27:
		key = cv2.waitKey (30)
	image = draw_points_xy (image, quadrants_ip['br'])
	cv2.imshow ('quadrants_ip', image)
	key = 0
	while key != 27:
		key = cv2.waitKey (30)
	image = draw_points_xy (image, quadrants_ip['bl'])
	cv2.imshow ('quadrants_ip', image)
	key = 0
	while key != 27:
		key = cv2.waitKey (30)

	#==========[ RANSAC ITERATIONS	]==========
	print '=====[ 	BEGIN ]====='
	best_num_inliers = -1
	best_BIH = np.zeros((3,3))
	for i in range(num_iterations):
		print i

		#=====[ Step 1: sample 4 board points, one from each quadrant	]=====
		board_points = sample_from_quadrants (qbp)

		#=====[ Step 2: sample 4 image points, one from each quadrant	]=====
		image_points = sample_from_quadrants (quadrants_ip)
		#####[ DEBUG: draw image points	]#####
		# cv2.namedWindow ('image_points')
		# image = draw_points_xy (image, image_points)
		# cv2.imshow ('image_points', image)
		# key = 0
		# while key != 27:
		# 	key = cv2.waitKey (30)

		#=====[ Step 3: compute BIH	]=====
		# print '	', board_points
		# print '	', image_points
		BIH = point_correspondences_to_BIH (board_points, image_points)

		#=====[ Step 4: compute number of inliers	]=====
		num_inliers = compute_inliers (BIH, corners)
		if num_inliers > best_num_inliers:
			best_num_inliers = num_inliers
			best_BIH = BIH

		if best_num_inliers >= 20:
			return BIH

	print '=====[ END ]====='


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


	BIH = get_chessboard_lines (corners, image)
	return BIH










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






















