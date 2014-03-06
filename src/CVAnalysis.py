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
#################[ --- FINDING POINT CORRESPONDENCES FROM IMAGE --- ]###############################
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


def cluster_lines (lines, num_clusters):
	"""
		Function: cluster_lines
		-----------------------
		given a list of lines represented as (a, b, c), this will
		this will return a list of lines with clusters compressed 
		to their centroid 
		comparison_function: returns true if two lines are in a cluster
	"""


	#=====[ Step 1: init old_lines, new_lines	]=====
	old_lines = np.array (lines)
	new_lines = []

	#=====[ ITERATE OVER OLD_LINES	]=====
	while len(old_lines) > 0:

		l1 = old_lines [0]
		idx = np.array([comparison_function(l1, l2) for l2 in old_lines])
		lines_cluster = old_lines[idx]
		new_line = get_centroid (lines_cluster)
		new_lines.append (new_line)
		old_lines = old_lines[np.invert(idx)]

	return new_lines


def get_chessboard_corner_candidates (image, corner_classifier):
	"""
		Function: get_chessboard_corner_candidates
		------------------------------------------
		given an image, returns a list of points (represented as tuples)
		that correspond to possible chessboard corners 
	"""
	#=====[ Step 1: get corners/sift descriptors	]=====
	hc = get_harris_corners(image)
	sd = get_sift_descriptors (image, hc)

	#=====[ Step 2: make predictions	]=====
	predictions = corner_classifier.predict (sd)
	idx = (predictions == 1)
	chessboard_corners = [c.pt for c, i in zip(hc, idx) if i]

	#=====[ Step 3: cluster corners	]=====
	chessboard_corners = cluster_points (chessboard_corners)

	return chessboard_corners


def filter_by_slope (lines, slope_predicate):
	"""
		Function: filter_by_slope
		-------------------------
		given a list of lines in (a, b, c) format, this will return 
		another list of lines in same format where all pass 'true' on 
		'slope_predicate'
	"""
	slopes = [abs(l[0]/l[1]) if l[1] != 0 else 10000 for l in lines]
	idx = np.array([slope_predicate (s) for s in slopes])
	return [l for l, i in zip(lines, idx) if i]


def avg_close_lines (lines_list):
	"""
		Function: avg_close_points 
		--------------------------
		given a list of keypoints, this returns another list of 
		(x, y) pairs for points that are very close 
	"""
	lines = [(rho, theta) for rho, theta in lines_list]

	#=====[ Step 1: get points out of each one	]=====
	old_lines = np.array(lines)

	#=====[ Step 2: get new_points	]=====
	new_lines = []
	while len(old_lines	) > 1:
		l1 = old_lines [0]
		distances = np.array([abs(l1[1] - l2[1]) for l2 in old_lines])
		idx = (distances < 0.1)
		lines_cluster = old_lines[idx]
		new_line = get_centroid (lines_cluster)
		new_lines.append (new_line)
		old_lines = old_lines[np.invert(idx)]

	return new_lines


def avg_close_lines_2 (lines_list):
	"""
		Function: avg_close_points 
		--------------------------
		given a list of keypoints, this returns another list of 
		(x, y) pairs for points that are very close 
	"""
	lines = [(rho, theta) for rho, theta in lines_list]

	#=====[ Step 1: get points out of each one	]=====
	old_lines = np.array(lines)

	#=====[ Step 2: get new_points	]=====
	new_lines = []
	while len(old_lines	) > 1:
		l1 = old_lines [0]
		distances = np.array([abs(l1[0] - l2[0]) for l2 in old_lines])
		idx = (distances < 10)
		lines_cluster = old_lines[idx]
		new_line = get_centroid (lines_cluster)
		new_lines.append (new_line)
		old_lines = old_lines[np.invert(idx)]

	return new_lines


def snap_points_to_lines (lines, points, max_distance=8):
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


def get_chessboard_lines (corners, image):
	"""
		Function: get_chessboard_lines
		------------------------------
		given a list of corners represented as tuples, this returns 
		(horizontal_lines, vertical_lines) represented as (a, b, c)
		pairs 
	"""
	#=====[ Step 1: get lines via Hough transform on corners ]=====
	corners_img = np.zeros (image.shape[:2], dtype=np.uint8)
	for corner in corners:
		corners_img[int(corner[1])][int(corner[0])] = 255
	lines = cv2.HoughLines (corners_img, 3, np.pi/180, 6)[0]
	lines = [rho_theta_to_abc (l) for l in lines]

	#=====[ Step 2: get vertical lines	]=====
	v_rh = [abc_to_rho_theta (l) for l in lines]
	v_rh = avg_close_lines (v_rh)
	vert_lines = [rho_theta_to_abc(v) for v in v_rh]
	vert_lines = filter_by_slope (vert_lines, lambda slope: (slope > 1.7))
	v_rh = [abc_to_rho_theta (l) for l in vert_lines]


	#=====[ Step 3: snap points to grid ]===
	points_grid = snap_points_to_lines (v_rh, corners)

	#=====[ Step 6: hough transform on points in grid to get horizontal lines	]=====
	all_points = [p for l in points_grid for p in l]
	corners_img = np.zeros (image.shape[:2], dtype=np.uint8)
	for p in all_points:
		corners_img[int(p[1])][int(p[0])] = 255
	lines = cv2.HoughLines (corners_img, 3, np.pi/180, 2)[0]
	lines = [rho_theta_to_abc (l) for l in lines]
	horz_lines = filter_by_slope (lines, lambda slope: (slope < 0.1))
	lines = [abc_to_rho_theta(l) for l in horz_lines]
	horz_lines = avg_close_lines_2 (lines)
	horz_lines = [rho_theta_to_abc(l) for l in horz_lines]

	# print '=====[ 	X intercepts]====='
	# x_intercepts = [[get_screen_bottom_intercept (l, image.shape[0])] for l in vert_lines]

	# print x_intercepts

	# km_h = KMeans(n_clusters=9)
	# km_v = KMeans(n_clusters=5)

	# h_idx = km_h.fit_predict (h_rhos)
	# v_idx = km_v.fit_predict (x_intercepts)

	# h_centroids = []
	# for i in range(9):
	# 	print "===[ cluster ", i, " ]==="
	# 	points = np.array([h for h, j in zip(h_rh, h_idx) if (j == i)])
	# 	print points
	# 	print np.mean (points, axis=0)
	# 	h_centroids.append (np.mean (points, axis=0))

	# v_centroids = []
	# for i in range(5):
		# print "===[ cluster ", i, " ]==="
		# points = np.array([h for h, j in zip(vert_lines, v_idx) if (j == i)])
		# ba_ratio = np.divide(points[:, 1], points[:, 0])
		# ca_ratio = np.divide(points[:, 2], points[:, 0])
		# v_centroids.append ((np.mean(ba_ratio), np.mean(ca_ratio)))

	# v_centroids = [(1, v[0],  v[1]) for v in v_centroids]
	# print v_centroids

	# horz_lines = [rho_theta_to_abc (l) for l in h_rh]
	vert_lines = [rho_theta_to_abc (l) for l in v_rh]

	return horz_lines, vert_lines



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


def find_point_correspondences (horz_lines, vert_lines, corners):
	"""
		Function: find_point_correspondences
		------------------------------------
		given horizontal lines, vertical lines (in abc) and a set of corners,
		this will return image_points, board_points, where the 
		two correspond
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
	horz_grid = snap_points_to_lines (horz_lines, grid_points)
	vert_grid = snap_points_to_lines (vert_lines, grid_points)
	horz_grid = sort_point_grid (horz_grid, 1)
	vert_grid = sort_point_grid (vert_grid, 0)

	#=====[ Step 4: extract point correspondences	]=====
	return extract_point_correspondences (horz_grid, vert_grid, grid_points)



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
	#=====[ Step 1: get chessboard corners	]=====
	corners = get_chessboard_corner_candidates (image, corner_classifier)

	#=====[ Step 2: get horizontal, vertical lines (as abc)	]=====
	horz_lines, vert_lines = get_chessboard_lines (corners, image)

	#=====[ Step 4: find point correspondences	]=====
	image_points, board_points = find_point_correspondences (horz_lines, vert_lines, corners)

	return image_points, board_points



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
##############################[ --- AUTOMATICALLY FINDING BOARD FROM IMAGE --- ]####################
####################################################################################################

def get_corner_features (harris_corner, sift_descriptor):
	"""
		Function: get_corner_features
		-----------------------------
		given a keypoint representing a harris corner and a sift 
		descriptor that describes it, this returns a feature 
		vector for it 
	"""
	return concatenate ([[harris_corner.pt[0]], [harris_corner.pt[1]], sift_descriptor])


def get_BIH_from_image (image, corner_classifier):
	"""
		Function: get_BIH_from_image
		----------------------------
		given an image, this will return the most likely BIH from it 
		Note: assumes image is in its final format
	"""
	#=====[ Step 1: get all harris corners, sift descriptors	]=====
	harris_corners = get_harris_corners (image)
	sift_descriptors = get_sift_descriptors (image, harris_corners)

	#=====[ Step 2: get feature representations for each hc	]====
	features = [get_corner_features (c, d) for c, d in zip(harris_corners, sift_descriptors)]

	#=====[ Step 3: get p(corner|features) for each one ]=====
	corner_probs = corner_classifier.predict_proba (features)









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






















