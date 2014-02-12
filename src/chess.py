import Image, ImageDraw
import numpy as np
from Board import Board


def extract_row (mat, row):
	""" 
		Function: extract_row
		---------------------
		given a matrix and a row, returns the row as a numpy array 
	"""
	return np.asarray (mat[row, :])[0]


def extract_col (mat, col):
	""" 
		Function: extract_col
		---------------------
		given a matrix and a col, returns the row as a numpy array 
	"""
	return np.asarray (mat[:, col])


def get_pw_matrix_rows (pw, pi):
	"""
		Function: get_pw_matrix_rows
		----------------------------
		goes from a point in the world to its coordinates in P
	"""
	#=====[ Step 1: make sure they are column vectors	]=====
	assert len(pw) == 3

	#=====[ Step 2: get matrix rows for them	]=====
	row1 = np.matrix([pw[0], pw[1], pw[2], 0, 0, 0, 0, 0, 0])
	row2 = np.matrix([0, 0, 0, pw[0], pw[1], pw[2], 0, 0, 0])
	row3 = np.matrix([0, 0, 0, 0, 0, 0, pw[0], pw[1], pw[2]])
	return np.concatenate ([row1, row2, row3], 0);


def get_P (world_points):
	"""
		Function: get_P 
		---------------
		given the set of all world points, this returns the matrix P s.t.
		P*a = y
		where a = columnized version of H, y = columnized version of image points
	"""
	#=====[ Step 1: check shape	]=====
	assert world_points.shape[1] == 3

	#=====[ Step 2: construct P	]=====
	num_points = world_points.shape[0]
	P = np.matrix (np.zeros ((3*num_points, 9)))
	for row in range (num_points):
		pw 					= extract_row (world_points, row)
		matrix_rows 		= get_pw_matrix_rows (pw, row)
		P[row*3:row*3+3, :] = matrix_rows
	return P


def get_y (image_points):
	"""
		Function: get_y
		---------------
		given all image points, this returns a columnized version 'y',
		such that the n*3th entry is the x coordinate of the nth point, etc.
	"""
	#=====[ Step 1: check shape	]=====
	assert image_points.shape[1] == 3
	
	#=====[ Step 2: construct y	]=====
	return np.transpose(image_points.flatten())
		

def mat_from_array (a):
	"""
		Function: mat_from_array
		------------------------
		given an, this will assemble a matrix from it 
		as follows:
		c = [c1, c2 ... c9, c10] 
		=> mat = [	c1 c2 c2 ]
				 [	c4 c5 c6 ]	
				 [	c7 c8 c9 ]
	"""
	#=====[ Step 1: check shape	]=====
	assert len(a)	== 9

	#=====[ Step 2: assemble/return matrix	]=====
	return np.concatenate ([np.matrix(a[0:3]), np.matrix(a[3:6]), np.matrix(a[6:9])], 0);


def find_board_homography (world_points, image_points):
	"""
		Function: find_board_homography
		-------------------------------
		given a list of corresponding points in the world and in the image,
		this will return the "board homography" H s.t. 
			H * pw = pi
		Note: world_points and image_points store each point in a row
	"""
	#=====[ Step 1: make sure shape is valid	]=====
	assert world_points.shape == image_points.shape
	assert world_points.shape[1] == 3

	#=====[ Step 2: get P and y ]=====
	P = get_P (world_points)
	y = get_y (image_points)

	#=====[ Step 3: find H from Pmat	]=====
	xls = extract_row(np.dot(np.linalg.pinv (P), y).flatten(), 0)
	H = mat_from_array (xls)

	return H


def world_to_image_coords (H, pw):
	"""
		Function: world_to_image_coords
		-------------------------------
		given a homography matrix and a point in the world, this will return
		its (homogenous) coordinates in the image 
	"""
	return np.dot (H, pw)



if __name__ == "__main__":

	#==========[ Step 1: load image/mark the points	]==========
	image = Image.open ('../data/basic_board.jpg')

	image_points =  np.matrix(	[ 	
									[228, 107, 1],			# top left		
									[228, 131.4, 1],		# bottom left
							 		[274.5, 107, 1],		# top right
									[275.5, 130.3, 1],		# bottom right
									[230, 84, 1],			# top top left
									[273, 84, 1],			# top top right
								])

	world_points = 	np.matrix ( [
									[4, 4, 1],
									[4, 5, 1],
									[5, 4, 1],
									[5, 5, 1],
									[4, 3, 1],
									[5, 3, 1]
								])
	world_points 




	#==========[ Step 2: find board homography, H]==========
	H = find_board_homography (world_points, image_points)

	#==========[ Step 3: construct the board	]==========
	b = Board (H)

	