#=====[ PIL	]=====
import Image, ImageDraw

#=====[ numpy/scipy	]=====
import numpy as np
from numpy import transpose, dot
from numpy.linalg import inv

#=====[ our modules	]=====
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
	xls = inv(transpose(P) * P) * transpose(P) * y
	# xls = extract_row(np.dot(np.linalg.pinv (P), y).flatten(), 0)

	H = mat_from_array (xls)

	return H
	# return np.transpose(H)


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
	image = Image.open ('../data/micahboard/1.jpg')

	board_points = np.matrix([	
								# (3, 0)
								[3, 0, 1],
								[4, 0, 1],
								[4, 1, 1],
								[3, 1, 1],

								# (3, 3)
								[3, 3, 1],
								[4, 3, 1],
								[4, 4, 1],
								[3, 4, 1],

								[4, 5, 1],
								[5, 5, 1],
								[5, 6, 1],
								[4, 6, 1],
								
								# (0, 7)
								[0, 7, 1],
								[1, 7, 1],
								[1, 8, 1],
								[0, 8, 1],

								# (6, 7)
								[6, 7, 1],
								[7, 7, 1],
								[7, 8, 1],
								[6, 8, 1]


							])

	image_points = np.matrix([	


								[530, 	504, 1],
								[580, 	504.5, 1],
								[583, 	518, 1],
								[531, 	518, 1],

								[534.5, 552, 1],
								[593.3,	552, 1],
								[594.5, 567, 1],
								[534.5, 570.75, 1],

								[603.5, 593, 1],
								[668.5, 592, 1],
								[678.5, 614, 1],
								[610, 	615, 1],

								[309.5, 649.5, 1],
								[387.06, 647, 1],
								[377.062, 678.25, 1],
								[294.56, 679.5, 1],	

								[767, 642, 1],
								[840.8, 642, 1],
								[865, 672, 1],
								[787, 672, 1]
							])

	#==========[ For "ABOVE"	]==========
	# image = Image.open ('../data/above.jpg')
	# board_points = np.matrix( [
	# 							[0,0,1],
	# 							[1,1,1],
	# 							[3,1,1],
	# 							[4,2,1],
	# 							[8,7,1]
	# 						])

	# image_points = np.matrix( [
	# 							[60,58,1],
	# 							[170,170,1],
	# 							[391,169,1],
	# 							[502,279,1],
	# 							[949,839.5,1]
	# 						])


	#==========[ Step 2: find board homography, H]==========
	H = find_board_homography (board_points, image_points)
	print H


	#==========[ Step 3: construct the board	]==========
	board = Board (H)


	#==========[ Step 5: draw squares on image	]==========
	drawer = ImageDraw.Draw (image)
	board.draw_vertices (drawer)


	#==========[ Step 6: display image	]==========
	image.show ()
	image.save('above_vertices_marked.jpg')




	