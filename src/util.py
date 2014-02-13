from numpy import matrix, array, dot, transpose
from numpy.linalg import inv

####################################################################################################
##############################[ --- LINEAR ALGEBRA --- ]############################################
####################################################################################################

def homogenize (x):
	"""
		Function: homogenize
		--------------------
		given a tuple, this will return a numpy array representing it in
		homogenous coordinates
	"""
	return transpose(matrix ((x[0], x[1], 1)))

def dehomogenize (x):
	"""
		Function: dehomogenize
		----------------------
		given a homogenous vector, this returns a tuple
	"""
	x = list(array(x).reshape(-1,))
	return (x[0]/x[2], x[1]/x[2])

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




####################################################################################################
##############################[ --- COORDINATE CONVERSIONS --- ]####################################
####################################################################################################

def board_to_image_coords (BIH, board_coords):
	"""
		FUNCTION: board_to_image_coords
		------------------------------
		BIH: board to image projective transformation
		board_coords: coordinates of point in board coordinates 
		
		returns: image coordinates of the same point
	"""
	return dehomogenize(BIH * homogenize(board_coords))


def image_to_board_coords (BIH, image_coords):
	"""
		FUNCTION: image_to_board_coords
		-------------------------------
		BIH: board to image projective transformation
		image_coords: coordinates of point in board coordinates 

		returns: board coordinates of the same point
	"""
	return dehomogenize(inv(BIH) * homogenize(image_coords))


an_letters, an_numbers = 'ABCDEFGH', range(1, 9)
def algebraic_notation_to_board_coords (an):
	"""
		Function: algebraic_notation_to_board_coords
		--------------------------------------------
		given algebraic notation of a square, this returns its board
		coordinates in clockwise fashion from the top left
	"""	
	tl = an_letters.index (an[0]), an_numbers.index (an[1])
	return [tl, (tl[0] + 1, tl[1]), (tl[0] + 1, tl[1] + 1), (tl[0], tl[1] + 1)]




####################################################################################################
##############################[ --- COMMON GENERATORS --- ]#########################################
####################################################################################################

def iter_algebraic_notations ():
	"""
		Generator: iter_squares_algebraic_notation
		------------------------------------------
		iterates over all board squares, returning their algebraic 
		notations
	"""
	for l in an_letters:
		for n in an_numbers:
			yield (l, n)


def iter_board_vertices ():
	"""
		Generator: iter_board_vertices
		------------------------------
		iterates over all vertices on the board, starting from top left
		of square (A, 1), returning them in board coordinates
	"""
	for i in range (8):
		for j in range (9):
			return (i, j)


####################################################################################################
##############################[ --- INTERFACE --- ]#################################################
####################################################################################################

def print_welcome ():
	"""
		Function: print_welcome
		-----------------------
		prints out a welcome message
	"""
	print "##########################################################################"
	print "####################[ CVChess: CV for board games 	  	]################"
	print "####################[ --------------------------------- 	]################"	
	print "####################[ Jay Hack and Prithvi Ramakrishnam 	]################"
	print "####################[ CS 231A, Winter 2014 				]################"
	print "##########################################################################"
	print "\n"


def print_message (message):
	"""
		Function: print_message
		-----------------------
		prints the specified message in a salient format
	"""
	print "-" * len(message)
	print message
	print "-" * len(message)


def print_status (stage, status):
	"""
		Function: print_status
		----------------------
		prints out a status message 
	"""
	print "-----> " + stage + ": " + status




