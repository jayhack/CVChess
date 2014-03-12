from itertools import tee, izip
import cv2
import numpy as np
from numpy import matrix, array, dot, transpose
from numpy.linalg import inv

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
##############################[ --- POINTS --- ]####################################################
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


def euclidean_distance (p1, p2):
	"""
		Function: euclidean_distance 
		----------------------------
		given two points as 2-tuples, returns euclidean distance 
		between them
	"""
	assert ((len(p1) == len(p2)) and (len(p1) == 2))
	return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_centroid (points):
	"""
		Function: get_centroid 
		----------------------
		given a list of points represented 
		as 2-tuples, returns their centroid 
	"""
	return (np.mean([p[0] for p in points]), np.mean([p[1] for p in points]))



####################################################################################################
##############################[ --- LINES --- ]#####################################################
####################################################################################################


def rho_theta_to_abc (line):
	"""
		Function: rho_theta_to_abc
		--------------------------
		given a line represented as (rho, theta),
		this returns the triplet (a, b, c) such that 
		ax + by + c = 0 for all points (x, y) on the 
		line
	"""
	rho, theta = line[0], line[1]
	return (np.cos(theta), np.sin(theta), -rho)


def abc_to_rho_theta (line):
	"""
		Function: abc_to_rho_theta
		--------------------------
		inverse of rho_theta_to_abc
	"""
	a, b, c = line[0], line[1], line[2]
	rho, theta = -c, np.arccos(a)
	return rho, theta


def get_screen_bottom_intercept (l, y_screen_bottom):
	"""
		Function: get_screen_bottom_intercept
		-------------------------------------
		given lines in (a, b, c), this returns their intercept 
		coordinates on the bottom of the screen 
	"""
	return (-l[1]/l[0])*y_screen_bottom - (l[2]/l[0])


def get_line_point_distance (line, point):
	"""
		Function: get_line_point_distance 
		---------------------------------
		returns the distance from the point to the line 
		assumes line is represented as (rho,theta), point 
		is (x, y)
	"""
	assert len(line) == 2
	assert len(point) == 2
	a, b, c = rho_theta_to_abc (line)
	x, y = point[0], point[1]
	return np.abs(a*x + b*y + c)/np.sqrt(a**2 + b**2)


def get_line_intersection (l1, l2):
	"""
		Function: get_line_intersection
		-------------------------------
		given two lines represented as (a,b,c), this will
		return their intersection
	"""
	return np.cross (l1, l2)

def get_parallel_lines (lines):
	"""
		Function: get_parallel_lines
		----------------------------
		given a set of lines in (a,b,c) format, this will
		return the largest set of them that are parallel in 
		3-space. (this means they intersect at the same point)
	"""
	def pairwise(iterable):
		a, b = tee(iterable)
		next(b, None)
		return izip(a, b)


	#=====[ Step 1: get all pairs of line intersections	]=====
	for l1, l2 in pairwize(lines):
		pass









####################################################################################################
##############################[ --- DRAWING UTILITIES --- ]#########################################
####################################################################################################

def draw_points_xy (image, points, color=(0, 0, 255), radius=5):
	"""
		Function: draw_points
		---------------------
		given a list of points represented as (x, y), this draws circles
		around them on the provided image in the provided color
	"""
	for p in points:
		cv2.circle (image, (int(p[0]), int(p[1])), radius, color)
	return image


def draw_lines_rho_theta (image, lines, color=(0, 0, 255)):
	"""
		Function: draw_lines_rho_theta
		------------------------------
		draws the provided lines, represented as (rho, theta),
		on the provided image according to the color.
		returns the modified image.
	"""
	for rho, theta in lines:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
		y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
		x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
		y2 = int(y0 - 1000*(a))
		cv2.line(image,(x1,y1),(x2,y2),color,2)
	return image


def draw_lines_abc (image, lines, color=(0, 0, 255)):
	"""
		Function: draw_lines_abc
		------------------------
		draws the provided lines, represented as (a, b, c),
		on the provided image according to the color.
		returns the modified image.
	"""
	rt_lines = [abc_to_rho_theta (l) for l in lines]
	return draw_lines_rho_theta (image, rt_lines, color)










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


def print_header (header_text):	
	"""
		Function: print_header
		----------------------
		prints the specified header text in a salient way
	"""
	print "-" * (len(header_text) + 12)
	print ('#' * 5) + ' ' +  header_text + ' ' + ('#' * 5)
	print "-" * (len(header_text) + 12)




