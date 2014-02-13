import os
import sys
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from BoardImage import BoardImage


def onclick(event):
	"""
		Function: onclick
		-----------------
		callback for clicking the image;
		has you input the board coords of the point you just clicked
	"""
	print "=====[ Enter board coordinates of point you just clicked: ]====="
	board_x = raw_input ('>>> x: ')
	board_y = raw_input ('>>> y: ')
	image_point = (event.x, event.y)
	board_point = (board_x, board_y)
	board_image.add_point_correspondance (board_point=board_point, image_point=image_point)
	print "Stored as: "
	print "	- board_point: ", board_point
	print "	- image_point: ", image_point


def onpress (event):
	"""
		Function: press
		---------------
		callback for user pressing keys;
		user enters esc -> this quits
	"""
	#=====[ Step 1: verify key	]=====
	if not key == 'escape':
		return

	#=====[ Step 2: save BoardImage ]=====
	print "---> Saving image"
	board_image = BoardImage (name=image_name, image=image, board_points=board_points, image_points=image_points)
	board_image.save (image_name + '.BoardImage')

	#=====[ Step 3: exit	]=====
	exit ()



if __name__ == "__main__":

	#==========[ Step 1: sanitize input 	]==========
	if not len(sys.argv) == 2:
		raise StandardError ("Enter only the filepath to the image you want to mark")
	filename 	= sys.argv[1]
	image_name 	= os.path.split (filename)
	if not os.path.isfile (filename):
		raise StandardError ("Couldn't find the file you passed in...")

	#==========[ Step 2: get/show image	]==========
	image = imread(filename)

	fig, ax = plt.subplots ()
	implot = plt.imshow (image)

	#==========[ Step 3: set callbacks	]==========
	mouseclick 	= fig.canvas.mpl_connect('button_press_event', onclick)
	keypress 	= fig.canvas.mpl_connect ('key_press_event', onpress)


	#==========[ Step 3: construct BoardImage	]==========
	board_image = BoardImage (image=image, name=image_name, board_points=[], image_points=[])
	plt.show ()



