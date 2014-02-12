import Image, ImageDraw
import numpy as np

if __name__ == "__main__":

	#==========[ Step 1: load in the image	]==========
	image = Image.open ('../data/basic_board.jpg')

	#==========[ Step 2: draw in box	]==========
	draw = ImageDraw.Draw (image)
	square_corners = [
		(228, 107),			# top left
		(274.5, 107),		# top right
		(275.5, 130.3),		# bottom right
		(228, 131.4),		# bottom left
		(230, 84),			# top top left
		(273, 84)			# top top right
	]

	for i in range(4):
		corner_1 = square_corners[i]
		corner_2 = square_corners[(i+1) % 4]
		draw.line((corner_1, corner_2), fill=(0, 0, 255), width=3)


	#==========[ Step 3: show image	]==========
	image.show ()
	