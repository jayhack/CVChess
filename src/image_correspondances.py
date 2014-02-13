# ---------------------------------------------------------------------- #
# FILE: image_correspondances
# ---------------------------
# contains correspondances between images we already have and, 
# in addition, a utilitiy for marking new images
# 
# =====[ Board Coordinates Conventions	]=====
# - refer to *vertices* (corners) on the board
# - represented as (x, y) pairs, where (0, 0) is the top left square from
#	the camera's point of view; this corresponds to square A1
#	- that is, 
#
# =====[ 	]=====
# ---------------------------------------------------------------------- #


from scipy.misc import imread
from numpy import matrix


board_points = matrix(	[	
							[3, 0, 1],
							[4, 0, 1],
							[4, 1, 1],
							[3, 1, 1],

							[3, 3, 1],
							[4, 3, 1],
							[4, 4, 1],
							[3, 4, 1],

							[4, 5, 1],
							[5, 5, 1],
							[5, 6, 1],
							[4, 6, 1],
							
							[0, 7, 1],
							[1, 7, 1],
							[1, 8, 1],
							[0, 8, 1],

							[6, 7, 1],
							[7, 7, 1],
							[7, 8, 1],
							[6, 8, 1]


						])

image_points = matrix(	[	


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