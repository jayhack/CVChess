#!/opt/local/bin/python
import cv2

if __name__ == "__main__":

	#==========[ Step 1: load in the image	]==========
	image = cv2.imread ('../data/basic_board.jpg')
	square_corners = {
		1: (229, 107),
		2: (274.5, 107),
		3: (275.5, 130.3),
		4: (228, 131.4)
	}


	#==========[ display image	]==========
	cv2.namedWindow ('DISPLAY')
	cv2.imshow ('DISPLAY', image)
	raw_input ('Enter to close')

	# #=====[ Step 1: open the video capture ]=====
	# vc = cv2.VideoCapture ()
	# vc.open (0)

	# #=====[ Step 2: grab the frame ]=====
	# for i in range (10):
	# 	vc.grab ()
	# 	frame = vc.retrieve ()[1]

	# #=====[ Step 3: display ]=====
	# cv2.namedWindow ('DISPLAY')
	# cv2.imshow ('DISPLAY', frame)


	# #=====[ Step 4: close camera ]=====
	# vc.release ()


