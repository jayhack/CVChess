import cv2
import numpy as np 

def show_image (image, title):
	cv2.imshow (title, cv2.pyrUp(cv2.pyrUp(image)))

if __name__ == "__main__":

	#=====[ Step 1: load in images	]=====
	k_img = cv2.imread ('knight_base.png')
	p_img = cv2.imread ('pawn_base.png')	
	c1_img = cv2.imread ('combo1.png')
	c2_img = cv2.imread ('combo2.png')

	#=====[ Step 2: run kmeans	]=====
	show_image (k_img, 'knight')
	# show_image (p_img, 'pawn')
	show_image (c1_img, 'combo1')
	# show_image (c2_img, 'combo2')			
