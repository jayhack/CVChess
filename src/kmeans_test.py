import pickle
import cv2
import numpy as np
from sklearn.cluster import KMeans
from Board import Board
from util import *
from copy import deepcopy

def preprocess_frame (frame):
	return frame [(frame.shape[0]/2):, :]

def annotate_image (image, km):
	km_image = np.zeros (image.shape)
	print km_image.shape
	for i in range(image.shape[0]):
		for j in range (image.shape[1]):
			if km.predict (image[i][j]) == 0:
				km_image[i][j] = (0, 0, 255)
			if km.predict (image[i][j]) == 1:
				km_image[i][j] = (0, 255, 0)
			if km.predict (image[i][j]) == 2:
				km_image[i][j] = (255, 0, 0)
	return km_image

if __name__ == '__main__':

	#=====[ Step 1: get images ]=====
	img_empty = cv2.imread ('../data/videos/0.jpg')
	img_1 = cv2.imread ('../data/videos/1.jpg')
	img_2 = cv2.imread ('../data/videos/2.jpg')

	#=====[ Step 2: initialize board	]=====
	corner_classifier_filename = '../data/classifiers/corner_classifier.clf'
	corner_classifier = pickle.load (open(corner_classifier_filename, 'r'))	#more data
	board = Board(corner_classifier=corner_classifier)

	#=====[ Step 3: add first frame	]=====
	board.add_frame (img_empty)	
	#####[ DEBUG: verify BIH is correct	]#####
	# img = board.draw_vertices(img_empty)
	# cv2.imshow ('BIH MARKED', img)
	# key = 0
	# while not key in [27, ord('Q'), ord('q')]: 
	# 	key = cv2.waitKey (30)

	#=====[ Step 4: add second frame	]=====
	board.add_frame (img_1)
	board.get_occlusion_changes ()

	#=====[ Step 5: get square representations ]=====
	s_reg = [s.image_region for s in board.iter_squares ()]
	s_norm = [s.image_region_normalized for s in board.iter_squares ()]
	s_hsv = [cv2.cvtColor(s.image_region, cv2.COLOR_BGR2HSV) for s in board.iter_squares()]

	#=====[ Step 6: reshape all data	]=====
	reshaped = [s.reshape ((s.shape[0]*s.shape[1], 3)) for s in s_reg]
	data = np.concatenate (reshaped, 0)
	print data.shape

	#=====[ Step 7: fit kmeans	]=====
	km = KMeans (n_clusters=4)
	km.fit (data)
	print km.cluster_centers_

	#=====[ Step 8: annotate some images	]=====
	a0 = annotate_image (s_reg[0], km)
	a30 = annotate_image (s_reg[30], km)
	a50 = annotate_image (s_reg[50], km)
	cv2.imshow ('s0', s_reg[0])
	cv2.imshow ('a0', a0)
	cv2.imshow ('s30', s_reg[30])
	cv2.imshow ('a30', a30)
	cv2.imshow ('s50', s_reg[50])				
	cv2.imshow ('a50', a50)








