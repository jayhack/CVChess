import os
import pickle
import cv2
import numpy as np
from Board import Board

def get_oc_name (n):
	return str(n) + '.oc'

def get_image_name (n):
	return str(n) + '.jpg'


if __name__ == "__main__":

	data_dir = '../data/p1'
	num_images = 14

	#=====[ Step 1: switch to the data directory	]=====
	os.chdir (data_dir)

	#=====[ Step 2: load the BIH	]=====
	BIH = pickle.load (open('BIH.mat', 'r'))

	#=====[ Step 3: get list of all boards with occlusions	]=====
	boards = []
	for n in range(1, num_images + 1):
		image_name, oc_name = get_image_name (n), get_oc_name (n)
		print image_name, oc_name

		image = cv2.imread (image_name)
		board = Board (name=str(n), image=image, BIH=BIH)
		board.add_occlusions (oc_name)
		boards.append (board)


	#=====[ Step 4: get all Xs, ys	]=====
	Xs, ys = [], []
	for board in boards:
		X, y = board.get_occlusion_features ()
		Xs.append (X)
		ys.append (y)

	#=====[ Step 5: get X, y	]=====
	X = np.concatenate (Xs, 0)
	y = np.concatenate (ys)

	#=====[ Step 6: save as features.mat, labels.mat	]=====
	pickle.dump (X, open('features.mat', 'w'))
	pickle.dump (y, open('labels.mat', 'w'))

