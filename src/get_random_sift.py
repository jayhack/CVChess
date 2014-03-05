import os
import pickle
import numpy as np 
import cv2
import CVAnalysis

if __name__ == "__main__":

	#=====[ Step 1: switch to directory	]=====
	os.chdir ('../data/toy_images')

	filenames = [	
					'random1.jpg',
					'random2.jpg',
					'random3.jpg',
					'random4.jpg',
					'random5.jpg',
					'random6.jpg',
					'random7.jpg',
	]

	images = [cv2.imread (f) for f in filenames]

	sift_desc = []
	for image in images:
		hc = CVAnalysis.get_harris_corners (image)
		desc = CVAnalysis.get_sift_descriptors (image, hc)
		sift_desc.append (desc)

	features = np.concatenate (sift_desc, 0)

	