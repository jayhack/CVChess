import cv2
import numpy as np
from util import *
from copy import deepcopy

def get_horizontal_lines (img):

	#=====[ Step 1: set parameters ]=====
	num_peaks = 5
	theta_buckets_horz = [-90, -89]
	theta_resolution_horz = 0.0175 #raidans
	rho_resolution_horz = 6
	threshold_horz = 5

	#=====[ Step 2: find lines in (rho, theta)	]=====
	# [H, theta, rho] = hough (corners_img, 'Theta', theta_buckets_horz, 'RhoResolution', rho_resolution_horz);
	# peaks = houghpeaks(H, num_peaks);
	lines_rt = cv2.HoughLines (deepcopy(img), rho_resolution_horz, theta_resolution_horz, threshold_horz)[0]
	print lines_rt
	#####[ DEBUG: draw lines in (rho, theta)	]#####
	img = draw_lines_rho_theta (img , lines_rt)
	cv2.imshow ('HORIZONTAL LINES', img)
	key = 0
	while key != 27:
		key = cv2.waitKey (30)

	#=====[ Step 3: convert peaks to rho, theta	]=====
	# theta_rad = fromDegrees ('radians', theta);
	# rhos = rho(peaks(:, 1));
	# thetas = theta_rad(peaks(:, 2));
	# lines = [rhos; thetas];

	#=====[ Step 4: figure out which lines they are	]=====
	# indexed_lines = horizontal_ransac (lines);

	#####[ DEBUG: show lines	]#####
	# draw_lines (corners_img, indexed_lines(1:2, :));

def get_vertical_lines (img):
	pass

if __name__ == "__main__":

	corners_img_name = './IPC/corners.png';
	corners_img = cv2.cvtColor(cv2.imread (corners_img_name), cv2.COLOR_BGR2GRAY);

	#=====[ Step 2: get horizontal/vertical lines, along with indices up to a shift	]=====
	horizontal_lines = get_horizontal_lines (corners_img);
	# vertical_lines = get_vertical_lines (corners_img);