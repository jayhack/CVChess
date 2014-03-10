from copy import deepcopy
import cv2
import numpy as np
import sklearn
import pickle
import CVAnalysis
from Board import Board




if __name__ == '__main__':

	#=====[ Step 1: read in image, classifier ]=====
	first_frame = cv2.imread ('../data/p1/0.jpg')
	# first_frame = cv2.imread ('../data/empty1/1.jpg')
	im_size = first_frame.shape[0]
	first_frame = first_frame[int(im_size/2):, :]

	# corner_classifier = pickle.load (open('../data/classifiers/corner_classifier.clf', 'r'))
	corner_classifier = pickle.load (open('./corner_data/corner_classifier.obj', 'r'))	#more data
	board = Board(corner_classifier=corner_classifier)
	board.add_frame (first_frame)
	board.draw_vertices (board.current_frame)




	#=====[ Step 5: snap points to grid	]=====
	# vert_points_grid = get_points_grid (vert_lines, corners)
	# for i in range(len(vert_points_grid)):
	# 	for point in vert_points_grid[i]:
	# 		p = (int(point[0]), int(point[1]))
	# 		cv2.circle (image, p, 5, (0, 0, 255), -1)


	# #=====[ Step 6: hough transform on remaining points	]=====
	# all_points = [p for l in vert_points_grid for p in l]
	# corners_img = np.zeros (image.shape[:2], dtype=np.uint8)
	# for p in all_points:
	# 	corners_img[int(p[1])][int(p[0])] = 255
	# horz_lines = cv2.HoughLines (corners_img, 3, np.pi/180, 2)
	# horz_lines = horz_lines [0]
	# horz_lines = filter_by_slope_2 (horz_lines)
	# horz_lines = avg_close_lines_2 (horz_lines)

	# #=====[ Step 6: snap points to grid 	]=====
	# horz_points_grid = get_points_grid (horz_lines, all_points)

	# #=====[ Step 7: order lines by average coordinates	]=====
	# hpg, vpg = horz_points_grid, vert_points_grid
	# hpg_y_means = [np.mean([v[1] for v in row]) for row in hpg]
	# vpg_x_means = [np.mean([v[0] for v in row]) for row in vpg]









	# #####[ VISUALIZE CORNERS	]#####
	# cv2.namedWindow ('houghImage')
	# cv2.imshow ('houghImage', corners_img)



	# ######[ VISUALIZE LINES	]#####
	# horz_lines = [CVAnalysis.abc_to_rho_theta (l) for l in horz_lines] 
	# vert_lines = [CVAnalysis.abc_to_rho_theta (l) for l in vert_lines]
	# for rho, theta in horz_lines:
	# 	a = np.cos(theta)
	# 	b = np.sin(theta)
	# 	x0 = a*rho
	# 	y0 = b*rho
	# 	x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
	# 	y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
	# 	x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
	# 	y2 = int(y0 - 1000*(a))
	# 	cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)

	# for rho, theta in vert_lines:
	# 	a = np.cos(theta)
	# 	b = np.sin(theta)
	# 	x0 = a*rho
	# 	y0 = b*rho
	# 	x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
	# 	y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
	# 	x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
	# 	y2 = int(y0 - 1000*(a))
	# 	cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
	# cv2.namedWindow ('houghLines')
	# cv2.imshow ('hoeughLines', image)