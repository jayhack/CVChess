import cv2
import numpy as np
import sklearn
import pickle
import CVAnalysis


num_pixels_away = 10

def euclidean_distance (p1, p2):
	return np.sqrt( (p1[0] - p2[0])**2 + ((p1[1] - p2[1])**2))


def point_avg (points_cluster):
	return np.mean([p[0] for p in points_cluster]), np.mean([p[1] for p in points_cluster])


def avg_close_points (keypoints_list):
	"""
		Function: avg_close_points 
		--------------------------
		given a list of keypoints, this returns another list of 
		(x, y) pairs for points that are very close 
	"""
	#=====[ Step 1: get points out of each one	]=====
	old_points = np.array([k.pt for k in keypoints_list])

	#=====[ Step 2: get new_points	]=====
	new_points = []
	while len(old_points) > 1:
		p1 = old_points [0]
		distances = np.array([euclidean_distance (p1, p2) for p2 in old_points])
		idx = (distances < num_pixels_away)
		points_cluster = old_points[idx]
		new_point = point_avg (points_cluster)
		new_points.append (new_point)
		old_points = old_points[np.invert(idx)]

	return new_points


def avg_close_lines (lines_list):
	"""
		Function: avg_close_points 
		--------------------------
		given a list of keypoints, this returns another list of 
		(x, y) pairs for points that are very close 
	"""
	lines = [(rho, theta) for rho, theta in lines_list[0]]

	#=====[ Step 1: get points out of each one	]=====
	old_lines = np.array(lines)

	#=====[ Step 2: get new_points	]=====
	new_lines = []
	while len(old_lines	) > 1:
		l1 = old_lines [0]
		distances = np.array([abs(l1[1] - l2[1]) for l2 in old_lines])
		idx = (distances < 0.1)
		lines_cluster = old_lines[idx]
		new_line = point_avg (lines_cluster)
		new_lines.append (new_line)
		old_lines = old_lines[np.invert(idx)]

	return new_lines


def avg_close_lines_2 (lines_list):
	"""
		Function: avg_close_points 
		--------------------------
		given a list of keypoints, this returns another list of 
		(x, y) pairs for points that are very close 
	"""
	lines = [(rho, theta) for rho, theta in lines_list]

	#=====[ Step 1: get points out of each one	]=====
	old_lines = np.array(lines)

	#=====[ Step 2: get new_points	]=====
	new_lines = []
	while len(old_lines	) > 1:
		l1 = old_lines [0]
		distances = np.array([abs(l1[0] - l2[0]) for l2 in old_lines])
		idx = (distances < 10)
		lines_cluster = old_lines[idx]
		new_line = point_avg (lines_cluster)
		new_lines.append (new_line)
		old_lines = old_lines[np.invert(idx)]

	return new_lines



def rho_theta_to_abc (line):
	"""
		Function: rho_theta_to_abc
		--------------------------
		given a line represented as (rho, theta),
		this returns the triplet (a, b, c) such that 
		ax + by + c = 0 for all points (x, y) on the 
		line
	"""
	rho, theta = line[0], line[1]
	return (np.cos(theta), np.sin(theta), -rho)


def filter_by_slope (line_list):
	"""
		Function: filter_by_slope
		-------------------------
		given a list of lines in rho, theta form,
		picks
	"""
	abc_lines = [rho_theta_to_abc(l) for l in line_list]
	return [l_rt for l_rt, l_abc in zip(line_list, abc_lines) if abs(l_abc[0]/(l_abc[1] + 0.0000967)) > 1]

def filter_by_slope_2 (line_list):
	"""
		Function: filter_by_slope
		-------------------------
		given a list of lines in rho, theta form,
		picks
	"""
	abc_lines = [rho_theta_to_abc(l) for l in line_list]
	return np.array([l_rt for l_rt, l_abc in zip(line_list, abc_lines) if abs(l_abc[0]/(l_abc[1] + 0.000096)) < 0.1])



def get_line_point_distance (line, point):
	"""
		Function: get_line_point_distance 
		---------------------------------
		returns the distance from the point to the line 
	"""
	a, b, c = rho_theta_to_abc (line)
	x, y = point[0], point[1]
	return np.abs(a*x + b*y + c)/np.sqrt(a**2 + b**2)


def get_points_grid (lines, corners):
	"""
		Function: get_points_grid
		-------------------------
		given a set of lines and a set of corners,
		this will find which line each corner falls on 
		and assign it to it; we will then sort by y-coord 
	"""
	#=====[ Step 1: initialize points grid	]=====
	points_grid = [[] for line in lines]

	#=====[ Step 2: iterate through corners, adding to lines	]=====
	for corner in corners:
		distances = np.array([get_line_point_distance (line, corner) for line in lines])
		if np.min (distances) < 8:
			line_index = np.argmin(distances)
			points_grid[line_index].append (corner)

	#=====[ Step 3: sort each one by y coordinate	]=====
	for i in range(len(points_grid)):
		points_grid[i].sort (key=lambda x: x[0])

	return points_grid



if __name__ == '__main__':

	#=====[ Step 1: read in image ]=====
	image = cv2.imread ('../data/p2/1.jpg')
	
	#=====[ Step 2: get harris corners	]=====
	hc = CVAnalysis.get_harris_corners (image)

	#=====[ Step 3: classify them	]=====
	sd = CVAnalysis.get_sift_descriptors (image, hc)
	clf = pickle.load (open('../data/classifiers/corner_classifier.clf', 'r'))
	predictions = clf.predict (sd)
	idx = (predictions == 1)
	corners = [c for c, i in zip(hc, idx) if i]
	corners = avg_close_points (corners)

	#=====[ Step 3: get black image, fill in corners	]=====
	corners_img = np.zeros (image.shape[:2], dtype=np.uint8)
	for corner in corners:
		corners_img[int(corner[1])][int(corner[0])] = 255

	#=====[ Step 4: apply hough transform to get lines	]=====
	lines = cv2.HoughLines (corners_img, 3, np.pi/180, 6)
	lines = avg_close_lines (lines)
	lines = filter_by_slope (lines)


	#=====[ Step 5: snap points to grid	]=====
	points_grid = get_points_grid (lines, corners)
	for i in range(len(points_grid)):
		for point in points_grid[i]:
			p = (int(point[0]), int(point[1]))
			cv2.circle (image, p, 5, (0, 0, 255), -1)


	#=====[ Step 6: hough transform on remaining points	]=====
	all_points = [p for l in points_grid for p in l]
	corners_img = np.zeros (image.shape[:2], dtype=np.uint8)
	for p in all_points:
		corners_img[int(p[1])][int(p[0])] = 255
	lines = cv2.HoughLines (corners_img, 3, np.pi/180, 2)
	lines = lines [0]
	lines = filter_by_slope_2 (lines)
	lines = avg_close_lines_2 (lines)


	# #####[ VISUALIZE CORNERS	]#####
	# cv2.namedWindow ('houghImage')
	# cv2.imshow ('houghImage', corners_img)



	# ######[ VISUALIZE LINES	]#####
	for rho, theta in lines:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
		y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
		x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
		y2 = int(y0 - 1000*(a))
		cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
	cv2.namedWindow ('houghLines')
	cv2.imshow ('houghLines', image)