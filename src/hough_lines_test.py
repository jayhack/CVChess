import cv2
import numpy as np
import pickle
import CVAnalysis

if __name__ == '__main__':

	#=====[ Step 1: read in image ]=====
	image = cv2.imread ('../data/p1/5.jpg')
	
	#=====[ Step 2: get positive keypoints	]=====
	hc = CVAnalysis.get_harris_corners (image)
	predictions = pickle.load (open('test_predictions.pkl', 'r'))
	idx = (predictions == 1)
	corners = [c for c, i in zip(hc, idx) if i]

	#=====[ Step 3: get black image, fill in corners	]=====
	corners_img = np.zeros (image.shape[:2], dtype=np.uint8)
	for corner in corners:
		pt = corner.pt
		corners_img[pt[1]][pt[0]] = 255


	#=====[ Step 4: apply hough transform	]=====
	lines = cv2.HoughLines (corners_img, 3, np.pi/180, 5)

	for rho, theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
		y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
		x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
		y2 = int(y0 - 1000*(a))
		cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

	cv2.namedWindow ('houghLines')
	cv2.imshow ('houghLines', image)