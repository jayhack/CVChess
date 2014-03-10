function BIH = autofind_BIH (corners_img)
% Function: autofind_BIH
% ----------------------
% prototype for finding the BIH from an image of an empty chessboard
% assumes that corners_img is all black except for on coordinates where
% corners were detected

	%=====[ Step 1: hough transform on image	]=====
	[H, theta, rho] = hough (corners_img)

	%=====[ Step 2: find hough peaks	]=====
	P = houghpeaks(H, 5, 'threshold', ceil(0.3*max(H(:))));

	%=====[ Step 3: find lines in the image from peaks	]=====
	lines = houghlines(corners_img, theta, rho, P, 'FillGap', 5, 'MinLength', 7);
