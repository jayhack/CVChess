function BIH = autofind_BIH (corners_img)
% Function: autofind_BIH
% ----------------------
% prototype for finding the BIH from an image of an empty chessboard
% assumes that corners_img is all black except for on coordinates where
% corners were detected
	num_peaks = 5;

	%=====[ Step 1: hough transform on image	]=====
	[H, theta, rho] = hough (corners_img);

	%=====[ Step 2: find hough peaks	]=====
	peaks = houghpeaks(H, 10);
	theta = fromDegrees ('radians', theta);

	%=====[ Step 3: convert these to rho, theta	]=====
	rhos = rho(peaks(:, 1));
	thetas = theta(peaks(:, 2));
	lines = [rhos; thetas];
	lines

	%=====[ Step 3: draw these lines on image	]=====
	draw_lines (corners_img, lines);

	%=====[ Step 4: ...	]=====
	BIH = 0