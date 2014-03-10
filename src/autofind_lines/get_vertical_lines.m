function indexed_lines = get_vertical_lines (corners_img)
% Function: get_vertical_lines
% ----------------------------
% gets horizontal lines from the image by applying the hough transform,
% then finding a relationship between the rho value and distance from 
% the image

	%=====[ Step 1: set parameters ]=====
	num_peaks = 5;
	theta_buckets = -40:40;
	rho_resolution = 6;

	%=====[ Step 2: find peaks	]=====
	[H, theta, rho] = hough (corners_img, 'Theta', theta_buckets);
	peaks = houghpeaks(H, num_peaks);

	%=====[ Step 3: convert peaks to rho, theta	]=====
	theta_rad = fromDegrees ('radians', theta);
	rhos = rho(peaks(:, 1));
	thetas = theta_rad(peaks(:, 2));
	lines = [rhos; thetas];

	%=====[ Step 4: figure out which lines they are	]=====
	indexed_lines = vertical_ransac (lines);

	%#####[ DEBUG: show lines	]#####
	% draw_lines (corners_img, lines)
