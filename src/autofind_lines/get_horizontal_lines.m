function indexed_lines = get_horizontal_lines (corners_img)
% Function: get_horizontal_lines
% ------------------------------
% gets horizontal lines from the image by applying the hough transform,
% then finding a relationship between the rho value and distance from 
% the image

	%=====[ Step 1: set parameters ]=====
	num_peaks = 4;
	theta_buckets_horz = [-90, -89];
	rho_resolution_horz = 6;

	%=====[ Step 2: find peaks	]=====
	[H, theta, rho] = hough (corners_img, 'Theta', theta_buckets_horz, 'RhoResolution', rho_resolution_horz);
	peaks = houghpeaks(H, num_peaks);

	%=====[ Step 3: convert peaks to rho, theta	]=====
	theta_rad = fromDegrees ('radians', theta);
	rhos = rho(peaks(:, 1));
	thetas = theta_rad(peaks(:, 2));
	lines = [rhos; thetas];

	%=====[ Step 4: figure out which lines they are	]=====
	indexed_lines = horizontal_ransac (lines);

	%#####[ DEBUG: show lines	]#####
	% draw_lines (corners_img, indexed_lines(1:2, :));