function y_intercepts = get_y_intercepts (lines_rt, image_height)
% Function: get_y_intercepts 
% --------------------------
% given a set of lines in (rho, theta) for mat, where 
% the ith column is the ith line and top row is rho, 
% this returns their y intercept on the image 

	%=====[ Step 1: convert lines to homogenous	]=====	
	rhos 				= lines_rt(1, :);
	thetas 				= lines_rt(2, :);
	lines_homogenous	= [cos(thetas); sin(thetas); -rhos];

	%=====[ Step 2: get homogenous coordinates for line on bottom	]=====
	line_baseline = [0; 1; -image_height];
	line_baseline = repmat(line_baseline, 1, size(lines_homogenous, 2));

	intersections_points = cross (lines_homogenous, line_baseline, 1);

	y_intercepts = intersections_points (1, :) ./ intersections_points (3, :);





