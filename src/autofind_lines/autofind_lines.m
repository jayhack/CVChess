function [horizontal_lines, vertical_lines] = autofind_lines ()
% Function: autofind_BIH
% ----------------------
% prototype for finding the BIH from an image of an empty chessboard
% assumes that corners_img is all black except for on coordinates where
% corners were detected

	%=====[ Step 1: get corners image from IPC	]=====
	corners_img_name = '../IPC/corners.png';
	corners_img = imread (corners_img_name);
	% imshow (corners_img);

	%=====[ Step 2: get horizontal/vertical lines, along with indices up to a shift	]=====
	horizontal_lines = get_horizontal_lines (corners_img);
	vertical_lines = get_vertical_lines (corners_img);

	%=====[ Step 3: write them to a file	]=====
	dlmwrite ('../IPC/horizontal_lines.csv', horizontal_lines);
	dlmwrite('../IPC/vertical_lines.csv', vertical_lines);

	%=====[ Step 4: quit out	]=====
	quit;