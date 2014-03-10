function BIH = autofind_BIH (corners_img)
% Function: autofind_BIH
% ----------------------
% prototype for finding the BIH from an image of an empty chessboard
% assumes that corners_img is all black except for on coordinates where
% corners were detected

	%=====[ Step 1: get horizontal/vertical lines, along with indices up to a shift	]=====
	horizontal_lines = get_horizontal_lines (corners_img);
	vertical_lines = get_vertical_lines (corners_img);
