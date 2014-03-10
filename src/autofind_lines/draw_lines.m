function draw_lines (image, lines)
% Function: draw_lines
% --------------------
% Image: image to draw lines on
% lines: 2xN matrix, where 1st row = rho, 2nd row = theta

	figure, imshow(image), hold on
	max_len = 0;
	num_lines = size(lines, 2);

	%=====[ ITERATE THROUGH LINES	]=====
	for k = 1:num_lines

		%=====[ Step 1: rho/theta to (x1, y1), (x2, y2)	]=====
		rho = lines (1, k);
		theta = lines (2, k);

		x = 0:1080;
 		y = (rho - (x * cos(theta)) ) / sin(theta);

 		plot(x, y, 'Color', 'red', 'LineWidth', 5);

	end

end

