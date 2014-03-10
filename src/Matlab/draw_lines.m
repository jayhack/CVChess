function draw_lines (image, lines)
% Function: draw_lines
% --------------------
% given a list of lines (output from houghLines), this draws
% it on a plot of the original image

	figure, imshow(image), hold on
	max_len = 0;

	%=====[ ITERATE THROUGH LINES	]=====
	for k = 1:length(lines)


		%=====[ Step 1: extract coordinates	]=====
		xy = [lines(k).point1; lines(k).point2];
		plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

		%=====[ Step 2: plot beginnings and ends of lines	]=====
		plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
		plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

	end

end
