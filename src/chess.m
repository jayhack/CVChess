

%==========[ Step 1: read in image, get corners	]==========
image = imread ('../data/basic_board.jpg');
corners =  [ 	228, 107;		% top left		
				228, 131.4;		% bottom left
			 	274.5, 107;		% top right
				275.5, 130.3;	% bottom right
				230, 84;		% top top left
				273, 84;		% top top right
			];

[points, boardsize] = detectCheckerboardPoints (image)

%==========[ Step 2: convert to homogenous coords, get parallel line pairs	]==========
corners = [ corners, ones(6, 1)];

vert1 = cross (corners(1, :)', corners(2, :)');	%	|
vert2 = cross (corners(3, :)', corners(4, :)');	%		|

pos1  = cross (corners(1, :)', corners(6, :)');	%	/
pos2  = cross (corners(2, :)', corners(3, :)');	% 		/

neg1  = cross (corners(3, :)', corners(5, :)');	% 		\
neg2  = cross (corners(4, :)', corners(1, :)');	%	\


%==========[ Step 3: get intersections of lines, line at infinity	]==========
int_vert = normalize(cross (vert1, vert2));
int_pos = normalize(cross (pos1, pos2));
int_neg = normalize(cross (neg1, neg2));

inf_line_1 = normalize(cross(int_vert, int_pos));
inf_line_2 = normalize(cross (int_pos, int_neg));
inf_line_3 = normalize(cross (int_neg, int_vert));




%==========[ Step 2: draw in corners	]==========
% imshow (image);
% hold on;
% plot (corners(:, 1), corners(:, 2), 'o');
% hold off;
