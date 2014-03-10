function indexed_lines = horizontal_ransac (lines)
% Function: horizontal_ransac
% ---------------------------
% lines: 2xN vector, with ith col = [rho; theta] of the ith line
% returns: 
%	- indexed_lines: 3xN vector, where 3rd column is index; (up to a shift)
%	- p: regresses from line index to where it should be.
% 	(returns them from left to right)

	%=====[ Step 1: preprocess lines ]=====
	lines
	[Y, sort_permutation] = sort (lines, 2, 'descend');
	sort_permutation = sort_permutation(2, :);	%stores the permutation for sorting thetas
	sorted_lines = lines;
	sorted_lines(1, :) = lines(1, sort_permutation);
	sorted_lines(2, :) = lines(2, sort_permutation);
	y = sorted_lines(2, :);


	%=====[ Step 2: initialize parameters for RANSAC ]=====
	indices = combnk(1:9, 4);
	num_iters = size(indices, 1);
	best_Rsq = -1;
	best_p = 0;
	best_indices = [0, 0, 0, 0];

	%=====[ RANSAC ITERATIONS	]=====
	for k = 1:num_iters

		%=====[ Step 3: get a random assignment to indices	]=====
		X = indices (k, :);

		%=====[ Step 4: run linear regression	]=====
		p = polyfit (X, y, 1);

		%=====[ Step 5: calculate Rsq	]=====
		yfit = polyval (p, X);
		yresid = y - yfit;
		SSresid = sum (yresid .^ 2);
		SStotal = (length(y) - 1) * var(y);
		Rsq = 1 - (SSresid/SStotal);

		%=====[ Step 6: update Rsq	]=====
		if Rsq > best_Rsq
			best_Rsq = Rsq;
			best_p = p;
			best_indices = X;
		end
	end

	%=====[ Step 7: construct indexed_lines	]=====
	indexed_lines = vertcat(sorted_lines, best_indices);











