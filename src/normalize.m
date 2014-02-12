function normalized = normalize (m)
% Function: normalize
% -------------------
% given a homogenous coord, this will normalize it
% so that the last coordinate is 1
last_coord = m(end, 1);
normalized = m ./last_coord;
end

