# CVChess: computer vision insights into board games
# --------------------------------------------------
# Jay Hack and Prithvi Ramakrishnan, Winter 2014 231A final project
# https://github.com/jayhack/CVChess

0. TODO
=======

main:
-----
• get it actually running
• integrate CVAnalyzer, BoardImage, Board

MarkImage:
----------
• Have it display points as you mark them

BoardImage:
-----------
• Finish draw_marked_points

CVAnalyzer:
-----------
• Figure out why it can only get affine projections
• Implement RANSAC
• Have it update BIH real-time

Report:
-------
• Examples of it correctly getting the homography, marking vertices





1. Setup
========

Dependences:
------------
• Numpy:
	- used for all linear algebra
• PIL:
	- used for dealing with images


2. Naming/Conventions
=====================

Coordinate Systems:
-------------------

	Algebraic Notation:
	-------------------
	• Tuples (character, integer) representing squares on the board in conventional
		chess algebraic notation.
	• Domain: {A-H}*{1-8}

	Board Coordinates:
	------------------
	• integer pairs (x, y) representing vertices (corners) on the board.
	• represented as python tuples
	• This is all from the *white* player's perspective
		-- that is, (0, 0) corresponds to the top left vertex of the board 
			from his perspective, namely square (A, 8)

	Image Coordinates:
	------------------
	• float pairs (x, y) representing locations within images
	• (0.0, 0.0) represents the top left coordinate of the image

	Board-Image Homography:
	-----------------------
	• Projective transformation (homography) transforming points from board coordinates
		to image coordinates
	• Represented as 'BIH' in the code wherever convenient


3. Classes
==========

Board
-----
• Defined in Board.py
• Contains a representation of the board state at all times

Square
------
• Defined in Square.py
• Contains a representation of a single square at all times, including:
	-- its algebraic notation
	-- the board coordinates of each of its vertices
	-- the image coordinates of each of its vertices
