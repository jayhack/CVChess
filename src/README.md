# CVChess: computer vision insights into board games
# --------------------------------------------------
# Jay Hack and Prithvi Ramakrishnan, Winter 2014 231A final project
# https://github.com/jayhack/CVChess

0. TODO
=======

Overview:
---------
• Get all packages working on one python installation (ipython)
	-- cv2
	-- numpy
	-- PIL

• Gather baseline data:
	-- board corners
	-- square occupation
	-- intuitive interface

• Implement automated corner-finding:
	-- find harris corners in image
	-- train/apply classifier for candidate filtering
	-- train/apply PGM for finding corners

• Baseline visualization:
	-- show each step in separate displayed images
	-- cv2 display functions

• Make it work in real-time
	-- create movement detector

• Integrate chess engine
	-- @prithvi

• Advanced Visualization:
	- provide chess engine insights

• Bootstrap more data from real games for better classifiers
	- makes for easier training
	- adapt to lighting conditions, etc.


Gathering Data:
---------------
• Update BoardImage:
	-- image representation:
		• cv2 image
	-- corner representation:
		• image coordinates
		• board coordinates
		• descriptor
	-- square representation:
		• corners
			-- pointer to all of them
		• algebraic notation
		• occupation
	-- visualization:
		• draw vertices
		• draw square edges
		• draw square surfaces

• MarkImages.py:
	-- Mark corners:
		• harris corner detector finds all possible candidates
		• clicking in near vicinity of harris corners, in pre-specified order,
			adds the correspondance
		• finds homography from this, marks all other probable corners?

• Square Occupation
	-- after corners marked, establish squares, draw outlines
	-- user clicks on *all* occupied ones

• Create dataset class
	-- 


Finding Corners:
----------------
• Get corner candidates:
	-- get_harris_corners (image)
	-- load_classifiers ()
		• load_harris_corner_classifier ()
		• load_square_occupation_classifier ()
	-- filter_harris_corners (harris_corners)
		• applies harris_corner_classifier to harris corners

• Get correspondences:
	-- 


CVAnalyzer:
-----------
• Implement RANSAC

Report:
-------
• References for finding corners in the first place





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
