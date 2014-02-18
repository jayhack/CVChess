from time import time
import pickle
import matplotlib.pyplot as plt


####################################################################################################
##############################[ --- CornerImage --- ]###############################################
####################################################################################################

class CornerImage:
	"""
		Class: CornerImage 
		------------------
		container for a marked corner in an image.
	"""
	
	def __init__ (self, image, board_coords, image_coords):
		"""
			Function: Constructor 
			---------------------
			image: image corner is located in 
			board_coords: location of corner in board coordinates 
			image_coords: location of corner in image coordinates
		"""
		self.board_coords = board_coords
		self.image_coords = image_coords
		self.descriptor = descriptor

	
	def __str__ (self):
		"""
			Function: __str__
			-----------------
			printable string rep
		"""
		raise NotImplementedError


	def draw (self):
		"""
			PUBLIC: draw
			------------
			draws a representation of this corner onto the image
		"""
		raise NotImplementedError










####################################################################################################
##############################[ --- SquareImage --- ]###############################################
####################################################################################################

class SquareImage:
	"""
		Class: SquareImage
		------------------
		container for a square in an image
	"""

	def __init__ (self, image, corners):
		"""
			PUBLIC: Constructor
			-------------------
			image: image that square is located in 
			corners: list of CornerImage objects defining this object
		"""
		raise NotImplementedError


	def __str__ (self, image, corners):
		"""
			Function: __str__
			-----------------
			printable string rep
		"""
		raise NotImplementedError


	def draw (self):
		"""
			PUBLIC: draw
			------------
			draws a representation of this corner onto the image
		"""
		raise NotImplementedError










####################################################################################################
##############################[ --- BoardImage --- ]################################################
####################################################################################################

class BoardImage:
	"""
		Class: BoardImage
		-----------------
		class for representing an image of a board

		Member Variables:
			- image: numpy ndarray representing the image 
			- corners: list of point correspondances between board coordinates
						and image coordinates. (board_coords, image_coords, descriptor)
			- squares: list of SquareImage 

	"""

	def __init__ (self, name=None, image=None, board_points=None, image_points=None, filename=None):
		"""
			PRIVATE: Constructor
			--------------------
			constructs a BoardImage from it's constituent data
			or the filename of a saved one
		"""
		#=====[ Step 1: file/not file ]=====
		if filename:
			self.load (filename)
			return
	
		#=====[ Step 2: check arguments	]=====
		if None in [name, image, board_points, image_points]:
				raise StandardError ("Must enter all data arguments or a filename")

		#=====[ Step 3: set data	]=====
		self.image = image
		self.board_points = board_points
		self.image_points = image_points
		assert len(board_points) == len(image_points)

		#=====[ Step 4: set name	]=====
		if not name:
			self.name = str(time ())
		else:
			self.name = name


	def __str__ (self):
		"""
			PUBLIC: __str__
			---------------
			prints out a summary of this object 
		"""
		title 			= "==========[ 	BoardImage: " + self.name + " ]=========="
		point_count		= "##### " + str(len(self.board_points)) + " point correspondances: #####"
		point_corr 		= '\n'.join(['	' + str(bp) + '->' + str(ip) for bp, ip in zip (self.board_points, self.image_points)])
		return '\n'.join ([title, point_count, point_corr])



	####################################################################################################
	##############################[ --- UPDATING --- ]##################################################
	####################################################################################################

	def add_point_correspondance (self, board_point=None, image_point=None):
		"""
			PUBLIC: add_point_correspondance
			--------------------------------
			add a single point correspondance
		"""
		assert not None in [board_point, image_point]
		self.board_points.append (board_point)
		self.image_points.append (image_point)










	####################################################################################################
	##############################[ --- DISPLAYING --- ]################################################
	####################################################################################################

	def draw (self):
		"""
			PUBLIC: draw
			------------
			draws a representation of this corner onto the image
		"""
		raise NotImplementedError










	####################################################################################################
	##############################[ --- LOADING/SAVING --- ]############################################
	####################################################################################################	

	def save (self, filename):
		"""
			PUBLIC: save
			------------
			saves this object to disk
		"""
		pickle.dump (	{	'name':self.name,
							'image':self.image,
							'board_points': self.board_points,
							'image_points': self.image_points
						}, 
						open(filename, 'w'))

	
	def load (self, filename):
		"""
			PUBLIC: load
			------------
			loads a past BoardImage from a file 
		"""
		save_file 	= open(filename, 'r')
		saved_dict 	= pickle.load (save_file)
		self.name 	= saved_dict['name']
		self.image 	= saved_dict['image']
		self.board_points = saved_dict['board_points']
		self.image_points = saved_dict['image_points']
