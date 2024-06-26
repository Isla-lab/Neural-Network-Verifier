import numpy as np;

from utils.propagation import get_estimation


class Node:

	def __init__(self, value, network, split_node_heuristic, split_pos_heuristic, max_depth=18, enumerate_unsafe_regions=True, depth=0, parent=None, rate_tolerance_probability=0.0001):
		
		# Parameters
		self.value = value
		self.parent = parent
		self.network = network
		self.depth = depth
		self.max_depth = max_depth
		self.enumerate_unsafe_regions = enumerate_unsafe_regions
		self.rate_tolerance_probability = rate_tolerance_probability

		# Heuristic definition
		self.split_node_heuristic = split_node_heuristic 
		self.split_pos_heuristic = split_pos_heuristic

		# Private variables
		self._probability = None
		self._children = [None, None]
		
		# Heuristic Variables
		self._propagated_median = None
		self._max_pivots = None
		self._min_pivots = None

		# Internal variables for debugging
		self.splitted_on = None


	def get_probability(self, point_cloud=3500):

		"""
		Public method that returns the probability of having a safe point in the current node, 
		this probability is calculated by sampling 'point_cloud' points

		Returns:
		--------
			probability : float
				the probability of having a safe point; 100% means that all the sampled points
				are safe
		"""
		
        # If the probability is already being calculated, just return
		if self._probability is not None: return self._probability

        # Generate a point cloud of size 'point cloud' and propagate it
		# through the network to get the probability of having a (un)safe point with Wilks probabilistic guarantees
		self._probability, sat_points = get_estimation(self.network, self.value, point_cloud, violation_rate=self.enumerate_unsafe_regions)
		
		# These two special cases will never been splitted, however for heuristic
		# purposes we must return a placeholder for the heuristic variables, it 
		# will never be used for real, but it must be not null. Otherwise it will 
		# generate an error. Notice that the children (although never used), must
		# be created for the computation of the entropy condition for the splitting.
		# It won't be used anyway beacuse the probability 1 or 0 has higher priority.
		if self._probability == 1 or self._probability == 0: 
			self._propagated_median = [0, 0]
			self._max_pivots = [0 for _ in self.value]
			self._min_pivots = [0 for _ in self.value]
			return self._probability

		# Computing additional info for heuristic purposes
		max_points = np.max(sat_points.numpy(), axis=0) 
		min_points = np.min(sat_points.numpy(), axis=0)
		# 		-> Storing the information
		self._propagated_median = np.median(sat_points, axis=0 )
		self._max_pivots = (max_points - self.value[:, 0]) / (self.value[:, 1] - self.value[:, 0])
		self._min_pivots = (min_points - self.value[:, 0]) / (self.value[:, 1] - self.value[:, 0])

		# Return the computed probability if not already returned
		return self._probability
		

	def get_children(self):

		"""
		Public method that generates the two children nodes, typically called only when the 
		testing method returns True.

		Returns:
		--------
			childred : list
				an array of two elements with the children of the node
		"""

		# If the children are already generated, just return
		if self._children[0] is not None: return self._children

		# print( "prob", self.get_probability() )

		# Call the function that implements the heuristic for the choice
		# of the node to split
		node, pivot = self._chose_split()
		self.splitted_on = node

		# Create a copy of the current area before changing it for the childre;
		# NB: the copy function is necessary to avoid any change in the parent node
		value_1 = self.value.copy() 
		value_2 = self.value.copy() 

		# Change lower and upper bound of the children
		value_1[node][1] = value_2[node][0] = pivot

		# Call the Node class for the generation of the nodes
		self._children = [
			Node(value=value_1, network=self.network, depth=self.depth+1, parent=self, split_node_heuristic=self.split_node_heuristic, split_pos_heuristic=self.split_pos_heuristic, max_depth=self.max_depth, enumerate_unsafe_regions=self.enumerate_unsafe_regions),
			Node(value=value_2, network=self.network, depth=self.depth+1, parent=self, split_node_heuristic=self.split_node_heuristic, split_pos_heuristic=self.split_pos_heuristic, max_depth=self.max_depth, enumerate_unsafe_regions=self.enumerate_unsafe_regions)
		]

		# Return the generated childred if not already returned
		return self._children
			

	def compute_area_size(self):

		"""
		Public method that computes the size of the area represented with the current node.

		Returns:
		--------
			size : float
				the size of the area represented with the node
		"""
				
		# Compute the size of each side and return the product
		sides_sizes = [side[1] - side[0] for side in self.value]
		return np.prod(sides_sizes) 
	

	def expansion_test( self ):

		if self._children[0] is None: 
			self.get_children()


		# Notice that the case self.get_probability() == 1 has already been computed
		if self.depth > self.max_depth or self.get_probability() < self.rate_tolerance_probability: 
			return False
	
		return True


	def _chose_split(self):

		"""
		Private method that select the node on which performs the splitting, the implemented heuristic is to always select the node
		with the largest bound.

		Returns:
		--------
			distance_index : int
				index of the selected node (based on the heuristic)
		"""

		########
		## Heuristic Based on The Distribution
		########
		if self.split_node_heuristic == "distr" or self.split_pos_heuristic == "distr":
			test_flag = False

			# Condition 1
			if np.min(self._max_pivots, axis=0) < 0.9: 
				pivot = np.min(self._max_pivots, axis=0)
				node = np.argmin(self._max_pivots, axis=0)
				test_flag = True
				
			# Condition 2
			if np.max(self._min_pivots, axis=0) > 0.1: 
				pivot = np.max(self._min_pivots, axis=0)
				node = np.argmax(self._min_pivots, axis=0)
				test_flag = True

			#
			if test_flag: 
				pivot_value = (pivot*(self.value[node][1]-self.value[node][0]))+self.value[node][0]
				return node, pivot_value 
			else:
				# If there is no need to use the distr heurisitc use the standard
				distances = [ (el[1] - el[0]) for el in self.value ]
				node = np.argmax(distances)
				pivot_value = self._propagated_median[node]
				return node, pivot_value
		
		########
		## Heuristic for the Node Selection
		########
		if self.split_node_heuristic == "rand": 
			node = np.random.randint(0, self.value.shape[0]) 
		elif self.split_node_heuristic == "size":
			distances = [(el[1] - el[0]) for el in self.value]
			node = np.argmax(distances)
		else:
			print(f"Invalid Heurisitc Check... {self.split_node_heuristic}"); quit()

		########
		## Heuristic for the Split Position
		########
		if self.split_pos_heuristic == "mean":
			pivot_value = (self.value[node][0]+self.value[node][1])*0.5
		elif self.split_pos_heuristic == "median":
				pivot_value = self._propagated_median[node]
		else:
			print(f"Invalid Heurisitc Check... {self.split_pos_heuristic}"); quit()

		# 
		return node, pivot_value
	