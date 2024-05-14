import os
import shutil
import time

import netver.utils.disk_utilities as disk_utils
import netver.utils.propagation_utilities as prop_utils
import numpy as np
import psutil
from netver.utils.colors import *


class ProVe( ):

	"""
	A class that implements ProVe, a verification tool based on the interval propagation. 
	This tool is based on a parallel implementation of the interval analysis on GPU that increase the performance.
	It can also run on CPU in a sequential fashion, but this drastically reduce the performance.
	ProVe can also run in a semi-formal implementation, this implementation can not formally prove when a property is statisfied, but provides a
	strict estimation of the real result.

	Attributes
	----------
		P : list
			input domain for the property in the form 'positive', each output from a point in this domain must be greater than zero.
			2-dim list: a list of two element (lower_bound, upper_bound) for each input nodes
		network : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
		unchecked_area: float
			indicates the percentage of the residual input domain to explore
		cpu_only: bool
			flag variables to indicate if the tool is running in CPU mode (default: False)
		semi_formal: bool
			flag variables to indicate if the tool is running in semi-formal mode (default: False)
		semi_formal_precision: int
			indicate the precision of the semi formal tool, this value is the number of sample generated for the 
			point cloud (default: 100)
		time_out_cycle: int
			provide a time out for the maximum number of cycle (default: 40) 
		time_out_checked: float
			provide a time out on the percentage of the verified domain, when the unknown percentage of the input domain
			go under this threshold the algorithm stop the execution, considering all the remaining domain as a violation (default: 0)
		rounding: int
			rounding for the input domain real values, expressed as an integer that represent the number of decimals value, None
			means that the rounding is the float precision of the system (default: None)
		reversed: bool
			this variables represent that the verification query is reversed, it means that "at least ONE output must be greater than 0" instead of the common
			form where "ALL inputs must be greater than zero".

	Methods
	-------
		verify( verbose )
			method that formally verify the property P on the ginve network
	"""


	# Verification hyper-parameters
	cpu_only = False
	time_out_cycle = 60
	time_out_checked = 0.0
	rounding = None
	reversed = False
	interval_propagation = 'naive'
	memory_limit = 0
	disk_limit = 0

	
	def __init__( self, network, P, dual_network, **kwargs ):

		"""
		Constructor of the class.

		Parameters
		----------
			network : tf.keras.Model
				tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
			P : list
				input domain for the property in the form 'positive', each output from a point in this domain must be greater than zero.
				2-dim list: a list of two element (lower_bound, upper_bound) for each input nodes
			kwargs : **kwargs
				dicitoanry to overide all the non-necessary paramters (if not specified the algorithm will use the default values)	
		"""

		# Input parameters
		self.network = network
		self.P = P
		self.verbose = False

		# Override the default parameters
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		# Private variables
		self.unchecked_area = 100
		self._split_matrix = self._generate_splitmatrix()
		self.dual_network = dual_network
		

		# Propagation method selection, there are different propagtion method for the different running mode: CPU or GPU
		if self.cpu_only: self._propagation_method = prop_utils.multi_area_propagation_cpu
		else: self._propagation_method = prop_utils.multi_area_propagation_gpu

		if self.memory_limit == 0:
			self.memory_limit = psutil.virtual_memory().available
		if self.disk_limit == 0:
			self.disk_limit = psutil.disk_usage('/').free

	def verify( self, start_time, verbose, estimate_VR=None ):
		"""
		Method that perform the formal analysis. When the solver explored and verify all the input domain it returns the
		violation rate, when the violation rate is zero we coclude that the original property is fully respected, i.e., SAT

		Parameters
		----------
			verbose : int
				when verbose > 0 the software print some log informations

		Returns:
		--------
			sat : bool
				true if the property P is verified on the given network, false otherwise
			info : dict
				a dictionary that contains different information on the process, the 
				key 'counter_example' returns the input configuration that cause a violation
				key 'exit_code' returns the termination reason (timeout or completed)
				key 'violation_rate' returns the value of the vilation rate as a percentage of the input domain
			"""
		
		folder_path = "./partial_results"

		if os.path.isdir(folder_path):
			shutil.rmtree(folder_path)
		
		os.mkdir(folder_path)

		# Flatten the input domain to aobtain the areas matrix to simplify the splitting
		areas_matrix = np.array([self.P.flatten()])

		disk_utils.write_array(areas_matrix, 0, folder_path)
		free_file_index = 1

		# Array with the number of violations for each depth level
		violation_rate_array = []
		
		# Loop until all the subareas are eliminated (verified) or a counter example is found

		for cycle in range(self.time_out_cycle):

			checked_area_log = f"\n{bcolors.BOLD}{bcolors.WARNING}Iteration cycle {cycle:3d}/{self.time_out_cycle:3d} (checked area  {100-self.unchecked_area:5.3f}%){bcolors.ENDC}"
			if verbose > 0: print( checked_area_log )

			iter = 0
			no_areas_left = True

			checked_size = 0
			violated_length = 0
			
			while True:
				# Read the next partial results from disk
				areas_matrix = disk_utils.read_array(folder_path, iter)

				# If all partial results have been considered exit the cycle
				if areas_matrix is None:
					break

				# If current partial results are empty move to the next batch
				if areas_matrix.shape[0] == 0:
					iter += 1
					continue

				if self.rounding is not None: areas_matrix = np.round(areas_matrix, self.rounding)
				test_domain = areas_matrix.reshape(-1, self.P.shape[0], 2)


				# Call the propagation method to obtain the output bound from the input area (primal and dual)
				test_bound = self._propagation_method( test_domain, self.network, self.interval_propagation, self.memory_limit )
				test_bound_dual = self._propagation_method( test_domain, self.dual_network, self.interval_propagation, self.memory_limit )

				# Call the verifier (N(x) >= 0) on all the subareas
				unknown_id, violated_id, proved_id = self._complete_verifier( test_bound, test_bound_dual )

				checked_size += proved_id[0].shape[0] + violated_id[0].shape[0]
				violated_length += len(violated_id[0])

				# Iterate only on the unverified subareas
				areas_matrix = areas_matrix[unknown_id]

				# Exit check when all the subareas are verified
				if areas_matrix.shape[0] > 0: no_areas_left = False

				# Save the current partial results to disk
				disk_utils.write_array(areas_matrix, iter, folder_path)

				iter += 1

			#print(f"Checked: {checked_size}")
			# Call the updater for the checked area
			self._update_unchecked_area( cycle, checked_size )

			# Update the violation rate array to compute the violation rate
			violation_rate_array.append( violated_length )
			
			if no_areas_left:
				break

			# Exit check when the checked area is below the timeout threshold or when disk usage exceeds the maximum threshold
			if (self.unchecked_area < self.time_out_checked or
					(free_file_index > 1 and free_file_index * self.memory_limit >= self.disk_limit)):
				# Update the violation rate array adding all the remaining elements from disk
				for iter in range(free_file_index):
					areas_matrix = disk_utils.read_array(folder_path, iter)
					violation_rate_array[-1] += areas_matrix.shape[0]

				break
			else:
				additional_violation = 0
				for iter in range(free_file_index):
					areas_matrix = disk_utils.read_array(folder_path, iter)
					additional_violation += areas_matrix.shape[0]
				
				violations_weigth = sum( [ 2**i * n for i, n in enumerate(reversed(violation_rate_array[:-1] + [violation_rate_array[-1] + additional_violation]))] ) 
				violation_rate =  violations_weigth / 2**(len(violation_rate_array ) -1) * 100 

				end_time = time.time()
				time_execution = round((end_time-start_time)/60,5)

				print(f"\tTime elapsed: {time_execution} min")
				print(f"\tVR (considering also the unknow areas as unsafe): {round(violation_rate,5)}%")

				violations_weigth = sum( [ 2**i * n for i, n in enumerate(reversed(violation_rate_array[:-1] + [violation_rate_array[-1]]))] ) 
				violation_rate =  violations_weigth / 2**(len(violation_rate_array ) -1) * 100 
				print(f"\tUnderestimated VR (considering the unknow areas as safe): {round(violation_rate, 4)}%")
				print(f"\tEstimated VR using sampling: {round(estimate_VR, 4)}%")

			# Split the inputs (Iterative Refinement) specifying the maximum amount of memory
			free_file_index = self._split( folder_path, free_file_index, self.memory_limit)


		# Check if the exit reason is the time out on the cycle
		if cycle >= self.time_out_cycle:
			# return UNSAT with the no counter example, specifying the exit reason
			return False, { "counter_example" : None, "exit_code" : "cycle_timeout" }

		# Compute the violation rate, multipling the depth for the number for each violation
		# and normalizing for the number of theoretical leaf
		violations_weigth = sum( [ 2**i * n for i, n in enumerate(reversed(violation_rate_array))] ) 
		violation_rate =  violations_weigth / 2**(len(violation_rate_array)-1) * 100 

		# All the input are verified, return SAT with no counter example
		return (violation_rate == 0), { "violation_rate": violation_rate }


	def _complete_verifier( self, test_bound, test_bound_dual ):
		
		"""
		Method that verify the property on a list of the computed (or sampled in semi-formal mode) output bound.

		Parameters
		----------
			test_bound : list
				the output bound expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
				(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
			test_bound_dual : list
				same as test_bound but for the dual network

		Returns:
		--------
			unknown_id : list
				list of integer with the index of the bound that dows not respect the property and require
				further investigations
			violated_id : list
				list of integer with the index of the bound that violated the give property
			proved_id : list
				list of integer with the index of the bound that respect the give property
		"""

		# Check the property in standard and reverse mode for both a violation and a proof. 
		# To prove the property, in the first case every bound must be greater than zero,
		# in the latter at least one bound must be greater than zero. 
		# To deny the property, in the first case at least one bound must be greater than zero,
		# in the latter every one bound must be greater than zero. 
		
		if not self.reversed:
			proved_bound = np.all(test_bound[:, :, 0] >= 0, axis=1) # Property proved here!
			violated_bound = np.any(test_bound_dual[:, :, 0] > 0, axis=1) # Property violated here!
		else:
			proved_bound = np.any(test_bound[:, :, 0] > 0, axis=1) # Property proved here!
			violated_bound = np.all(test_bound_dual[:, :, 0] >= 0, axis=1) # Property violated here!

		# Create a mask for the unknown, when a property is neither proved or vioalted
		unknown_mask = np.logical_or(proved_bound, violated_bound)

		# Find the unknown and proved index with the built-in numpy function		
		unknown_id = np.where( unknown_mask == False )
		violated_id = np.where( violated_bound == True )
		proved_id = np.where( proved_bound == True )

		#
		return unknown_id, violated_id, proved_id
	

	def _update_unchecked_area( self, depth, verified_nodes ):

		"""
		Update the percentage of the explorated area. This value exploiting the fact that the genereted tree is a binary tree,
		so at each level, the number of leaf is 2^depth. The size of the verified input is computed normalizing the number of
		node (i.e., sub-intervals) for the size of the current depth. 
		e.g., in level 2 we will have 2^2=4 nodes, if 3 nodes are verified at this depth we know that 3/4 of the network is verified
		at this level. We then update the global counter.

		Parameters
		----------
			depth : int 
				the depth of the tree on which we are performing the verification
			verified_nodes : int
				the number of verified nodes at this depth
		"""

		# Update of the area, normalizing in a percentage value
		self.unchecked_area -= 100 / 2**depth * verified_nodes
		

	def _split( self, folder_path, free_file_index, memory_limit):

		"""
		Split the current domain in 2 domain, selecting on which node perform the cut
		with an heuristic from the function _chose_node. For the splitting this methods
		exploits the dot product, (see [a] for details on this process).

		Parameters
		----------
			folder_path : disk folder where the partial results for area_matrix have been saved
			free_file_index : the currently free batch index for subsequent partial results
			memory_limit : maximum threshold for virtual memory usage

		Returns:
		--------
			areas_matrix : list
				the splitted input matrix reshaped in the original form, notice that the splitted matrix will have
				2 times the number of rows of the input one
		"""

		iter = 0
		while True:
			areas_matrix = disk_utils.read_array(folder_path, iter)
			if areas_matrix.shape[0] == 0:
				iter += 1
				continue

			i = self._chose_node( areas_matrix)
			break

		# Set the maximum size for partial result batches to the maximum threshold for virtual memory usage
		max_chunk_rows = int(memory_limit / (areas_matrix.nbytes / len(areas_matrix)))

		fixed_range = free_file_index
		for iter in range(fixed_range):
			# Get the current partial result from disk
			areas_matrix = disk_utils.read_array(folder_path, iter)

			if areas_matrix.shape[0] == 0:
				continue

			# Compute batch size (rows of AreaMatrix to compute simultaneously) based on the memory limit by the size of each row
			batch_size = int(memory_limit / (areas_matrix.nbytes / len(areas_matrix)))
			# Compute the total number of batches
			batches = len(areas_matrix) // batch_size

			if areas_matrix.nbytes % batch_size != 0:
				batches += 1

			res = np.empty((0, self._split_matrix[i].shape[1]), dtype=np.float32)

			for batch_iter in range(batches):
				if batch_iter == batches - 1:
					upper_limit = len(areas_matrix)
				else:
					upper_limit = batch_size
				
				temp_result = areas_matrix[0 : upper_limit].dot(self._split_matrix[i])

				if batch_iter < batches - 1:
					areas_matrix = areas_matrix[upper_limit:]
				
				res = np.concatenate((res, temp_result), axis=0)

			areas_matrix = res.reshape((len(res) * 2, self.P.shape[0] * 2))

			# Save partial results to either a single file or multiple ones
			if areas_matrix.shape[0] <= max_chunk_rows:
				disk_utils.write_array(areas_matrix, iter, folder_path)
			else:
				disk_utils.write_array(areas_matrix[:max_chunk_rows], iter, folder_path)
				disk_utils.write_array(areas_matrix[max_chunk_rows:], free_file_index, folder_path)
				free_file_index += 1


		return free_file_index


	def _chose_node( self, area_matrix):

		"""
		Select the node on which performs the splitting, the implemented heuristic is to always select the node
		with the largest bound.

		Parameters
		----------
			areas_matrix : list 
				matrix that represent the current state of the domain, a list where each row is a sub-portion
				of the global input domain

		Returns:
		--------
			distance_index : int
				index of the selected node (based on the heuristic)
		"""

		# Compute the size of the bound for each sub-interval (i.e., rows)
		#print(area_matrix.shape)

		row = area_matrix[0]
		closed = []

		split_idx = 0
		smear = 0

		for index, el in enumerate(row.reshape(self.P.shape[0], 2)):

			distance = el[1] - el[0]
			
			if(distance == 0): closed.append(index)

			if distance > smear:
				smear = distance
				split_idx = index

		# Find the index of the node with the largest influence over the output
		return split_idx
	

	def _generate_splitmatrix( self ):

		"""
		Method that generated the split matrix for the splitting of the input domain, it works as a multiplier for the 
		dot product (see [a] for the implementation details)

		Returns:
		--------
			split_matrix : list
				hypermatrix with a splitting matrix for each input node of the domain
		"""

		# If the tool is running with a rounding value it's necesary to add an eps value on the value for the splitting (0.5)
		# fn the multiplication matrix, to prevent a infinite loop. 
		rounding_fix = 0 if self.rounding is None else 1/10**(self.rounding+1)

		# Compute the number of necessary matrix (one for each node), all the matrices will have 2 times this size 
		# to compute lower and upper bound
		n = (self.P.shape[0] * 2)
		split_matrix = []

		# Iterate over each node times upper and lower to update the splitting matrix, each matrix is an identity matrix with
		# a splitting value (0.5) on the coordinate of the interested node.
		for i in range( self.P.shape[0] ):
			split_matrix_base = np.concatenate((np.identity(n, dtype="float32"), np.identity(n, dtype="float32")), 1)
			split_matrix_base[(i * 2) + 0][(i * 2) + 1] = 0.5 
			split_matrix_base[(i * 2) + 1][(i * 2) + 1] = (0.5 - rounding_fix)
			split_matrix_base[(i * 2) + 0][n + (i * 2)] = 0.5 
			split_matrix_base[(i * 2) + 1][n + (i * 2)] = (0.5 + rounding_fix)
			split_matrix.append(split_matrix_base)

		#
		return split_matrix

