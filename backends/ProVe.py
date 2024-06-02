###########################################################################
##            This file is part of the NetVer verifier                   ##
##                                                                       ##
##   Copyright (C) 2023-2024 The NetVer Team                             ##                                                                 ##
##   See CONTRIBUTORS for all author contacts and affiliations.          ##
##                                                                       ##
##     This program is licensed under the BSD 3-Clause License,          ##
##        contained in the LICENCE file in this directory.               ##
##                                                                       ##
###########################################################################


import os
import shutil
import torch
import time
import utils.disk_utilities as disk_utils
import utils.general_utils as gen_utilities
import utils.propagation as prop_utility
import numpy as np
import psutil


class ProVe():

	"""
	A class that implements ProVe (Corsi et al. 2021), a verification tool based on the interval propagation. 
	This tool is based on a parallel implementation of the interval analysis on GPU that increase the performance.
	It can also run on CPU in a sequential fashion, but this drastically reduce the performance.
	Attributes
	----------
	
	Methods
	-------
		verify(verbose)
			method that formally verify the property P on the ginve network
	"""

	
	def __init__(self, config):
		# Input parameters
		self.network = torch.load(config['model']['path'])

		self.property = gen_utilities.create_property(config)[0]
		self.input_predicate = np.array(self.property["inputs"])
		self.output_predicate = np.array(self.property["outputs"])

		self.compute_violation_rate = config['verifier']['params']['compute_violation_rate']
		self.estimation_points = config['verifier']['params']['estimation_points']
		self.estimated_VR = prop_utility.get_estimation(self.network, self.input_predicate, self.estimation_points, self.compute_violation_rate)
		
		# Verification hyper-parameters
		self.cpu_only = config['verifier']['params']['cpu_only']
		self.time_out_cycle = int(config['verifier']['params']['time_out_cycle'])
		self.time_out_checked = int(config['verifier']['params']['time_out_checked'])
		self.rounding = int(config['verifier']['params']['rounding'])
		self.interval_propagation_type = config['verifier']['params']['interval_propagation_type']
		self.memory_limit = int(config['verifier']['params']['memory_limit'])
		self.disk_limit = int(config['verifier']['params']['disk_limit'])

		# Private variables
		self.unchecked_area = 100
		self._split_matrix = self._generate_splitmatrix()
		
		# Propagation method selection, there are different propagtion method for the different running mode: CPU or GPU
		if self.cpu_only: 
			self._propagation_method = prop_utility.multi_area_propagation_cpu
		else: 
			self._propagation_method = prop_utility.multi_area_propagation_gpu

		if self.memory_limit == 0:
			self.memory_limit = psutil.virtual_memory().available
		if self.disk_limit == 0:
			self.disk_limit = psutil.disk_usage('/').free


		self.violation_rate = None
		self.safe_rate = None
		self.property_sat = None


	def verify(self, verbose):
		"""
		Method that perform the formal analysis. When the solver explored and verify all the input domain it returns the
		violation rate, when the violation rate is zero we coclude that the original property is fully respected, i.e., SAT

		Parameters
		----------
			verbose : int
				when verbose > 0 the software print some log informations

		Compute
		----------
		property_sat : bool
			true if the property P is verified on the given network, false otherwise

		violation_rate: float
			the input domain's volume that violates the safety property

		safe_rate: float
			the input domain's volume for which the property holds
		
		"""
		
		folder_path = "./ProVe_partial_results"

		if os.path.isdir(folder_path):
			shutil.rmtree(folder_path)
		
		os.mkdir(folder_path)

		# Flatten the input domain to aobtain the areas matrix to simplify the splitting
		areas_matrix = np.array([self.input_predicate.flatten()])

		disk_utils.write_array(areas_matrix, 0, folder_path)
		free_file_index = 1

		# Array with the number of violations for each depth level
		violation_rate_array = []
		
		# Loop until all the subareas are eliminated (verified) or a counter example is found

		start_time = time.time()
		for cycle in range(self.time_out_cycle):

			if verbose > 0: 
				print(f"\n{gen_utilities.bcolors.BOLD}{gen_utilities.bcolors.WARNING}Iteration cycle {cycle:3d}/{self.time_out_cycle:3d} (checked area  {100-self.unchecked_area:5.3f}%){gen_utilities.bcolors.ENDC}")

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
				test_domain = areas_matrix.reshape(-1, self.input_predicate.shape[0], 2)


				# Call the propagation method to obtain the output bound from the input area (primal and dual)
				test_bound = self._propagation_method(test_domain, self.network, self.interval_propagation_type, self.memory_limit)

				# Call the verifier (N(x) >= 0) on all the subareas
				unknown_id, violated_id, proved_id = self._complete_verifier(test_bound)

				checked_size += proved_id[0].shape[0] + violated_id[0].shape[0]
				violated_length += len(violated_id[0])

				# Iterate only on the unverified subareas
				areas_matrix = areas_matrix[unknown_id]

				# Exit check when all the subareas are verified
				if areas_matrix.shape[0] > 0: no_areas_left = False

				# Save the current partial results to disk
				disk_utils.write_array(areas_matrix, iter, folder_path)

				iter += 1

			# Call the updater for the checked area
			self._update_unchecked_area(cycle, checked_size)

			# Update the violation rate array to compute the violation rate
			violation_rate_array.append(violated_length)
			
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
				if verbose > 0: 
					print(f"\tTime elapsed: {time_execution} min")
					print(f"\tVR (considering also the unknow areas as unsafe): {round(violation_rate,5)}%")
				
				
				violations_weigth = sum( [ 2**i * n for i, n in enumerate(reversed(violation_rate_array[:-1] + [violation_rate_array[-1]]))] ) 
				violation_rate =  violations_weigth / 2**(len(violation_rate_array ) -1) * 100 
			
				if verbose > 0: 
					print(f"\tUnderestimated VR (considering the unknow areas as safe): {round(violation_rate, 4)}%")
					print(f"\tEstimated VR using sampling: {round(self.estimated_VR*100, 4)}%")

			# Split the inputs (Iterative Refinement) specifying the maximum amount of memory
			free_file_index = self._split( folder_path, free_file_index, self.memory_limit)


		# Check if the exit reason is the time out on the cycle
		if cycle >= self.time_out_cycle:
			# return UNSAT with the no counter example, specifying the exit reason
			return False, { "counter_example" : None, "exit_code" : "cycle_timeout" }

		# Compute the violation rate, multipling the depth for the number for each violation
		# and normalizing for the number of theoretical leaf
		violations_weigth = sum( [ 2**i * n for i, n in enumerate(reversed(violation_rate_array))] ) 
		self.violation_rate =  violations_weigth / 2**(len(violation_rate_array)-1) * 100 

		# All the input are verified, return SAT with no counterexample
		self.property_sat = self.violation_rate == 0
		self.safe_rate = 100 - self.violation_rate

	def compute_bounds(self, input_domain):
		bounds = self._propagation_method(input_domain, self.network, self.interval_propagation_type, self.memory_limit)

		return bounds

	def _complete_verifier(self, test_bound):
		
		"""
		Method that verify the property on a list of the computed (or sampled in semi-formal mode) output bound.

		Parameters
		----------
			test_bound : list
				the output bound expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
				(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
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
		
		safe_regions = np.any(test_bound[:, :, 0] > 0, axis=1) # Property proved here!
		unsafe_regions = np.any(test_bound[:, :, 1] <= 0, axis=1) # Property violated here!

		# Create a mask for the unknown, when a property is neither proved or vioalted
		unknown_mask = np.logical_or(safe_regions, unsafe_regions)

		# Find the unknown and proved index with the built-in numpy function		
		unknown_id = np.where(unknown_mask==False)
		unsafe_regions_id = np.where(unsafe_regions==True)
		safe_regions_id = np.where(safe_regions==True)

		return unknown_id, unsafe_regions_id, safe_regions_id
	

	def _update_unchecked_area(self, depth, verified_nodes):

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

			areas_matrix = res.reshape((len(res) * 2, self.input_predicate.shape[0] * 2))

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

		for index, el in enumerate(row.reshape(self.input_predicate.shape[0], 2)):

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
		n = (self.input_predicate.shape[0] * 2)
		split_matrix = []

		# Iterate over each node times upper and lower to update the splitting matrix, each matrix is an identity matrix with
		# a splitting value (0.5) on the coordinate of the interested node.
		for i in range(self.input_predicate.shape[0]):
			split_matrix_base = np.concatenate((np.identity(n, dtype="float32"), np.identity(n, dtype="float32")), 1)
			split_matrix_base[(i * 2) + 0][(i * 2) + 1] = 0.5 
			split_matrix_base[(i * 2) + 1][(i * 2) + 1] = (0.5 - rounding_fix)
			split_matrix_base[(i * 2) + 0][n + (i * 2)] = 0.5 
			split_matrix_base[(i * 2) + 1][n + (i * 2)] = (0.5 + rounding_fix)
			split_matrix.append(split_matrix_base)

		#
		return split_matrix
	

	def print_results(self):
		if self.property_sat:
			print(gen_utilities.bcolors.OKGREEN + "\nThe property is SAT!")
		else:
			print(gen_utilities.bcolors.BOLD + gen_utilities.bcolors.FAIL + "\nThe property is UNSAT!"+ gen_utilities.bcolors.ENDC)
			if self.compute_violation_rate:
				print(gen_utilities.bcolors.BOLD + gen_utilities.bcolors.FAIL + "\t"f"Violation rate: {self.violation_rate}%"+ gen_utilities.bcolors.ENDC)
			else:
				print(gen_utilities.bcolors.BOLD + gen_utilities.bcolors.FAIL + "\t"f"Safe rate: {self.safe_rate}%"+ gen_utilities.bcolors.ENDC)