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

import torch
import utils.general_utils as gen_utilities
from utils.backends_utils import Node
import numpy as np
from tqdm import tqdm

class eProVe:

	def __init__(self, config, prop):
		# Input parameters
		self.path_to_network = config['model']['path']
		self.network = torch.load(self.path_to_network)

		self.property = prop
		self.input_predicate = np.array(self.property["inputs"])
		self.output_predicate = np.array(self.property["outputs"])

		# Verification hyperparameters
		if 'alpha' in config['verifier']['params']:
			self.confidence = config['verifier']['params']['alpha']
			self.R = config['verifier']['params']['R']
			self.estimation_points = int(np.emath.logn(self.R, (1-self.confidence)))
		else:
			self.estimation_points = config['verifier']['params']['estimation_points']
			self.R = config['verifier']['params']['R']
			self.confidence = 1 - (self.R**self.estimation_points)


		self.enumerate_unsafe_regions = config['property']['target_volume'] == 'unsafe'
		self.compute_only_estimation = config['verifier']['params']['compute_only_estimation']
		self.max_depth = config['verifier']['params']['max_depth']
		self.split_node_heuristic = config['verifier']['params']['split_node_heuristic']
		self.split_pos_heuristic = config['verifier']['params']['split_pos_heuristic']
		self.rate_tolerance_probability = config['verifier']['params']['rate_tolerance_probability']

		# Private config variables
		self._time_out_cycle = 25
		self.property_sat = False
		self.total_rate = 0
		self.info = {}

    

	def verify(self):

		if self.compute_only_estimation:
			pass

		else:
			root = Node(value=self.input_predicate, network=self.network, split_node_heuristic=self.split_node_heuristic, split_pos_heuristic=self.split_pos_heuristic, max_depth=self.max_depth, enumerate_unsafe_regions=self.enumerate_unsafe_regions, rate_tolerance_probability=self.rate_tolerance_probability)

			frontier = [root]
			next_frontier = []
			areas_verified = []

			for depth in tqdm(range(self._time_out_cycle)):

				# Itearate over the nodes in the frontier
				for node in frontier:

					# If the node is verified, add it to the safe areas list
					if node.get_probability(self.estimation_points) == 1: 
						areas_verified.append( node )
						continue

					# If the node passes the test, split into the two children
					if node.expansion_test():
						child_1, child_2 = node.get_children()
						next_frontier.append(child_1)
						next_frontier.append(child_2)

				# The tree has been completely explored
				if not frontier: break

				# Compute the underestimation of the safe areas and the safe rate
				regions_size = sum([ subarea.compute_area_size() for subarea in areas_verified ])
				total_size = root.compute_area_size()
				total_rate = regions_size / total_size
		
				# Update the frontier with the new explored nodes
				frontier.clear()
				frontier = next_frontier.copy()
				next_frontier.clear()


			# Compute the underestimation of the safe areas and the safe rate
			regions_size = sum([ subarea.compute_area_size() for subarea in areas_verified ])
			total_size = root.compute_area_size()
			self.total_rate = regions_size / total_size

			# Create a dictionary with the additional info
			self.info = { 
				'areas': [subarea.value for subarea in areas_verified], 
				'areas-object': [subarea for subarea in areas_verified], 
				'areas-number': len([subarea.value for subarea in areas_verified]),
				'depth-reached': depth
			}

			self.property_sat = self.total_rate == 0

	def compute_bounds(self):
		network_input = np.random.uniform(self.input_predicate[:, 0], self.input_predicate[:, 1], size=(3_000_000, self.input_predicate.shape[0]))
		network_input = torch.from_numpy(network_input).float()
		network_output = self.network(network_input).detach().numpy()

		return [min(network_output).item(), max(network_output).item()]
			
	def print_results(self):
		if self.property_sat:
			print(gen_utilities.bcolors.OKGREEN + "\nThe property is SAT!")
		else:
			print(gen_utilities.bcolors.BOLD + gen_utilities.bcolors.FAIL + "\nThe property is UNSAT!"+ gen_utilities.bcolors.ENDC)

		print(f"\tLower bound {'VR' if self.enumerate_unsafe_regions else 'SR'}: {self.total_rate*100}%")
		print(f"\tNum {'unsafe' if self.enumerate_unsafe_regions else 'safe'} areas {self.info['areas-number']}")	
		print(f"\tEach {'unsafe' if self.enumerate_unsafe_regions else 'safe'} area presents at most {round((1-self.R)*100, 2)}% of {'safe' if self.enumerate_unsafe_regions else 'unsafe'} volume")
		print(f"\tConfidence: {round(self.confidence*100, 8)}%")
		print(f"\tEstimation points used: {self.estimation_points}\n")
		
		
		

