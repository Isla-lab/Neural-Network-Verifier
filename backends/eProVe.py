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

class eProVe():

	def __init__(self, config):
		# Input parameters
		self.path_to_network = config['model']['path']
		self.network = torch.load(self.path_to_network)

		self.property = gen_utilities.create_property(config)[0]
		self.input_predicate = np.array(self.property["inputs"])
		self.output_predicate = np.array(self.property["outputs"])

		# Verification hyper-parameters
		self.enumerate_unsafe_regions = config['verifier']['params']['enumerate_unsafe_regions']
		self.estimation_points = config['verifier']['params']['estimation_points']
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

    

	def verify(self, verbose=False):

		if self.compute_only_estimation:
			pass

		else:
			root = Node(value=self.input_predicate, network=self.network, split_node_heu=self.split_node_heuristic, split_pos_heu=self.split_pos_heuristic, max_depth=self.max_depth, unsafe_regions=self.enumerate_unsafe_regions, rate_tolerance_probability=self.rate_tolerance_probability)

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
				if frontier == []: break

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
			
