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
from backends.ProVe import ProVe
import utils.general_utils as gen_utilities
from utils.propagation import get_optimized_estimation
from utils.propagation import get_estimation
import numpy as np
from tqdm import tqdm

class CountingProVe():

    def __init__(self, config):
        # Input parameters
        self.path_to_network = config['model']['path']
        self.network = torch.load(self.path_to_network)

        self.property = gen_utilities.create_property(config)[0]
        self.input_predicate = np.array(self.property["inputs"])
        self.output_predicate = np.array(self.property["outputs"])

        # Verification hyper-parameters
        self.compute_violation_rate = config['property']['target_volume'] == 'unsafe'
        self.compute_only_lower_bound = config['verifier']['params']['compute_only_lower_bound']
        self.T = config['verifier']['params']['T']
        self.estimation_points = config['verifier']['params']['estimation_points']
        self.S = config['verifier']['params']['splits']
        self.rounding = int(config['verifier']['params']['rounding'])
        self.cpu_only = config['verifier']['params']['cpu_only']
        self.beta = config['verifier']['params']['beta']
        self.initial_area_size = self._compute_area_size()

        self.lower_violation_rate = 1
        self.lower_safe_rate = 1

        self.lower_bound = 0
        self.upper_bound = 1
        self.property_sat = False

    
    def verify(self, verbose):

        for t in tqdm(range(self.T)):

            violation_t = self.count()
            self.lower_violation_rate = min(self.lower_violation_rate, violation_t)
            if not self.compute_only_lower_bound:
                self.compute_violation_rate = False
                safe_t = self.count()
                self.lower_safe_rate = min(self.lower_safe_rate, safe_t)

        self.lower_bound = round(self.lower_violation_rate*100, 3)
        self.upper_bound = round(self.lower_safe_rate*100, 3)
        self.property_sat = self.lower_bound == 0

        
     
    def count(self):	

        initial_area = self._compute_area_size(self.input_predicate)
        self.property_copy = self.input_predicate.copy()
        node_index = -1
        
        for _s in range(self.S):

            node_index += 1
            if node_index > (self.property_copy.shape[0]-1): node_index = 0
                                    
            _, violated_points = get_optimized_estimation(neural_net=self.network, property=self.property_copy, points=self.estimation_points, violation_rate=self.compute_violation_rate)

            if violated_points.shape[0] < 2:
                median = np.random.uniform(self.property_copy[node_index][0], self.property_copy[node_index][1])
            else:	
                median = np.median(violated_points[:, node_index])

            random_side = np.random.randint(0, 2)
            self.property_copy[node_index][random_side] = median


        if not self.cpu_only:
            config = { 'model': { 'path': self.path_to_network }, 
                      'property': { 'domain': self.input_predicate},
                      'verifier': { 'params': 
                                   { 'compute_violation_rate': self.compute_violation_rate, 
                                    'estimation_points': self.estimation_points,
                                     'cpu_only': self.cpu_only,
                                        'time_out_cycle': 0,
                                        'time_out_checked': 0,
                                        'rounding': self.rounding,
                                        'interval_propagation_type': 'relaxation',
                                        'memory_limit': 0,
                                        'disk_limit': 0
                                       } } }
            
            verifier = ProVe(config)
            verifier.verify(verbose=1)
            rate_split = (verifier.violation_rate / 100)
            if not self.compute_violation_rate: 
                rate_split = 1 - rate_split
        else:
            rate_split, _ = get_optimized_estimation(neural_net=self.network, property=self.property_copy, points=self.estimation_points, violation_rate=self.compute_violation_rate)


        area_leaf = self._compute_area_size(self.property_copy)
        violated_leaf_points = area_leaf * rate_split
        ratio = violated_leaf_points / initial_area
        
        return min(2**(self.S-self.beta) * ratio, 1)


    def _compute_area_size(self, area=None):
        if area is None:
            area = self.input_predicate
        sides_sizes = [side[1] - side[0] for side in area]
        return np.prod(sides_sizes) 
    
    def compute_bounds(self):
        config = { 'model': { 'path': self.path_to_network }, 
                      'property': { 'domain': self.input_predicate},
                      'verifier': { 'params': 
                                   { 'compute_violation_rate': self.compute_violation_rate, 
                                    'estimation_points': self.estimation_points,
                                     'cpu_only': self.cpu_only,
                                        'time_out_cycle': 0,
                                        'time_out_checked': 0,
                                        'rounding': self.rounding,
                                        'interval_propagation_type': 'relaxation',
                                        'memory_limit': 0,
                                        'disk_limit': 0
                                       } } }
            
        verifier = ProVe(config)
        bounds = verifier.compute_bounds(config['property']['domain'])
        return bounds

    def print_results(self):
         
        if self.property_sat:
            print(gen_utilities.bcolors.OKGREEN + "\nThe property is SAT!")
        else:
            print(gen_utilities.bcolors.BOLD + gen_utilities.bcolors.FAIL + "\nThe property is UNSAT!"+ gen_utilities.bcolors.ENDC)
                
        print(f"\tConfidence: {round((1 - 2 ** (-self.beta*(self.T)))*100, 2)}%")
        print(f"\tLower bound {'VR' if self.compute_violation_rate else 'SR'}: {self.lower_bound}%" )
        if not self.compute_only_lower_bound: 
            print(f"\tUpper bound {'VR' if self.compute_violation_rate else 'SR'}: {100 - self.upper_bound}%" )
            print(f"\tSize of the interval: {round((100 - self.upper_bound) - self.lower_bound, 2)}%" )
