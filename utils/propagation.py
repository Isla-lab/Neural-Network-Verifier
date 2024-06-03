import numpy as np; 
import torch
import torch.nn.functional as F


def get_estimation(neural_net, property, points=3000, violation_rate=True):

	network_input = np.random.uniform(property[:, 0], property[:, 1], size=(points, property.shape[0]))
	network_input = torch.from_numpy(network_input).float()
	network_output = neural_net(network_input).detach().numpy()

	if violation_rate:
		where_indexes = np.where([network_output <= 0])[1]
	else:
		where_indexes = np.where([network_output > 0])[1]

	sat_points = network_input[where_indexes]
	rate = (len(where_indexes)/points)

	return rate, sat_points



def multi_area_propagation_cpu(input_domain, net_model, prop_type, memory_limit):

	"""
	Propagation of the input domain through the network to obtain the OVERESTIMATION of the output bound. 
	The process is performed applying the linear combination node-wise and the necessary activation functions.
	The process is on CPU, without any form of parallelization. 
	This function iterate over the function "single_area_propagation_cpu" that compute the propagation
	of a single input domain.

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
			(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
		net_model : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format

	Returns:
	--------
		reshaped_bound : list
			the propagated bound in the same format of the input domain (3-dim)
	"""

	# Iterate over every single domain of the input domain list and call the single_area_propagation_cpu function
	reshaped_bound = np.array([single_area_propagation_cpu(d, net_model) for d in input_domain])
	
	return reshaped_bound


def single_area_propagation_cpu(input_domain, net_model):

	"""
	Implementation of the real propagation of a single bound.
	Auxiliary function for the main 'multi_area_propagation_cpu' function.

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 2-dim matrix. (a) a list of bound for each input node and 
			(b) a list of two element for the node, lower and upper
		net_model : tf.keras.Model
			tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format

	Returns:
	--------
		entering : list
			the propagated bound in the same format of the input domain (2-dim)
	"""


	weights = [layer.weight.detach().numpy().T for layer in net_model.children()]
	biases = [layer.bias.detach().numpy() for layer in net_model.children()]
	activations = net_model.activations_torch

	# The entering values of the first iteration are the input for the propagation
	entering = input_domain

	# Iteration over all the layer of the network for the propagation
	for layer_id, layer in enumerate(weights):

		# Pre-computation for the linear propagation of the bounds
		max_ = np.maximum(layer, 0)
		l = entering[:, 0]

		# Pre-computation for the linear propagation of the bounds
		min_ = np.minimum(layer, 0)
		u = entering[:, 1]
		
		# Update the linear propagation with the standard formulation [Liu et. al 2021]
		l_new = np.dot(l, max_) + np.dot(u, min_) + biases[layer_id]
		u_new = np.dot(u, max_) + np.dot(l, min_) + biases[layer_id]

		
		# Check and apply the activation function 
		if activations[layer_id] is not F.linear:
			l_new = activations[layer_id](torch.tensor(l_new)).detach().numpy()
			u_new = activations[layer_id](torch.tensor(u_new)).detach().numpy()


		# Reshape of the bounds for the next iteration
		entering = np.concatenate([np.vstack(l_new), np.vstack(u_new)], axis=1)

	#
	return entering



def multi_area_propagation_gpu(input_domain, net_model, propagation, memory_limit, thread_number=2):

	"""
	Propagation of the input domain through the network to obtain the OVERESTIMATION of the output bound. 
	The process is performed applying the linear combination node-wise and the necessary activation functions.
	The process is on GPU, completely parallelized on NVIDIA CUDA GPUs and c++ code. 

	Parameters
	----------
		input_domain : list 
			the input domain expressed as a 3-dim matrix. (a) a list of list for each splitted domain;
			(b) a list of bound for each input node and (c) a list of two element for the node, lower and upper
		net_model : torch.nn.Module
			PyTorch model to analyze
		propagation : str
			propagation method
		thread_number : int
			number of CUDA thread to use for each CUDA block, the choice is free and does not effect the results, 
			can however effect the performance

	Returns:
	--------
		reshaped_bound : list
			the propagated bound in the same format of the input domain (3-dim)
	"""

	# Ignore the standard warning from CuPy
	import warnings
	warnings.filterwarnings("ignore")

	# Import the necessary library for the parallelization (Cupy) and also the c++ CUDA code.
	import cupy as cp

	if propagation == 'naive':
		from utils.cuda_code import cuda_code
	elif propagation == 'symbolic':
		from utils.cuda_code_symbolic import cuda_code
	else:
		from utils.cuda_code_linear_relaxation import cuda_code

	# Load network shape, activations and weights
	layer_sizes = []
	activations = net_model.activations_torch
	full_weights = np.array([])
	full_biases = np.array([])
	layer_sizes.append(net_model.fc1.weight.shape[1:][0])

	i = 0
	encoded_activations = []
	for layer in net_model.children():
		layer_sizes.append(layer.out_features)
		full_weights = np.concatenate((full_weights, layer.weight.detach().numpy().flatten()), axis=0)
		full_biases = np.concatenate((full_biases, layer.bias.detach().numpy()), axis=0)

		# Obtain the activation function list
		if activations[i] == F.linear: encoded_activations.append(0)
		elif activations[i] == F.relu: encoded_activations.append(1)
		elif activations[i] == F.tanh: encoded_activations.append(2)
		elif activations[i] == F.sigmoid: encoded_activations.append(3)
		i+=1

	# Initialize the kernel loading the CUDA code
	if propagation == 'naive':
		my_kernel = cp.RawKernel(cuda_code, 'my_kernel')
	elif propagation == 'symbolic':
		my_kernel = cp.RawKernel(cuda_code, 'my_kernel_symbolic')
	else:
		my_kernel = cp.RawKernel(cuda_code, 'my_kernel_relaxation')

	# Convert all the data in cupy array beore the kernel call
	max_layer_size = max(layer_sizes)

	batch_size = int(memory_limit / (input_domain.nbytes / len(input_domain)))
	batches = len(input_domain) // batch_size

	if input_domain.nbytes % batch_size != 0:
		batches += 1

	results_cuda = np.array([], dtype=np.float32)

	layer_sizes = cp.array(layer_sizes, dtype=cp.int32)
	activations = cp.array(encoded_activations, dtype=cp.int32)

	full_weights = cp.array(full_weights, dtype=cp.float32)
	full_biases = cp.array(full_biases, dtype=cp.float32)

	for iter in range(batches):
		if iter == batches - 1:
			upper_limit = len(input_domain)
		else:
			upper_limit = batch_size * (iter + 1)

		subinput_domain = cp.array(input_domain[batch_size * iter : upper_limit], dtype=cp.float32)
		subresults_cuda = cp.zeros(int(layer_sizes[-1] * 2 * len(subinput_domain)), dtype=cp.float32)

		# Define the number of CUDA block
		block_number = int(len(subinput_domain) / thread_number) + 1

		# Create and launch the kernel, wait for the sync of all threads
		kernel_input = (subinput_domain, len(subinput_domain), layer_sizes, len(layer_sizes), full_weights, full_biases, subresults_cuda, max_layer_size, activations)
		my_kernel((block_number, ), (thread_number, ), kernel_input)
		cp.cuda.Stream.null.synchronize()

		results_cuda = np.concatenate((results_cuda, cp.asnumpy(subresults_cuda)), axis=0)

		cp.get_default_memory_pool().free_all_blocks()

	cp.get_default_memory_pool().free_all_blocks()

	# Reshape the results and convert in numpy array
	reshaped_bound = results_cuda.reshape((len(input_domain), 1, 2))

	return reshaped_bound

	