import numpy as np
import torch
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten
import torch.nn as nn
import torch.nn.functional as F

def convert_keras_to_pytorch(keras_model):

    # Define a PyTorch model that replicates the Keras model
	class PyTorchModel(nn.Module):
		def __init__(self):
			super(PyTorchModel, self).__init__()

			self.nLayers = 1
			self.activations = []

			# Define the first layer
			if isinstance(keras_model.layers[0], Conv2D):
				self.add_module("fc1", nn.Conv2d(in_channels=keras_model.layers[1].input_shape[1],  
										out_channels=keras_model.layers[1].filters, 
										kernel_size=keras_model.layers[1].kernel_size, 
										stride=keras_model.layers[1].strides, 
										padding=keras_model.layers[1].padding, 
										bias=keras_model.layers[1].use_bias))
			else:
				self.add_module("fc1", nn.Linear(in_features=keras_model.layers[1].input.shape[1],  
									out_features=keras_model.layers[1].units, 
									bias=keras_model.layers[1].use_bias))
				
			# Define the rest of the layers 
			i = 2
			for layer in keras_model.layers[2:]:
				if isinstance(layer, Conv2D):
					self.add_module(f"fc{i}", nn.Conv2d(in_channels=layer.input_shape[1],  
										out_channels=layer.filters, 
										kernel_size=layer.kernel_size, 
										stride=layer.strides, 
										padding=layer.padding, 
										bias=layer.use_bias))
				else:
					self.add_module(f"fc{i}", nn.Linear(in_features=layer.input.shape[1],  
										out_features=layer.units, 
										bias=layer.use_bias))
				i += 1
				self.nLayers += 1
            
		def forward(self, x):
			# Define the forward pass
			for i in range(1, self.nLayers + 1):
				if keras_model.layers[i].activation.__name__ == 'relu':
					x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append(F.relu)
				elif keras_model.layers[i].activation.__name__ == 'softmax':
					x = F.softmax(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append('softmax')
				elif keras_model.layers[i].activation.__name__ == 'sigmoid':
					x = F.sigmoid(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append('sigmoid')
				elif keras_model.layers[i].activation.__name__ == 'tanh':
					x = F.tanh(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append('tanh')
				else:
					x = getattr(self, f'fc{i}')(x).to(x.dtype)
					self.activations.append(F.linear)
				
			return x
         

	# Create an instance of the PyTorch model
	pytorch_model = PyTorchModel()
	keras_model.summary()
	print(pytorch_model)

    # Copy the weights and biases from the Keras model to the PyTorch model
	keras_weights = [layer.get_weights() for layer in keras_model.layers[1:]]
	for i, layer in enumerate(pytorch_model.children()):
		if isinstance(layer, nn.Linear):
			layer.weight = nn.Parameter(torch.tensor(keras_weights[i][0].T, requires_grad=True))
			layer.bias = nn.Parameter(torch.tensor(keras_weights[i][1], requires_grad=True))

		elif isinstance(layer, nn.Conv2d):
			layer.weight = nn.Parameter(torch.tensor(keras_weights[i][0].transpose(3, 2, 0, 1), requires_grad=True))
			layer.bias = nn.Parameter(torch.tensor(keras_weights[i][1], requires_grad=True))

	return pytorch_model



keras_model = tf.keras.models.load_model('model_5_09.h5')
pytorch_model = convert_keras_to_pytorch(keras_model)

# test conversion from keras to pytorch
input_vector = np.random.rand(1, 5)
keras_output = keras_model(input_vector).numpy().flatten()


input_tensor = torch.tensor(input_vector, dtype=torch.float32)
pytorch_output = pytorch_model(input_tensor)
pytorch_output = pytorch_output.detach().numpy().flatten()

print(keras_output)
print(pytorch_output)


# layer_sizes = []
# activations = []
# full_weights = np.array([])
# full_biases = np.array([])

# layer_sizes.append(pytorch_model.fc1.weight.shape[1:][0])

# for layer in pytorch_model.children():
# 	layer_sizes.append(layer.out_features)
# 	full_weights = np.concatenate((full_weights, layer.weight.detach().numpy().flatten()), axis=0)
# 	full_biases = np.concatenate((full_biases, layer.bias.detach().numpy()), axis=0)


# # print the necessary information
# print(layer_sizes)
# print(pytorch_model.activations)
# print(full_weights)
# print(full_biases)

# print("-----------")

# layer_sizes = []
# activations = []
# full_weights = np.array([])
# full_biases = np.array([])


# for layer in keras_model.layers[1:]:

# 	# Obtain the activation function list
# 	if layer.activation == tf.keras.activations.linear: 
# 		activations.append(0)
# 	elif layer.activation == tf.keras.activations.relu: 
# 		activations.append(1)
# 	elif layer.activation == tf.keras.activations.tanh: 
# 		activations.append(2)
# 	elif layer.activation == tf.keras.activations.sigmoid: 
# 		activations.append(3)

# 	# Obtain the netowrk shape as a list
# 	layer_sizes.append(layer.input.shape[1])

# 	# Obtain all the weights for paramters and biases
# 	weight, bias = layer.get_weights()
# 	full_weights = np.concatenate((full_weights, weight.T.reshape(-1)))
# 	full_biases = np.concatenate((full_biases, bias.reshape(-1)))

# # Fixe last layer size
# layer_sizes.append(1)    

# # print the necessary information
# print(layer_sizes)
# print(activations)
# print(full_weights)
# print(full_biases)



print("---------")

weights = [layer.weight.detach().numpy().T for layer in pytorch_model.children()]
biases = [layer.bias.detach().numpy() for layer in pytorch_model.children()]
activations = pytorch_model.activations

print(weights)
print(biases)
print(activations)

print("---------")

# weights = [ layer.get_weights()[0].T for layer in keras_model.layers[1:] ]
# biases = [ layer.get_weights()[1] for layer in keras_model.layers[1:]  ]
# activations = [ layer.activation for layer in keras_model.layers[1:] ]


# print(weights)
# print(biases)
# print(activations)


# The entering values of the first iteration are the input for the propagation
entering = np.array([[0,1], [0,1], [0,1], [0,1], [0,1]])

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
	print(entering)