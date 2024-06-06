import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_KERAS'] = '1'
import warnings; warnings.filterwarnings("ignore")
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch

# Define a PyTorch model that replicates the Keras model
class PyTorchModel(nn.Module):
		def __init__(self, info_keras):
			super(PyTorchModel, self).__init__()

			self.nLayers = info_keras['n_layers']
			self.input_shape = info_keras['input_shape']
			self.layers = info_keras['layers']
			self.activations = info_keras['activations']
			self.activations_torch = []
			
			# Define the layers
			self.add_module(f"fc1", nn.Linear(in_features=self.input_shape,  
										out_features=self.layers[0]))
			
			for i in range(2, len(self.layers)+1):
				self.add_module(f"fc{i}", nn.Linear(in_features=self.layers[i-2],  
										out_features=self.layers[i-1]))
			
            
		def forward(self, x):
			# Define the forward pass
			for i in range(1, self.nLayers):
				if self.activations[i-1] == 'relu':
					x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations_torch.append(F.relu)
				elif self.activations[i-1] == 'softmax':
					x = F.softmax(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations_torch.append(F.softmax)
				elif self.activations[i-1] == 'sigmoid':
					x = F.sigmoid(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations_torch.append(F.sigmoid)
				elif self.activations[i-1] == 'tanh':
					x = F.tanh(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations_torch.append(F.tanh)
				else:
					x = getattr(self, f'fc{i}')(x).to(x.dtype)
					self.activations_torch.append(F.linear)
				
			return x

def convert_keras_to_pytorch(keras_model_path):

	keras_model = tf.keras.models.load_model(keras_model_path, compile=False)

	info = {'n_layers': len(keras_model.layers),
		 'input_shape': keras_model.input_shape[1],
		 'layers': keras_model.layers[1:],
		 'activations': [layer.activation.__name__ for layer in keras_model.layers[1:]]}
	
	for layer in keras_model.layers[1:]:
		info['layers'][info['layers'].index(layer)] = layer.get_config()['units']

	
	# Create an instance of the PyTorch model
	pytorch_model = PyTorchModel(info)
	
    # Copy the weights and biases from the Keras model to the PyTorch model
	keras_weights = [layer.get_weights() for layer in keras_model.layers[1:]]
	for i, layer in enumerate(pytorch_model.children()):
		if isinstance(layer, nn.Linear):
			layer.weight = nn.Parameter(torch.tensor(keras_weights[i][0].T, requires_grad=True))
			layer.bias = nn.Parameter(torch.tensor(keras_weights[i][1], requires_grad=True))

		elif isinstance(layer, nn.Conv2d):
			layer.weight = nn.Parameter(torch.tensor(keras_weights[i][0].transpose(3, 2, 0, 1), requires_grad=True))
			layer.bias = nn.Parameter(torch.tensor(keras_weights[i][1], requires_grad=True))

	torch.save(pytorch_model, 'temp_files/torch_model.pth')


def get_netver_model(config):

	"""
	Method the convert any PyTorch model into a NetVer encoded model with one single output.

	Parameters
	----------
		torch_model : any PyTorch model

	Returns:
	--------
		netver_model : torch model with augmented output layer
	"""
	
	# load/ convert original model to torch
	if config['model']['type'] == 'keras':
		convert_keras_to_pytorch(config['model']['path'])
		path = 'temp_files/torch_model.pth'
		config['model']['path'] = path
	   
	# compute properties
	# get original output shape
	model = torch.load(config['model']['path'])
	output_shape = model(torch.rand(1, model.input_shape)).shape[1]
	config['model']['output_shape'] = output_shape

	# convert original model to NetVer model
	#TODO: implement conversion

	return config['model']['path']
