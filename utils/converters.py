import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_KERAS'] = '1'
import warnings; warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.layers import Conv2D
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
import torch

# Define a PyTorch model that replicates the Keras model
class PyTorchModel(nn.Module):
		def __init__(self, keras_model):
			super(PyTorchModel, self).__init__()

			self.nLayers = 1
			self.activations = []
			self.keras_model = keras_model

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
				if self.keras_model.layers[i].activation.__name__ == 'relu':
					x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append(F.relu)
				elif self.keras_model.layers[i].activation.__name__ == 'softmax':
					x = F.softmax(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append(F.softmax)
				elif self.keras_model.layers[i].activation.__name__ == 'sigmoid':
					x = F.sigmoid(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append(F.sigmoid)
				elif self.keras_model.layers[i].activation.__name__ == 'tanh':
					x = F.tanh(getattr(self, f'fc{i}')(x).to(x.dtype))
					self.activations.append(F.tanh)
				else:
					x = getattr(self, f'fc{i}')(x).to(x.dtype)
					self.activations.append(F.linear)
				
			return x

def convert_keras_to_pytorch(keras_model_path):

	keras_model = tf.keras.models.load_model(keras_model_path, compile=False)

	# Create an instance of the PyTorch model
	pytorch_model = PyTorchModel(keras_model)

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
	
    if config['model']['type'] == 'keras':
        torch_model = convert_keras_to_pytorch(config['model']['path'])
        path = 'torch_model.pth'
        torch.save(torch_model, path)
        config['model']['path'] = path
       
    return config['model']['path']
