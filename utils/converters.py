import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_KERAS'] = '1'
import warnings; warnings.filterwarnings("ignore")
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch
import onnx
from onnx2torch import convert


# Define a pytorch model to convert multiple output nodes to a single one based on the property to verify
class NetVerModel(nn.Module):
    def __init__(self, model, bounds):
        super(NetVerModel, self).__init__()
        self.model = model
        self.bounds = bounds

        self.bounds_layer = nn.Linear(len(self.bounds[0]), len(self.bounds[0]) * 2 * len(bounds))
        self.junction_layer = nn.Linear(len(self.bounds[0]) * 2 * len(bounds), len(self.bounds))

        self.relu = nn.ReLU()

        nn.init.constant_(self.bounds_layer.weight, 0)
        nn.init.constant_(self.bounds_layer.bias, 0)

        nn.init.constant_(self.junction_layer.weight, 0)
        nn.init.constant_(self.junction_layer.bias, 0)

        for disjunction_idx, disjunction in enumerate(self.bounds):
            for bound_idx, bound in enumerate(disjunction):
                min_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2
                max_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2 + 1

                if isinstance(bound[0], float) and bound[0] > float('-inf'):
                    self.bounds_layer.weight.data[min_index][bound_idx] = -1.
                    self.bounds_layer.bias.data[min_index] = bound[0]
                elif isinstance(bound[0], str):
                    node_idx = int(bound[0].split('_')[1])
                    self.bounds_layer.weight.data[min_index][node_idx] = 1.
                    self.bounds_layer.weight.data[min_index][bound_idx] = -1.

                if isinstance(bound[1], float) and bound[1] < float('inf'):
                    self.bounds_layer.weight.data[max_index][bound_idx] = 1.
                    self.bounds_layer.bias.data[max_index] = -bound[1]
                elif isinstance(bound[1], str):
                    node_idx = int(bound[1].split('_')[1])
                    self.bounds_layer.weight.data[max_index][node_idx] = -1.
                    self.bounds_layer.weight.data[max_index][bound_idx] = 1.

        for disjunction_idx in range(len(self.bounds)):
            for bound_idx in range(len(self.bounds[0])):
                min_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2
                max_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2 + 1

                self.junction_layer.weight.data[disjunction_idx][min_index] = 1.
                self.junction_layer.weight.data[disjunction_idx][max_index] = 1.


    def forward(self, x):
        outputs = self.model.forward(x)

        outputs = self.bounds_layer(outputs)
        outputs = self.relu(outputs)

        outputs = self.junction_layer(outputs)

        output = torch.min(outputs, dim=1).values

        return output


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


def convert_model(config):

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
    elif config['model']['type'] == 'onnx':
        onnx_model = onnx.load(config['model']['path'])
        torch_model = convert(onnx_model)
        path = "./temp_files/CIFAR100_resnet_large.pth"
        torch.save(torch_model, path)
        config['model']['path'] = path
    else:
        pass

    return config['model']['path']
