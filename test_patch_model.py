import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 input channel, 8 output channels, 3x3 kernel
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8 input channels, 16 output channels, 3x3 kernel
        # Define the output layer
        self.fc = nn.Linear(16 * 4, 3)  # Fully connected layer from flattened 16 channels of 4x4 to 3 output nodes

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling over a 2x2 window

    def forward(self, x):
        # Pass the input through the first convolutional layer, apply ReLU, and then pool
        x = self.pool(F.relu(self.conv1(x)))
        # Pass the output through the second convolutional layer and apply ReLU
        x = F.relu(self.conv2(x))
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        # Pass through the output layer
        x = self.fc(x)
        return x


# Create the neural network model
model = ConvolutionalNetwork()

# Example input (4x4 with 1 channel, indicating grayscale image)
example_input = torch.randn(1, 1, 4, 4)  # Batch size of 1

# Forward pass to get the output
output = model(example_input)

print("Output of the network:", output)

torch.save(model, "test_patch_model.pt")