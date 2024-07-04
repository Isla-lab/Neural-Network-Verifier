import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

# Transformation for the CIFAR-10 images
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to PyTorch tensor
])

# Function to load an image and transform it
def load_and_transform_image(image_path, transform):
    img = Image.open(image_path).convert('RGB')  # Open the image file
    img_t = transform(img)  # Apply the transformations
    return img_t

# Example usage
image_path = 'car.jpg'
img_tensor = load_and_transform_image(image_path, transform)

# Function to save an image and its label to a file
def save_random_image(tensor, file_path):
    # Save both the image tensor and label to a file
    torch.save(tensor, file_path)


# Save a random image and label to a file
file_path = 'tensor_image.pt'
save_random_image(img_tensor, file_path)
