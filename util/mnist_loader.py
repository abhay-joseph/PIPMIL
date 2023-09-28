import os
import shutil
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image


# Define the target path
path = './data/MNIST/dataset/'

if not os.path.isdir(path):
    os.mkdir(path)

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = MNIST(root='./data/MNIST', train=True, transform=transform, download=True)
mnist_classes = [str(i) for i in range(10)]

# Organize the training images
training_output = os.path.join(path, 'train/')
if not os.path.exists(training_output):
    os.mkdir(training_output)

for idx, (image, label) in enumerate(mnist_dataset):
    label = str(label)
    output_path = os.path.join(training_output, label)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_image(image,os.path.join(output_path, f"{idx}.png"))

# Organize the testing images
mnist_dataset = MNIST(root='./data/MNIST', train=False, transform=transform, download=True)
testing_output = os.path.join(path, 'test/')
if not os.path.exists(testing_output):
    os.mkdir(testing_output)

for idx, (image, label) in enumerate(mnist_dataset):
    label = str(label)
    output_path = os.path.join(testing_output, label)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_image(image,os.path.join(output_path, f"{idx}.png"))

print('MNIST dataset processed!')
