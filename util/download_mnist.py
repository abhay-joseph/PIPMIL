import requests
import gzip
import shutil

# MNIST URLs
train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

# Target paths
target_path_train_images = './data/mnist/train-images-idx3-ubyte.gz'
target_path_train_labels = './data/mnist/train-labels-idx1-ubyte.gz'
target_path_test_images = './data/mnist/t10k-images-idx3-ubyte.gz'
target_path_test_labels = './data/mnist/t10k-labels-idx1-ubyte.gz'

# Download and extract training images
response = requests.get(train_images_url, stream=True)
if response.status_code == 200:
    with open(target_path_train_images, 'wb') as f:
        f.write(response.raw.read())
    with gzip.open(target_path_train_images, 'rb') as f_in, open(target_path_train_images[:-3], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Download and extract training labels
response = requests.get(train_labels_url, stream=True)
if response.status_code == 200:
    with open(target_path_train_labels, 'wb') as f:
        f.write(response.raw.read())
    with gzip.open(target_path_train_labels, 'rb') as f_in, open(target_path_train_labels[:-3], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Download and extract test images
response = requests.get(test_images_url, stream=True)
if response.status_code == 200:
    with open(target_path_test_images, 'wb') as f:
        f.write(response.raw.read())
    with gzip.open(target_path_test_images, 'rb') as f_in, open(target_path_test_images[:-3], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Download and extract test labels
response = requests.get(test_labels_url, stream=True)
if response.status_code == 200:
    with open(target_path_test_labels, 'wb') as f:
        f.write(response.raw.read())
    with gzip.open(target_path_test_labels, 'rb') as f_in, open(target_path_test_labels[:-3], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
