import os
import random
import shutil
import numpy as np
import time
from PIL import Image
import math

random.seed(2023)

path = './data/CUB_200_2011_MIL/CUB_200_2011_MIL/'

time_start = time.time()

path_images = os.path.join(path, 'images.txt')
train_save_path = os.path.join(path, 'dataset/train_crop/')
test_save_path = os.path.join(path, 'dataset/test_crop/')
bbox_path = os.path.join(path, 'bounding_boxes.txt')

bag_size = 10
target_class = 'Black_footed_Albatross'

# folder to store bags
bags_path = os.path.join(path, 'bags/')
if not os.path.isdir(bags_path):
    os.makedirs(bags_path)

images = []
with open(path_images, 'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))

print("Images: ", images)

bboxes = dict()
with open(bbox_path, 'r') as bf:
    for line in bf:
        id, x, y, w, h = tuple(map(float, line.split(' ')))
        bboxes[int(id)] = (x, y, w, h)

num = len(images)
bag_index = 1
check_images = set()  # Used to keep track of sampled images

for k in range(math.ceil(float(num/bag_size))):
    id, fn = images[k][0].split(' ')
    id = int(id)
    file_name = fn.split('/')[0]

    bag = []
    for _ in range(bag_size):
        if len(check_images) != num:
            random_idx = random.choice([i for i in range(num) if i not in check_images])
            check_images.add(random_idx)
            bag.append(images[random_idx][0].split(' ')[1])

    is_positive = any(target_class in img_name for img_name in bag)

    bag_save_path = os.path.join(bags_path, f'bag_{bag_index}')
    if not os.path.isdir(bag_save_path):
        os.makedirs(bag_save_path)

    for img_name in bag:
        img = Image.open(os.path.join(os.path.join(path, 'images'), img_name)).convert('RGB')
        # x, y, w, h = bboxes[id]
        # cropped_img = img.crop((x, y, x+w, y+h))
        img.save(os.path.join(bag_save_path, img_name.split('/')[1]))

    bag_index += 1
    print(f'Bag {bag_index - 1}: Positive: {is_positive}')

time_end = time.time()
print('CUB200, %s!' % (time_end - time_start))
