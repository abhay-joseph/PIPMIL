"""Pytorch Dataset object that loads 32x32 patches that contain single cells."""

import random
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from skimage import io, color
from skimage.util import view_as_blocks
from sklearn.model_selection import StratifiedKFold

# import utils_augemntation


class BreastCancerBagsCross(data_utils.Dataset):
    def __init__(self, path, train=True, test=False, push=False, shuffle_bag=False, data_augmentation=False,
                 loc_info=False, folds=10, fold_id=1, random_state=207121037, all_labels=False):
        self.path = path
        self.train = train
        self.test = test
        self.folds = folds
        self.fold_id = fold_id
        self.random_state = random_state
        self.push = push
        self.all_labels = all_labels
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.location_info = loc_info
        self.r = np.random.RandomState(random_state)

        # self.data_augmentation_img_transform = transforms.Compose([utils_augemntation.RandomHEStain(),
        #                                                            utils_augemntation.HistoNormalize(),
        #                                                            utils_augemntation.RandomRotate(),
        #                                                            transforms.RandomVerticalFlip(),
        #                                                            transforms.RandomHorizontalFlip(),
        #                                                            transforms.RandomCrop(32, padding=(3, 3),
        #                                                                                  padding_mode='reflect'),
        #                                                            transforms.ToTensor(),
        #                                                            ])

        # self.normalize_to_tensor_transform = transforms.Compose([utils_augemntation.HistoNormalize(),
        #                                                          transforms.ToTensor(),
        #                                                          ])
        # self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])

        self.dir_list = self.get_dir_list(self.path)

        folds = list(
            StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state).split(self.dir_list, [
                1 if 'malignant' in d else 0 for d in self.dir_list]))

        if self.test:
            output_path = "./data/Bisque/dataset/test/"
            indices = set(folds[self.fold_id][1])            
        else:
            if self.train:
                output_path = "./data/Bisque/dataset/train/"
                #val_indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]))
                #indices = set(folds[self.fold_id][0]) - set(val_indices)
                indices = set(folds[self.fold_id][0])
            else:  # valid
                indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]))
        self.bag_list, self.labels_list = self.create_bags_and_save(np.asarray(self.dir_list)[list(indices)],output_path)

    @staticmethod
    def get_dir_list(path):
        import glob
        dirs = glob.glob(path + '/*.tif')
        dirs.sort()
        return dirs

    def create_bags_and_save(self, dir_list, output_path):
        bag_list = []
        labels_list = []
        for i, dir in enumerate(dir_list):
            img = io.imread(dir)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            bags = view_as_blocks(img, block_shape=(32, 32, 3)).reshape(-1, 32, 32, 3)

            # store single cell labels
            label = 1 if 'malignant' in dir else 0

            # shuffle
            if self.shuffle_bag:
                random.shuffle(bags)

            for j, bag in enumerate(bags):
                # Create the directory structure
                save_dir = os.path.join(output_path, str(label), f"bag_{i}")
                os.makedirs(save_dir, exist_ok=True)

                # Save the bag
                io.imsave(os.path.join(save_dir, f"{i}_{j}.jpg"), bag)

            bag_list.append(bags)
            labels_list.append(label)
        return bag_list, labels_list

    # def transform_and_data_augmentation(self, bag, raw=False):
    #     if raw:
    #         img_transform = self.to_tensor_transform
    #     elif not raw and self.data_augmentation:
    #         img_transform = self.data_augmentation_img_transform
    #     else:
    #         img_transform = self.normalize_to_tensor_transform

    #     bag_tensors = []
    #     for img in bag:
    #         if self.location_info:
    #             bag_tensors.append(torch.cat(
    #                 (img_transform(img[:, :, :3]),
    #                  torch.from_numpy(img[:, :, 3:].astype(float).transpose((2, 0, 1))).float(),
    #                  )))
    #         else:
    #             bag_tensors.append(img_transform(img))
    #     return torch.stack(bag_tensors)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bag_list[index]

        if self.all_labels:
            label = torch.LongTensor([self.labels_list[index]] * bag.shape[0])
        else:
            label = torch.LongTensor([self.labels_list[index]]).max().unsqueeze(0)

        # if self.push:
        #     return self.transform_and_data_augmentation(bag, raw=True), self.transform_and_data_augmentation(
        #         bag), label
        # else:
        #     return self.transform_and_data_augmentation(bag), label


if __name__ == '__main__':
    cwd = os.getcwd()
    print(cwd)
    ds = BreastCancerBagsCross(path="../ProtoMIL/data/Bisque/Images/", train=True, shuffle_bag=True, data_augmentation=True, fold_id=0, folds=10, random_state=207121037)
    ds = BreastCancerBagsCross(path="../ProtoMIL/data/Bisque/Images/", train=False, test=True, all_labels=True, fold_id=0, folds=10, random_state=207121037)
    
    