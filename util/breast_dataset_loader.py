"""Pytorch Dataset object that loads 32x32 patches that contain single cells."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from skimage import io, color
from skimage.util import view_as_blocks
from sklearn.model_selection import StratifiedKFold

import PIPNet.util.utils_augemntation as utils_augemntation


class BreastCancerBagsCross(data_utils.Dataset):
    def __init__(self, path, train=True, test=False, push=False, shuffle_bag=False, data_augmentation=False,
                 loc_info=False, folds=10, fold_id=1, random_state=3, all_labels=False):
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

        self.data_augmentation_img_transform = transforms.Compose([utils_augemntation.RandomHEStain(),
                                                                   utils_augemntation.HistoNormalize(),
                                                                   utils_augemntation.RandomRotate(),
                                                                   transforms.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.RandomCrop(32, padding=(3, 3),
                                                                                         padding_mode='reflect'),
                                                                   transforms.ToTensor(),
                                                                   ])

        self.normalize_to_tensor_transform = transforms.Compose([utils_augemntation.HistoNormalize(),
                                                                 transforms.ToTensor(),
                                                                 ])
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])

        self.dir_list = self.get_dir_list(self.path)

        folds = list(
            StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state).split(self.dir_list, [
                1 if 'malignant' in d else 0 for d in self.dir_list]))

        if self.test:
            indices = set(folds[self.fold_id][1])
        else:
            if self.train:
                val_indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]))
                indices = set(folds[self.fold_id][0]) - set(val_indices)
            else:  # valid
                indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]))
        self.bag_list, self.labels_list = self.create_bags(np.asarray(self.dir_list)[list(indices)])

    @staticmethod
    def get_dir_list(path):
        import glob
        dirs = glob.glob(path + '/*.tif')
        dirs.sort()
        return dirs

    def create_bags(self, dir_list):
        bag_list = []
        labels_list = []
        for dir in dir_list:
            img = io.imread(dir)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            bag = view_as_blocks(img, block_shape=(32, 32, 3)).reshape(-1, 32, 32, 3)

            # store single cell labels
            label = 1 if 'malignant' in dir else 0

            # shuffle
            if self.shuffle_bag:
                random.shuffle(bag)

            bag_list.append(bag)
            labels_list.append(label)
        return bag_list, labels_list

    def transform_and_data_augmentation(self, bag, raw=False):
        if raw:
            img_transform = self.to_tensor_transform
        elif not raw and self.data_augmentation:
            img_transform = self.data_augmentation_img_transform
        else:
            img_transform = self.normalize_to_tensor_transform

        bag_tensors = []
        for img in bag:
            if self.location_info:
                bag_tensors.append(torch.cat(
                    (img_transform(img[:, :, :3]),
                     torch.from_numpy(img[:, :, 3:].astype(float).transpose((2, 0, 1))).float(),
                     )))
            else:
                bag_tensors.append(img_transform(img))
        return torch.stack(bag_tensors)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bag_list[index]

        if self.all_labels:
            label = torch.LongTensor([self.labels_list[index]] * bag.shape[0])
        else:
            label = torch.LongTensor([self.labels_list[index]]).max().unsqueeze(0)

        if self.push:
            return self.transform_and_data_augmentation(bag, raw=True), self.transform_and_data_augmentation(
                bag), label
        else:
            return self.transform_and_data_augmentation(bag), label


if __name__ == '__main__':
    ds = BreastCancerBagsCross(path="../data/Bisque/Images", train=False, all_labels=True, fold_id=1,
                               folds=10, random_state=3, push=True)
    for i, (raw, proc, _) in enumerate(ds):
        for idx, (r, p) in enumerate(zip(raw, proc)):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title('raw')
            ax1.imshow(r.permute(1, 2, 0))
            ax2.set_title('processed')
            ax2.imshow(p.permute(1, 2, 0))
            fig.savefig('../test/{}_{}.jpg'.format(i, idx))
