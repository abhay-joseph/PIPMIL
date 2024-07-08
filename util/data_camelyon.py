import numpy as np
import argparse
import csv
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.optim
import torch.utils.data
import torchvision
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split
from util.utils_augemntation import HistoNormalize
from util.utils_augemntation import RandomHEStain
from util.utils_augemntation import RandomRotate

class MultipleInstanceDataset(Dataset):
    '''Custom Dataset class imitating Imagefolder format for MIL setup, for CAMELYON'''
    def __init__(self, root, transform=None, max_bag=512, random_state=3):
        self.max_bag = max_bag
        self.random_state = random_state
        self.root = root
        self.transform = transform
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.r = np.random.RandomState(random_state)

        self.bags = []  # List to store bag information
        self.imgs = []  # List of (image path, class_index) tuples 
        self.samples = []  # List of (image path, class_index) tuples for all instances

        for class_label, class_name in enumerate(self.classes):
            class_path = os.path.join(root, class_name)
            for bag_name in os.listdir(class_path):
                bag_path = os.path.join(class_path, bag_name)
                self.bags.append((bag_path, class_label))  # Store bag information
                instance_names = os.listdir(bag_path) # List of instances within select bag
                if len(instance_names) > self.max_bag:
                    indices = self.r.permutation(len(instance_names))[:self.max_bag] # trim number of instances to 20000
                else:
                    indices = np.arange(0, len(instance_names))
                select_instance_paths = [(os.path.join(bag_path, instance_names[index]), class_label) for index in indices] # List of (instance_path, bag_label) tuples
                self.imgs += select_instance_paths
                self.samples += select_instance_paths

    def __len__(self):
        return len(self.bags)
       
    def __getitem__(self, idx):
        bag_path, class_label = self.bags[idx]

        # Load all instances in the bag
        instances = [Image.open(os.path.join(bag_path, instance_name)) for instance_name in os.listdir(bag_path) if (os.path.join(bag_path, instance_name), class_label) in self.imgs]

        # Apply transformations if needed
        if self.transform:
            instances = [self.transform(instance) for instance in instances]

        # Stack images into single bag tensor
        instances = torch.stack(instances)

        # Determine the label based on the class label (negative or positive)
        bag_label = class_label  

        return instances, bag_label
    
    def get_instances_list(self):
        return self.imgs
    
class MILPretrainDataset(Dataset):
    '''Custom Dataset class to pretrain CAMELYON'''
    def __init__(self, root, transform=None, max_bag=1000, random_state=3):
        self.max_bag = max_bag
        self.random_state = random_state
        self.root = root
        self.transform = transform
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.r = np.random.RandomState(random_state)

        self.bags = []  # List to store bag information
        self.imgs = []  # List of (image path, class_index) tuples 
        self.samples = []  # List of (image path, class_index) tuples for all instances

        for class_label, class_name in enumerate(self.classes):
            class_path = os.path.join(root, class_name)
            for bag_name in os.listdir(class_path):
                bag_path = os.path.join(class_path, bag_name)
                self.bags.append((bag_path, class_label))  # Store bag information
                instance_names = os.listdir(bag_path) # List of instances within select bag
                if len(instance_names) > self.max_bag:
                    indices = self.r.permutation(len(instance_names))[:self.max_bag] # trim number of instances to 20000
                else:
                    indices = np.arange(0, len(instance_names))
                select_instance_paths = [(os.path.join(bag_path, instance_names[index]), class_label) for index in indices] # List of (instance_path, bag_label) tuples
                self.imgs += select_instance_paths
                self.samples += select_instance_paths

    def __len__(self):
        return len(self.imgs)
       
    def __getitem__(self, idx):
        img_path, class_label = self.imgs[idx]
        
        instance = Image.open(img_path)

        # Apply transformations if needed
        if self.transform:
            instance = self.transform(instance)

        # Determine the label based on the class label (negative or positive)
        bag_label = class_label  

        return instance, bag_label

    def get_instances_list(self):
        return self.imgs


class CamelyonPreprocessedBagsCross(Dataset):
    def __init__(self, path, train=True, test=False, push=False, shuffle_bag=False, data_augmentation=False,
                 loc_info=False, folds=10, fold_id=1, random_state=3, all_labels=False, max_bag=20000):
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
        self.max_bag = max_bag
        self.labels = {}        
        for i in range(1,161):
            self.labels['normal_{:03d}.tif'.format(i)] = 0
        for i in range(1,112):
            self.labels['tumor_{:03d}.tif'.format(i)] = 1
        with open(os.path.join(path, 'reference.csv')) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                self.labels[str(row['slide'])+'.tif'] = 0 if row['label'] == 'Normal' else 1
        
        self.classes = list(set(self.labels.values()))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.embed_name = 'embeddings_resnet.pth'
        self.dir_list = [d for d in self.labels.keys() if os.path.exists(os.path.join(path, d, self.embed_name))]
        if self.train:
            self.dir_list = [d for d in self.dir_list if 'test' not in d]
            self.bags = [(os.path.join(path, k), v) for k, v in self.labels.items() if 'test' not in k and os.path.exists(os.path.join(path, k, self.embed_name))]
        else:
            self.dir_list = [d for d in self.dir_list if 'test' in d]
            self.bags = [(os.path.join(path, k), v) for k, v in self.labels.items() if 'test' in k and os.path.exists(os.path.join(path, k, self.embed_name))]
        
        self.imgs = []
        self.samples = []

        for bag_path, bag_label in self.bags:
            instance_paths = [(os.path.join(root, file), bag_label) for root, _, files in os.walk(bag_path) for file in files if file.endswith('.jpg')]
            if len(instance_paths) > self.max_bag:
                # rng = np.random.default_rng(random_state)
                indices = self.r.permutation(len(instance_paths))[:self.max_bag]
            else:
                indices = np.arange(0, len(instance_paths))
                
            select_instance_paths = [instance_paths[index] for index in indices]
            self.imgs += select_instance_paths
            self.samples += select_instance_paths 

    @classmethod
    def load_raw_image(cls, path):
        return to_tensor(pil_loader(path))

    class LazyLoader:
        def __init__(self, path, dir, indices):
            self.path = path
            self.dir = dir
            self.indices = indices

        def __getitem__(self, item):
            return CamelyonPreprocessedBagsCross.load_raw_image(
                os.path.join(self.path, self.dir, 'patch.{}.jpg'.format(int(self.indices[item]))))

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        dir = self.dir_list[index]
        try:
            bag = torch.load(os.path.join(self.path, dir, self.embed_name))
        except:
            print(dir)
            raise

        if bag.shape[0] > self.max_bag:
            indices = self.r.permutation(bag.shape[0])[:self.max_bag]
            bag = bag[indices].detach().clone()
        else:
            indices = np.arange(0, bag.shape[0])
            pad_size = self.max_bag - bag.shape[0]
            padding = torch.zeros(pad_size, *bag.shape[1:])
            bag = torch.cat([bag, padding], dim=0)
        # if self.all_labels:
        #     label = torch.LongTensor([self.labels[dir]] * bag.shape[0])
        # else:
        #     label = torch.LongTensor([self.labels[dir]]).max().unsqueeze(0)
        label = self.labels[dir]


        if self.push:
            instances = [CamelyonPreprocessedBagsCross.load_raw_image(
                os.path.join(self.path, dir, 'patch.{}.jpg'.format(int(index)))) for index in indices]
            instances = torch.stack(instances)
            return instances, label
        else:
            if self.train:
                return bag, bag, label
            else:
                return bag, label

# Custom Dataloader class
class MILBagLoader:
    def __init__(self, path, transform1=None, transform2=None, train=True, batch_size=32, shuffle=False, drop_last=False, random_state=3, max_bag=20000):
        self.path = path
        self.train = train
        self.transform1 = transform1
        self.transform2 = transform2
        self.random_state = random_state
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.r = np.random.RandomState(random_state)
        self.max_bag = max_bag

        self.full_labels = {}        
        for i in range(1,161):
            self.full_labels['normal_{:03d}.tif'.format(i)] = 0
        for i in range(1,112):
            self.full_labels['tumor_{:03d}.tif'.format(i)] = 1
        with open(os.path.join(path, 'reference.csv')) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                self.full_labels[str(row['slide'])+'.tif'] = 0 if row['label'] == 'Normal' else 1
        
        self.classes = list(set(self.full_labels.values()))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.embed_name = 'embeddings.pth'
        self.imgs = []
        self.samples = []

        # using pretrained embeddings
        #self.dir_list = [d for d in self.full_labels.keys() if os.path.exists(os.path.join(path, d, self.embed_name))]
        
        if self.train:
            self.bags = [(os.path.join(path, k), v) for k, v in self.full_labels.items() if 'test' not in k and os.path.exists(os.path.join(path, k))]
        else:
            self.bags = [(os.path.join(path, k), v) for k, v in self.full_labels.items() if 'test' in k and os.path.exists(os.path.join(path, k))]
            
        del self.full_labels 
        
        for bag_path, bag_label in self.bags:
            instance_paths = [(os.path.join(root, file), bag_label) for root, _, files in os.walk(bag_path) for file in files if file.endswith('.jpg')]
            if len(instance_paths) > self.max_bag:
                # rng = np.random.default_rng(random_state)
                indices = self.r.permutation(len(instance_paths))[:self.max_bag]
            else:
                indices = np.arange(0, len(instance_paths))
                
            select_instance_paths = [instance_paths[index] for index in indices]
            self.imgs += select_instance_paths
            self.samples += select_instance_paths
        
        self.indexes = np.arange(len(self.bags))
        if self.shuffle:
            self.r.shuffle(self.indexes)

        self.bag_paths = [paths for paths, _ in self.bags]
        self.bag_labels = [labels for _, labels in self.bags]  

    def __get_image(self, image):

        image = Image.open(image)

        if self.transform1:
            image = self.transform1(image)
        else:
            transform = transforms.Compose([ 
                transforms.PILToTensor() 
            ]) 
            image = transform(image) 

        return image

    def __get_image_2(self, image):

        if self.transform2:
            image = self.transform2(image)
            
        return image
        
    def __get_data(self, bags, labels):
        batch_bags = []
        batch_bags_2 = []
        batch_labels = labels
        for bag in bags:
            # loads images within each bag as list of tensors
            instances = [self.__get_image(instance_path) for instance_path, _ in self.imgs if bag in instance_path]
            
            # to return two augmentations in case of trainloader, trainloader_pretrain
            if self.transform2:
                instances_1 = [self.transform2(instance) for instance in instances]
                instances_2 = [self.transform2(instance) for instance in instances]
                instances_1 = torch.stack(instances_1)
                instances_2 = torch.stack(instances_2)
                
                # padding each bag tensor to len=20000 to maintain uniform dimensions across batch
                if len(instances_1) < self.max_bag:
                    pad_size = self.max_bag - len(instances_1)
                    padding = torch.zeros(pad_size, *instances_1.shape[1:]) # assuming images are 3D tensors (channels, height, width)
                    instances_1 = torch.cat([instances_1, padding], dim=0)
                if len(instances_2) < self.max_bag:
                    pad_size = self.max_bag - len(instances_2)
                    padding = torch.zeros(pad_size, *instances_2.shape[1:])  
                    instances_2 = torch.cat([instances_2, padding], dim=0)
                
                batch_bags.append(instances_1)
                batch_bags_2.append(instances_2)
                
                del instances
            
            else:
                instances = torch.stack(instances)
                if len(instances) < self.max_bag:
                    pad_size = self.max_bag - len(instances)
                    padding = torch.zeros(pad_size, *instances.shape[1:])
                    instances = torch.cat([instances, padding], dim=0)
                batch_bags.append(instances)

        if self.transform2:
            batch_bags_1 = torch.stack(batch_bags)
            batch_bags_2 = torch.stack(batch_bags_2)
            return batch_bags_1, batch_bags_2, batch_labels
        
        else:
            batch_bags = torch.stack(batch_bags)
            return batch_bags, batch_labels

    def __getitem__(self, index):
        # Loads select shuffled indices per itration
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        if self.drop_last and end_index > len(self.bags):
            raise IndexError("Index out of bounds, as drop_last is enabled.")
        
        indexes = self.indexes[start_index:end_index]

        batch_paths = [self.bag_paths[i] for i in indexes]
        batch_labels = [self.bag_labels[i] for i in indexes]

        if self.transform2:
            batch_1, batch_2, labels = self.__get_data(batch_paths, batch_labels)
            return batch_1, batch_2, labels
        else:
            batch, labels = self.__get_data(batch_paths, batch_labels)
            return batch, labels

    def __len__(self):
        if self.drop_last:
            return len(self.bags) // self.batch_size
        else:
            return (len(self.bags) + self.batch_size - 1) // self.batch_size
    
    def get_instances_list(self):
        return self.imgs

def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset =='CUB-200-2011':   
        return get_birds(True, './data/CUB_200_2011/dataset/train_crop', './data/CUB_200_2011/dataset/train', './data/CUB_200_2011/dataset/test_crop', args.image_size, args.seed, args.validation_size, './data/CUB_200_2011/dataset/train', './data/CUB_200_2011/dataset/test_full')

    if args.dataset == 'CAMELYON':
        # return get_camelyon('/pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/ProtoMIL/data/CAMELYON_patches',train= True, img_size = 224, seed = args.seed, validation_size = args.validation_size)
        return get_camelyon(True, 
        './data/CAMELYON/dataset/train','./data/CAMELYON/dataset/train','./data/CAMELYON/dataset/test', img_size = 224, seed = args.seed, validation_size = args.validation_size)
        
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')

def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, trainset_pretraining, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(args)
    
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = args.num_workers
    
    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor([(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain 
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
                                        
    else:        
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    
    projectloader = torch.utils.data.DataLoader(projectset,
                                              batch_size = 1,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=False
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True, 
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, trainloader_pretraining, projectloader, testloader, test_projectloader, classes

def create_datasets(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None):
    
    trainvalset = torchvision.datasets.ImageFolder(train_dir)
    classes = trainvalset.classes
    targets = trainvalset.targets
    indices = list(range(len(trainvalset)))

    train_indices = indices
    
    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError("There is no test set directory, so validation size should be > 0 such that training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices,test_size=validation_size,stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=test_indices)
        print("Samples in trainset:", len(indices), "of which",len(train_indices),"for training and ", len(test_indices),"for testing.", flush=True)
    else:
        testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    
    trainset = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
    trainset_normal = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transforms.Compose([transform1, transform2])), indices=train_indices)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)

    if test_dir_projection is not None:
        testset_projection = torchvision.datasets.ImageFolder(test_dir_projection, transform=transform_no_augment)
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
        targets_pr = trainvalset_pr.targets
        indices_pr = list(range(len(trainvalset_pr)))
        train_indices_pr = indices_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr,test_size=validation_size,stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p, transform2=transform2), indices=train_indices_pr)
    else:
        trainset_pretraining = None
    
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)

def create_datasets_MIL(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None):
    
    # trainvalset = CamelyonPreprocessedBagsCross(path="/pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/ProtoMIL/data/CAMELYON_patches/", train=True, shuffle_bag=True, data_augmentation=True, random_state=seed)
    # classes = trainvalset.classes
    # targets = [label for k, label in trainvalset.labels.items() if os.path.exists(os.path.join(trainvalset.path, k, trainvalset.embed_name))]
    # indices = list(range(len(trainvalset)))

    # train_indices = indices
    
    # trainset = CamelyonPreprocessedBagsCross(path="/pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/ProtoMIL/data/CAMELYON_patches/", train=True, shuffle_bag=True,
    #                                        data_augmentation=True,
    #                                        random_state=seed)
    
    # trainset_pretraining = None
    # trainset_normal = trainset
    # trainset_normal_augment = trainset

    # projectset = CamelyonPreprocessedBagsCross(path="/pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/ProtoMIL/data/CAMELYON_patches/", train=True, push=True, shuffle_bag=True,
    #                                             random_state=seed)
    
    # testset = CamelyonPreprocessedBagsCross(path="/pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/ProtoMIL/data/CAMELYON_patches/", train=False, test=True, all_labels=True,
    #                                             random_state=seed)
    # testset_projection = CamelyonPreprocessedBagsCross(path="/pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/ProtoMIL/data/CAMELYON_patches/", train=False, test=True,
    #                                                  all_labels=True, push=True)
    
    
    trainvalset = MultipleInstanceDataset(train_dir, random_state=seed)
    classes = trainvalset.classes
    targets = [label for _, label in trainvalset.bags]
    indices = list(range(len(trainvalset)))

    train_indices = indices
    
    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError("There is no test set directory, so validation size should be > 0 such that the training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices, test_size=validation_size, stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(MultipleInstanceDataset(train_dir, transform=transform_no_augment), indices=test_indices)
        print("Samples in trainset:", len(indices), "of which", len(train_indices), "for training and ", len(test_indices), "for testing.", flush=True)
    else:
        testset = MultipleInstanceDataset(test_dir, transform=transform_no_augment)
    
    trainset = torch.utils.data.Subset(TwoAugSupervisedDataset_MIL(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
    projectset = MultipleInstanceDataset(project_dir, transform=transform_no_augment)
    # projectset = MILPretrainDataset(project_dir, random_state=seed, transform=transform_no_augment)


    if test_dir_projection is not None:
        testset_projection = MultipleInstanceDataset(test_dir_projection, transform=transform_no_augment)
    else:
        testset_projection = testset

    if train_dir_pretrain is not None:
        trainvalset_pr = MultipleInstanceDataset(train_dir_pretrain, transform=transform_no_augment)
        targets_pr = trainvalset_pr.targets
        indices_pr = trainvalset_pr.indices
        train_indices_pr = indices_pr

        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=validation_size, stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(MultipleInstanceDataset(train_dir_pretrain, transform=transforms.Compose([transform1p, transform2])), indices=train_indices_pr)
    else:
        # trainset_pretraining = None
        trainvalset_pr = MILPretrainDataset(train_dir, random_state=seed)
        targets_pr = [label for _, label in trainvalset_pr.imgs]
        indices_pr = list(range(len(trainvalset_pr)))
        train_indices_pr = indices_pr

        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=validation_size, stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1, transform2=transform2), indices=train_indices_pr)

    
    return trainset, trainset_pretraining, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)

# def create_datasets_CAMELYON(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None):
    
#     trainvalset = CamelyonPreprocessedBags(train_dir, train=True)
#     classes = trainvalset.classes
#     targets = [label for _, label in trainvalset.bags]
#     indices = list(range(len(trainvalset)))

#     train_indices = indices
    
#     if test_dir is None:
#         if validation_size <= 0.:
#             raise ValueError("There is no test set directory, so validation size should be > 0 such that the training set can be split.")
#         subset_targets = list(np.array(targets)[train_indices])
#         train_indices, test_indices = train_test_split(train_indices, test_size=validation_size, stratify=subset_targets, random_state=seed)
#         testset = torch.utils.data.Subset(InstanceStack_CAMELYON(CamelyonPreprocessedBags(train_dir, train=True), transform=transform_no_augment), indices=test_indices)
#         print("Samples in trainset:", len(indices), "of which", len(train_indices), "for training and ", len(test_indices), "for testing.", flush=True)
#     else:
#         testset = InstanceStack_CAMELYON(CamelyonPreprocessedBags(test_dir, train=False), transform=transform_no_augment)
    
#     trainset = torch.utils.data.Subset(TwoAugSupervisedDataset_CAMELYON(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
#     trainset_normal = torch.utils.data.Subset(InstanceStack_CAMELYON(CamelyonPreprocessedBags(train_dir, train=True), transform=transform_no_augment), indices=train_indices)
#     trainset_normal_augment = torch.utils.data.Subset(InstanceStack_CAMELYON(CamelyonPreprocessedBags(train_dir, train=True), transform=transforms.Compose([transform1, transform2])), indices=train_indices)
#     projectset = InstanceStack_CAMELYON(CamelyonPreprocessedBags(project_dir), transform=transform_no_augment)

#     if test_dir_projection is not None:
#         testset_projection = InstanceStack_CAMELYON(CamelyonPreprocessedBags(test_dir_projection, train=False), transform=transform_no_augment)
#     else:
#         testset_projection = testset

#     if train_dir_pretrain is not None:
#         trainvalset_pr = InstanceStack_CAMELYON(CamelyonPreprocessedBags(train_dir_pretrain, train=True))
#         targets_pr = trainvalset_pr.targets
#         indices_pr = trainvalset_pr.indices
#         train_indices_pr = indices_pr

#         if test_dir is None:
#             subset_targets_pr = list(np.array(targets_pr)[indices_pr])
#             train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=validation_size, stratify=subset_targets_pr, random_state=seed)

#         trainset_pretraining = torch.utils.data.Subset(InstanceStack_CAMELYON(CamelyonPreprocessedBags(train_dir_pretrain, train=True), transform=transforms.Compose([transform1p, transform2])), indices=train_indices_pr)
#     else:
#         trainset_pretraining = None
    
#     return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)

def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain, test_dir_projection, transform1p)

def get_camelyon(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    transform_no_augment = transforms.Compose([transforms.Resize(size=(img_size+8, img_size+8)),
                                               HistoNormalize(),
                                               transforms.ToTensor()
                                               ])
    
    if augment:
        transform1 = transforms.Compose([RandomRotate(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         ])
        
        transform2 = transforms.Compose([RandomHEStain(),
                                         HistoNormalize(),
                                         transforms.RandomCrop(size=(img_size, img_size), padding=(3, 3),padding_mode='reflect'),
                                         transforms.ToTensor(),
                                         ])

    else:
        transform1 = transform_no_augment 
        transform2 = transform_no_augment           

    return create_datasets_MIL(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

class InstanceStack(torch.utils.data.Dataset):
    r"""Returns stacked instances within a bag and a label."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.bags = dataset.bags  # Assumes that bags and their labels are stored in the dataset
        self.targets = [label for _, label in self.bags]  # Assuming bags are (path, label) tuples
        self.imgs = dataset.imgs
        self.transform = transform
        self.samples = dataset.samples


    def __getitem__(self, index):
        bag_path, target = self.bags[index]

        # Load all instances in the bag
        instances = [Image.open(os.path.join(bag_path, instance_name)) for instance_name in os.listdir(bag_path)]

        # Apply transformations if needed
        if self.transform:
            instances = [self.transform(instance) for instance in instances]
        
        instances = torch.stack(instances)
        
        return instances, target

    def __len__(self):
        return len(self.bags)

class TwoAugSupervisedDataset_MIL(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        self.bags = dataset.bags  # Assumes that bags and their labels are stored in the dataset
        self.targets = [label for _, label in self.bags]  # Assuming bags are (path, label) tuples
        self.imgs = dataset.imgs
        self.transform1 = transform1
        self.transform2 = transform2


    def __getitem__(self, index):
        bag_path, target = self.bags[index]

        # Load all instances in the bag
        instances = [Image.open(os.path.join(bag_path, instance_name)) for instance_name in os.listdir(bag_path) if (os.path.join(bag_path, instance_name), target) in self.imgs]

        # Apply transformations if needed
        instances = [self.transform1(instance) for instance in instances]
        instances1 = [self.transform2(instance) for instance in instances]
        instances2 = [self.transform2(instance) for instance in instances]
        instances1 = torch.stack(instances1)
        instances2 = torch.stack(instances2)
        return instances1, instances2, target

    def __len__(self):
        return len(self.bags)

class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        self.imgs = dataset.imgs
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = [label for _, label in self.imgs] 
        self.transform1 = transform1
        self.transform2 = transform2
        

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)

# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

