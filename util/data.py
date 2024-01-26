import numpy as np
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split
from util.utils_augemntation import HistoNormalize
from util.utils_augemntation import RandomHEStain
from util.utils_augemntation import RandomRotate

class MultipleInstanceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.bags = []  # List to store bag information
        self.imgs = []  # List of (image path, class_index) tuples 
        self.samples = []  # List of (image path, class_index) tuples for all instances

        for class_label, class_name in enumerate(self.classes):
            class_path = os.path.join(root, class_name)
            for bag_name in os.listdir(class_path):
                bag_path = os.path.join(class_path, bag_name)
                self.bags.append((bag_path, class_label))  # Store bag information
                for instance_name in os.listdir(bag_path):
                    instance_path = os.path.join(bag_path, instance_name)
                    self.imgs.append((instance_path, class_label)) 
                    self.samples.append((instance_path, class_label)) 
        
    def __len__(self):
        return len(self.bags)
       
    def __getitem__(self, idx):
        bag_path, class_label = self.bags[idx]

        # Load all instances in the bag
        instances = [Image.open(os.path.join(bag_path, instance_name)) for instance_name in os.listdir(bag_path)]

        # Apply transformations if needed
        if self.transform:
            instances = [self.transform(instance) for instance in instances]

        # Determine the label based on the class label (negative or positive)
        bag_label = class_label  # Assume bag label is the same as class label

        return instances, bag_label
    
    def get_instances_list(self):
        return self.imgs

def create_datasets_MIL(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None):
    
    trainvalset = MultipleInstanceDataset(train_dir)
    classes = trainvalset.classes
    targets = [label for _, label in trainvalset.bags]
    indices = list(range(len(trainvalset)))

    train_indices = indices
    
    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError("There is no test set directory, so validation size should be > 0 such that the training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices, test_size=validation_size, stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(InstanceStack(MultipleInstanceDataset(train_dir), transform=transform_no_augment), indices=test_indices)
        print("Samples in trainset:", len(indices), "of which", len(train_indices), "for training and ", len(test_indices), "for testing.", flush=True)
    else:
        testset = InstanceStack(MultipleInstanceDataset(test_dir), transform=transform_no_augment)
    
    trainset = torch.utils.data.Subset(TwoAugSupervisedDataset_MIL(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
    trainset_normal = torch.utils.data.Subset(InstanceStack(MultipleInstanceDataset(train_dir), transform=transform_no_augment), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(InstanceStack(MultipleInstanceDataset(train_dir), transform=transforms.Compose([transform1, transform2])), indices=train_indices)
    projectset = InstanceStack(MultipleInstanceDataset(project_dir), transform=transform_no_augment)

    if test_dir_projection is not None:
        testset_projection = InstanceStack(MultipleInstanceDataset(test_dir_projection), transform=transform_no_augment)
    else:
        testset_projection = testset

    if train_dir_pretrain is not None:
        trainvalset_pr = InstanceStack(MultipleInstanceDataset(train_dir_pretrain))
        targets_pr = trainvalset_pr.targets
        indices_pr = trainvalset_pr.indices
        train_indices_pr = indices_pr

        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=validation_size, stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(InstanceStack(MultipleInstanceDataset(train_dir_pretrain), transform=transforms.Compose([transform1p, transform2])), indices=train_indices_pr)
    else:
        trainset_pretraining = None
    
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)

def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset =='CUB-200-2011':   
        return get_birds(True, './data/CUB_200_2011/dataset/train_crop', './data/CUB_200_2011/dataset/train', './data/CUB_200_2011/dataset/test_crop', args.image_size, args.seed, args.validation_size, './data/CUB_200_2011/dataset/train', './data/CUB_200_2011/dataset/test_full')
    
    # if args.dataset == 'MNIST':
    #     return get_digits(True, 
    #     './data/MNIST/dataset/train','./data/MNIST/dataset/train','./data/MNIST/dataset/test', args.image_size, args.seed, args.validation_size)
    
    # if args.dataset == 'pets':
    #     return get_pets(True, './data/PETS/dataset/train','./data/PETS/dataset/train','./data/PETS/dataset/test', args.image_size, args.seed, args.validation_size)
    # if args.dataset == 'partimagenet': #use --validation_size of 0.2
    #     return get_partimagenet(True, './data/partimagenet/dataset/all', './data/partimagenet/dataset/all', None, args.image_size, args.seed, args.validation_size) 
    # if args.dataset == 'CARS':
    #     return get_cars(True, './data/cars/dataset/train', './data/cars/dataset/train', './data/cars/dataset/test', args.image_size, args.seed, args.validation_size)
    # if args.dataset == 'grayscale_example':
    #     return get_grayscale(True, './data/train', './data/train', './data/test', args.image_size, args.seed, args.validation_size)
    
    if args.dataset == 'Bisque':
        return get_bisque(True, 
        './data/Bisque/dataset/train','./data/Bisque/dataset/train','./data/Bisque/dataset/test', img_size = 32, seed = args.seed, validation_size = args.validation_size)
    
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')

def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(args)
    
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

    trainloader_normal = torch.utils.data.DataLoader(trainset_normal,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    trainloader_normal_augment = torch.utils.data.DataLoader(trainset_normal_augment,
                                            batch_size=args.batch_size,
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
    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes

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


# def get_pets(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform_no_augment = transforms.Compose([
#                             transforms.Resize(size=(img_size, img_size)),
#                             transforms.ToTensor(),
#                             normalize
#                         ])
    
#     if augment:
#         transform1 = transforms.Compose([
#             transforms.Resize(size=(img_size+48, img_size+48)), 
#             TrivialAugmentWideNoColor(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
#         ])
        
#         transform2 = transforms.Compose([
#         TrivialAugmentWideNoShape(),
#         transforms.RandomCrop(size=(img_size, img_size)), #includes crop
#         transforms.ToTensor(),
#         normalize
#         ])
#     else:
#         transform1 = transform_no_augment    
#         transform2 = transform_no_augment           

#     return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

# def get_digits(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform_no_augment = transforms.Compose([
#                             transforms.Resize(size=(img_size, img_size)),
#                             transforms.ToTensor(),
#                             normalize
#                         ])
    
#     if augment:
#         transform1 = transforms.Compose([
#             transforms.Resize(size=(img_size+48, img_size+48)), 
#             TrivialAugmentWideNoColor(),
#             #transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
#         ])
        
#         transform2 = transforms.Compose([
#         TrivialAugmentWideNoShape(),
#         transforms.RandomCrop(size=(img_size, img_size)), #includes crop
#         transforms.ToTensor(),
#         normalize
#         ])
#     else:
#         transform1 = transform_no_augment    
#         transform2 = transform_no_augment           

#     return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)


# def get_partimagenet(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
#     # Validation size was set to 0.2, such that 80% of the data is used for training
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform_no_augment = transforms.Compose([
#                             transforms.Resize(size=(img_size, img_size)),
#                             transforms.ToTensor(),
#                             normalize
#                         ])

#     if augment:
#         transform1 = transforms.Compose([
#             transforms.Resize(size=(img_size+48, img_size+48)), 
#             TrivialAugmentWideNoColor(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
#         ])
#         transform2 = transforms.Compose([
#                             TrivialAugmentWideNoShape(),
#                             transforms.RandomCrop(size=(img_size, img_size)), #includes crop
#                             transforms.ToTensor(),
#                             normalize
#                             ])
#     else:
#         transform1 = transform_no_augment    
#         transform2 = transform_no_augment           

#     return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

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

# def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
#     shape = (3, img_size, img_size)
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)

#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform_no_augment = transforms.Compose([
#                             transforms.Resize(size=(img_size, img_size)),
#                             transforms.ToTensor(),
#                             normalize
#                         ])

#     if augment:
#         transform1 = transforms.Compose([
#             transforms.Resize(size=(img_size+32, img_size+32)), 
#             TrivialAugmentWideNoColor(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
#         ])
       
#         transform2 = transforms.Compose([
#                     TrivialAugmentWideNoShapeWithColor(),
#                     transforms.RandomCrop(size=(img_size, img_size)), #includes crop
#                     transforms.ToTensor(),
#                     normalize
#                     ])
                            
#     else:
#         transform1 = transform_no_augment    
#         transform2 = transform_no_augment           

#     return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

# def get_grayscale(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None): 
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform_no_augment = transforms.Compose([
#                             transforms.Resize(size=(img_size, img_size)),
#                             transforms.Grayscale(3), #convert to grayscale with three channels
#                             transforms.ToTensor(),
#                             normalize
#                         ])

#     if augment:
#         transform1 = transforms.Compose([
#             transforms.Resize(size=(img_size+32, img_size+32)), 
#             TrivialAugmentWideNoColor(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(224+8, scale=(0.95, 1.))
#         ])
#         transform2 = transforms.Compose([
#                             TrivialAugmentWideNoShape(),
#                             transforms.RandomCrop(size=(img_size, img_size)), #includes crop
#                             transforms.Grayscale(3),#convert to grayscale with three channels
#                             transforms.ToTensor(),
#                             normalize
#                             ])
#     else:
#         transform1 = transform_no_augment    
#         transform2 = transform_no_augment           

#     return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

def get_bisque(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #normalize = transforms.Compose([utils_augemntation.HistoNormalize(),transforms.ToTensor(),])
    transform_no_augment = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                               HistoNormalize(),
                                               transforms.ToTensor()
                                               ])
    
    # normalize = transforms.Normalize(mean=mean,std=std)
    # transform_no_augment = transforms.Compose([
    #                         transforms.Resize(size=(img_size, img_size)),
    #                         transforms.ToTensor(),
    #                         normalize
    #                     ])
    
    if augment:
        transform1 = transforms.Compose([RandomRotate(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         ])
        # transform2 = transforms.Compose([
        #                     TrivialAugmentWideNoShape(),
        #                     transforms.RandomCrop(32, padding=(3, 3),padding_mode='reflect'), #includes crop
        #                     transforms.ToTensor(),
        #                     normalize
        #                     ])
        transform2 = transforms.Compose([RandomHEStain(),
                                         HistoNormalize(),
                                         transforms.RandomCrop(32, padding=(3, 3),padding_mode='reflect'),
                                         transforms.ToTensor(),
                                         ])

    else:
        transform1 = transform_no_augment 
        transform2 = transform_no_augment           

    return create_datasets_MIL(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)
    #return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size)

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
        instances = [Image.open(os.path.join(bag_path, instance_name)) for instance_name in os.listdir(bag_path)]

        # Apply transformations if needed
        instances = [self.transform1(instance) for instance in instances]
        instances = [self.transform2(instance) for instance in instances]
        instances = torch.stack(instances)
        return instances, instances, target

    def __len__(self):
        return len(self.bags)

class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
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

