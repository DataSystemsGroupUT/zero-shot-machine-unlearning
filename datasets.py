
import random
import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from logger import Logger
from utils import cifar100_to_cifar20

ds_normalization = transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])


class Dataset:
    def __init__(self, dataset_name: str, dataset_path: str, enable_logs = True):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.logger = Logger(tag = "dataset", enabled = enable_logs)
        self.num_classes = 10
        
        if ( self.dataset_name == "MNIST" ):
            self.num_classes = 10
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            self.dataset = dset.MNIST( dataset_path, train=True, transform=transform, download=True)
            self.test_dataset = dset.MNIST( dataset_path, train=False, transform=transform, download=True)
        
        if ( self.dataset_name == "cifar10" ):
            self.dataset = dset.CIFAR10(root=dataset_path, download=True,
                                    transform=ds_normalization)
            self.test_dataset = dset.CIFAR10(root=dataset_path, download=True, train = False,
                                    transform=ds_normalization)
            

        elif ( self.dataset_name == "cifar100" ):
            self.num_classes = 100
            self.dataset = dset.CIFAR100(root=dataset_path, download=True, train = True, 
                                    transform=ds_normalization)
            self.test_dataset = dset.CIFAR100(root=dataset_path, download=True, train = False, 
                                    transform=ds_normalization)
        elif (self.dataset_name == "SVHN"):
            self.dataset = dset.SVHN(root=dataset_path, download=True,  split = "train",
                           transform=ds_normalization)
            self.test_dataset = dset.SVHN(root=dataset_path, download=True,  split = "test",
                           transform=ds_normalization)
        # elif (self.dataset_name == "MNIST"):
        #     self.num_classes = 10
        #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #     self.dataset = dset.MNIST(root=dataset_path, train=True, download=True, transform=transform)
        #     self.test_dataset = dset.MNIST(root=dataset_path, train=False, download=True, transform=transform)
        elif (self.dataset_name == "cifar20"):
            self.num_classes = 20
            self.dataset = dset.CIFAR100(root=dataset_path, download=True, train = True, 
                                    transform=ds_normalization)
            self.test_dataset = dset.CIFAR100(root=dataset_path, download=True, train = False, 
                                    transform=ds_normalization)
            
            new_targets = self.dataset.targets
            for idx, target in enumerate(self.dataset.targets):
                new_targets[idx] = cifar100_to_cifar20(target)
            
            self.dataset.targets = new_targets
            
            new_test_targets = self.test_dataset.targets
            for idx, target in enumerate(self.test_dataset.targets):
                new_test_targets[idx] = cifar100_to_cifar20(target)
                
            self.test_dataset.targets = new_test_targets
            self.dataset.classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 'household electrical devices', 'househould furniture', 'insects', 'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

        else:
            raise Exception("dataset not supported!")
        
        random_seed = 42
        torch.manual_seed(random_seed)

        val_size = len(self.dataset) // 20
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.validate_dataset = random_split(self.dataset, [train_size, val_size])
        self.logger.log( f"dataset {self.dataset_name} initilazed!"  )
        
        self.dataset_split_by_class = {}
        for datapoint in self.train_dataset:
            if ( datapoint[1] in self.dataset_split_by_class ):
                self.dataset_split_by_class[ datapoint[1] ].append( [datapoint[0], datapoint[1]  ] )
            else:
                self.dataset_split_by_class[ datapoint[1] ] = [ [datapoint[0], datapoint[1]  ] ]

        self.logger.log( f"splitted the dataset by class successfully!"  )
        
        
        self.validated_dataset_split_by_class = {}
        for datapoint in self.validate_dataset:
            if ( datapoint[1] in self.validated_dataset_split_by_class ):
                self.validated_dataset_split_by_class[ datapoint[1] ].append( [datapoint[0], datapoint[1]  ] )
            else:
                self.validated_dataset_split_by_class[ datapoint[1] ] = [ [datapoint[0], datapoint[1]  ] ]
                
        self.logger.log( f"splitted the validate dataset by class successfully!"  )
            
    def get_train_dataset(self):
        self.logger.log( f"size of train_dataset = {len(self.train_dataset)}"  )
        return self.train_dataset
    
    def get_validate_dataset(self):
        self.logger.log( f"size of validate_dataset = {len(self.validate_dataset)}"  )
        return self.validate_dataset
    
    def get_test_dataset(self):
        self.logger.log( f"size of test_dataset = {len(self.test_dataset)}"  )
        return self.test_dataset
    
    def split_train_dataset_to_dr_and_df(self, Df_class_ids):
        Dr = []
        Df = []
        for class_id in range(self.num_classes):
            if class_id in Df_class_ids:
                Df.extend( self.dataset_split_by_class[class_id] )
            else:
                Dr.extend( self.dataset_split_by_class[class_id] )
        
        self.logger.log( f"size of Df = {len(Df)} & size of Dr = { len(Dr)}"  )
        return Dr, Df
    
    def split_validate_dataset_to_dr_and_df(self, Df_class_ids):
        
        Dr = []
        Df = []
        for class_id in range(self.num_classes):
            if class_id in Df_class_ids:
                Df.extend( self.validated_dataset_split_by_class[class_id] )
            else:
                Dr.extend( self.validated_dataset_split_by_class[class_id] )
        
        self.logger.log( f"Size of Validate Df = {len(Df)} & Size of Validate Dr = { len(Dr)}"  )
        return Dr, Df
        
    
    
