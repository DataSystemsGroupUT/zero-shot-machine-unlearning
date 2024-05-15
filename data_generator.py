import torch
import random
import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as func
from evaluation import *
from copy import deepcopy

from logger import Logger

from models import Generator
from utils import *
from config import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataGenerator:
    def __init__(self,
                 model,
                 dataset_name,
                 nc = 3,
                 samples_count = 300,
                 number_of_classes = 10):
        
        self.logger = Logger(tag = "Data Generator", enabled=True)

        self.model = model
        self.samples_count = samples_count
        self.number_of_classes = number_of_classes
        self.dataset_name = dataset_name

        self.forget_dataset = {}
        
        for class_id in generator_path[self.dataset_name]:
            generator = self.__get_generator(class_id, nc)
            self.forget_dataset[ class_id  ] = self.generate_image_from_class( generator,  class_id , self.samples_count )  

        self.logger.log ("finished generating images ")
    
    
    def __get_generator(self, class_id, nc):
        generator = Generator(1, nc).to(device)
        generator.apply(self.__weights_init)
        generator.load_state_dict(torch.load(generator_path[self.dataset_name][class_id]))
        return generator
    
    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def get_per_class_dataset(self):
        return self.forget_dataset
            
    def generate_image_from_class(self, generator, target_class, target_collection_size):
        self.logger.log (f"generating images for {target_class}")

        self.model.eval()
        
        with torch.no_grad():
            generated = 0
            images_collection = []
            while generated < target_collection_size:
                fixed_noise = torch.randn(128, 100, 1, 1, device=device)
                fake = generator(fixed_noise)

                denormalize = transforms.Compose([transforms.Resize(32, antialias=None),])
                
                images = denormalize(fake)
                outputs, *_ = self.model(images)
                posteriors = nn.Softmax(dim=1)(outputs)

                for index, posterior in enumerate (posteriors.cpu().detach().numpy()):
                    most_probable_class = np.argmax( posterior )

                    if target_class ==  most_probable_class or ( self.dataset_name == 'cifar20' and cifar100_to_cifar20(target_class) == most_probable_class  ) :
                        images_collection.append( images[index]  )
                        generated += 1

            return images_collection
        
    def __initialize_list(self, value, size):
        return torch.Tensor( [ value ] *  size ).long().cuda()
    
    def get_synthetic_dr_and_df(self, target_forget_class):
        
        df_X =  self.forget_dataset[target_forget_class]
        df_y =  self.__initialize_list( target_forget_class, len( self.forget_dataset[ target_forget_class ])  )
        
        Df_formated = []
        for indx, sample in  enumerate(df_X):
            pt =  [ Variable(sample), df_y[indx]  ]
            Df_formated.append( pt  )
            
        remember_dataset = []
        for class_id in  generator_path[self.dataset_name]:
            if class_id != target_forget_class :
                remember_dataset.extend(    zip(self.forget_dataset[ class_id ],
                                                self.__initialize_list( class_id, len( self.forget_dataset[ class_id ])  )
                                            )
                                        )
        
        random.shuffle(remember_dataset)
        Dr_formated = []
        for indx, sample in  enumerate( remember_dataset ):
            pt = ( Variable(sample[0] ), sample[1]  )
            Dr_formated.append( pt  )

        return Dr_formated, Df_formated
    
