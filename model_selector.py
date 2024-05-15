from models import *
from logger import Logger


class ModelSelector:
    def __init__(self, model_name, num_channels, num_classes, model_weights_path,  initalize_wights = True):
        self.model_name = model_name
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model_weights_path = model_weights_path
        
        self.logger = Logger(tag = "Model Selector", enabled=True)
        
        if model_name == "ALLCNN" :
            self.model = AllCNN( num_classes=num_classes, n_channels = num_channels)
        elif model_name == "VGG16" :
            self.model = VGG16(num_classes= num_classes, num_channels=num_channels, return_activations=True)
        elif model_name == "RESNET9" :
            raise Exception("Not implemented yet!")
        elif model_name == "LENET" :
            self.logger.log( "not adjusted for return activation" )
            self.model = LeNet( num_classes= num_classes, num_channels = num_channels )
        else:
            raise Exception("Model Not implmented. That's it!")
        
        if  initalize_wights :
            self.logger.log( f"Model selector ({self.model_name}) Loading weights .." )
            self.model.load_state_dict(torch.load(model_weights_path))
        
        self.logger.log( f"Model selector ({self.model_name}) initilazed!" )
        
    def get_model( self ):
        return self.model
    
    def get_retrained_model(self, path):
        
        if self.model_name == "ALLCNN" :
            retrained_model = AllCNN( num_classes=self.num_classes, n_channels = self.num_channels)
        elif self.model_name == "VGG16" :
            retrained_model = VGG16(num_classes= self.num_classes, num_channels=self.num_channels, return_activations=True)
        elif self.model_name == "RESNET9" :
            raise Exception("Not implemented yet!")
        elif self.model_name == "LENET" :
            self.logger.log( "not adjusted for return activation" )
            retrained_model = LeNet( num_classes= self.num_classes, num_channels = self.num_channels )
        else:
            raise Exception("Model Not implmented. That's it!")
        
       
        self.logger.log( f"Model selector ({self.model_name}) Loading weights .." )
        retrained_model.load_state_dict(torch.load(path))
        
        return retrained_model