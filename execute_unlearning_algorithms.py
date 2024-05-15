import torch

from datasets import Dataset
from model_selector import ModelSelector
from unlearning_techniques import UnlearningTechnique
from utils import *
import pandas as pd
from evaluation import *
from datetime import datetime
import time
from data_generator import DataGenerator

from logger import Logger

import optuna

from copy import deepcopy
import argparse

logger = Logger(tag = "Experiment Execution", enabled=True)

parser = argparse.ArgumentParser()

def get_report_row (technique_name, learning_rate, df_acc, dr_acc, mia_socre, ain_score,  exec_time  ):

    report_raw = {
        "dataset" : args.dataset,
        "synthetic_data" : args.gan_output,
        "model"  : args.model,
        "technique" :  technique_name,
        "class_id"  : args.target_class,
        "df"  : df_acc,
        "dr"  : dr_acc,
        "mia" : mia_socre,
        "learning_rate" : args.learning_rate,
        "exec_time" : exec_time,
        "dataset_size" : args.gan_dataset_size if args.gan_output == "true" else "NA",
        "ain_score" : ain_score,
        }
    
    return report_raw

parser.add_argument('-model', type=str, required=True, help='model type')
parser.add_argument('-weight_path', type=str, required=True, help='Path to model weights. If you need to train a new model use pretrain_model.py')
parser.add_argument('-dataset', type=str, required=True, nargs='?',
                    choices=['cifar10', 'cifar100', 'SVHN'],
                    help='dataset to train on')
parser.add_argument('-dataset_path', type=str, required=True,help='dataset path')
parser.add_argument('-classes', type=int, required=True,help='number of classes')
parser.add_argument('-method', type=str, required=True, nargs='?',
                    choices=['retrain','finetune', 'SCRUM' ,'UNSIR', 'negative_gradiant' , 'lipschitz', 'randomize_label', 'original', 'emmn', 'experimental_method'],
                    help='select unlearning method from choice set')    
parser.add_argument('-target_class', type=int, required=True,help='number of classes')
parser.add_argument('-gan_output', type=str, required=True, choices=['true', 'false'], help='using gan or real data')
parser.add_argument('-gan_dataset_size', type=int, required=True,help='size of generated dataset')
parser.add_argument('-learning_rate', type=float, required=True,help='learning rate')
parser.add_argument('-lipschitz_std', type=float, required=False,help='lipschitz std')
parser.add_argument('-calc_ain', type=bool, default=False, required=False,help='bool for calc_ain')

args = parser.parse_args()


logger.log( args )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Dataset(dataset_name =  args.dataset , dataset_path = args.dataset_path )

train_ds = dataset.get_train_dataset()
val_ds = dataset.get_validate_dataset()
test_ds = dataset.get_test_dataset()
test_dataset_dl = torch.utils.data.DataLoader(test_ds, 128, shuffle=True, num_workers=4, pin_memory=True)
test_dataset_dl = DeviceDataLoader(test_dataset_dl, device)     

ms = ModelSelector(model_name=args.model, 
                    num_channels=3, 
                    num_classes=args.classes, 
                    initalize_wights = True,
                    model_weights_path= args.weight_path )
original_model = ms.get_model()
to_device(original_model, device)

report = []

class_id = args.target_class

logger.log( f"forgetting class_{class_id}" )

Dr_real , Df_real = dataset.split_train_dataset_to_dr_and_df(Df_class_ids= [class_id] )

if ( args.gan_output  == "true" ):
    logger.log( f"using synthetic data" )
    dg = DataGenerator  (model=original_model, 
                            dataset_name= args.dataset,
                            number_of_classes=args.classes,
                            samples_count= args.gan_dataset_size ) 
    Dr , Df = dg.get_synthetic_dr_and_df( target_forget_class=class_id )
else:
    logger.log( f"using real data" )
    Dr , Df = Dr_real , Df_real
       
Df_dl = torch.utils.data.DataLoader(Df_real, 128, shuffle=True )
Df_dl = DeviceDataLoader(Df_dl, device)  
Dr_dl = torch.utils.data.DataLoader(Dr_real, 128, shuffle=True)
Dr_dl = DeviceDataLoader(Dr_dl, device)  


start  = time.time()
learning_rate = args.learning_rate

ms = ModelSelector(model_name=args.model, 
                num_channels=3, 
                num_classes=args.classes, 
                initalize_wights = True if not args.method == 'retrain' else False , 
                model_weights_path= args.weight_path )
model = ms.get_model()
to_device(model, device)
model.train()

unlearn = UnlearningTechnique( enable_logs = True)

logger.log(" unlearning started!"  )

if ( args.method == 'finetune' ) :

    model_ = unlearn.finetuning( target_forget_class=class_id,
                        Dr=Dr,
                        model=model,
                        number_of_epochs=5,
                        opt_func = torch.optim.Adam,
                        learning_rate=learning_rate,
                    )

elif (  args.method == 'original' ): 
    model_ = model
elif ( args.method == 'emmn' ):
    model_ =  unlearn.emmn(model=model,
                        num_classes = args.classes,
                        forget_class=class_id,
                        Dr= Dr,
                        learning_rate=learning_rate )
    
elif (  args.method == 'retrain' ): 

    model_ =  unlearn.retraining(
                target_forget_class = class_id,
                model_name= args.model,
                dataset_name = args.dataset,
                model=model,
                Dr=Dr,
                opt_func = torch.optim.Adam,
                learning_rate=0.001,
                number_of_epochs=15,
                save_model=True,
                plot_exp=False
            )
    
elif (  args.method == 'SCRUM' ): 

    model_ = unlearn.SCRUM(
                model=model,
                Df=Df,
                Dr=Dr,
                learning_rate=learning_rate
            )
    
elif (  args.method == 'lipschitz' ): 

    model_ = unlearn.lipschitz(Df=Df,
                                model=model,
                                opt_func=torch.optim.Adam,
                                learning_rate=learning_rate, 
                                noise_std=args.lipschitz_std)

elif (  args.method == 'UNSIR' ): 

    model_ = unlearn.UNSIR( model=model,
                            Dr = Dr,
                            Df=Df,
                            opt_func=torch.optim.Adam,
                            learning_rate=learning_rate)

elif (  args.method == 'negative_gradiant' ): 
    
    model_ = unlearn.negative_gradiant( model=model,
                            Dr = Dr,
                            Df= Df,
                            opt_func=torch.optim.Adam,
                            learning_rate=learning_rate,
                            num_epochs=2)
    
elif (  args.method == 'randomize_label' ): 

    model_ = unlearn.randomize_labels(      model=model,
                                            Dr = Dr,
                                            Df=Df,
                                            number_of_classes=args.classes,
                                            learning_rate=learning_rate, 
                                            opt_func=torch.optim.Adam,    
                                            target_class=class_id)
    
    
elif (  args.method == 'experimental_method' ): 

    model_ = unlearn.experimental_method(      model=model,
                                                Dr = Dr,
                                                Df=Df,
                                                number_of_classes=args.classes,
                                                learning_rate=learning_rate, 
                                                opt_func=torch.optim.Adam,    
                                                target_class=class_id)


logger.log(" unlearning ended!"  )
end = time.time()

post_df, post_dr = calculate_class_accuracy(model=model_, 
                                            dataloader= test_dataset_dl, 
                                            num_classes=args.classes,  
                                            target_class =class_id, 
                                            print_details= True)

# mia_score = get_membership_attack_prob(Dr_dl, Df_dl, test_dataset_dl, model_)
mia_score = 1

print (args.calc_ain)

if (args.calc_ain):

    val_dr, val_df =  dataset.split_validate_dataset_to_dr_and_df([class_id])

    ms = ModelSelector(model_name=args.model, 
                        num_channels=3, 
                        num_classes=args.classes, 
                        initalize_wights = True,
                        model_weights_path= args.weight_path )
    original_model = ms.get_model()
    to_device(original_model, device)


    retrained_model = ms.get_retrained_model(f"./weights/{args.model}_{args.dataset}_epochs=15_lr=0.001_without class_{class_id}.pth")
    to_device(retrained_model, device)

    forget_model = deepcopy(model_)

    # ain(full_model, model, gold_model,
    #         train_data,
    #         val_retain, val_forget, 
    #         batch_size = 256,
    #         error_range = 0.05,
    #         lr = 0.001)

    ain_score = ain(original_model, forget_model, retrained_model, train_ds, val_dr, val_df )
else:
    ain_score = 1
report_raw = get_report_row (technique_name = args.method, 
                             learning_rate= learning_rate, 
                             df_acc=post_df, 
                             dr_acc=post_dr, 
                             mia_socre =mia_score, 
                             ain_score = ain_score, 
                             exec_time=end-start  )

report.append(  report_raw  )
logger.log ( report_raw )

date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
pd.DataFrame.from_records( report ).to_csv( f"{date_time}_report_{args.dataset}_{args.model}_{args.method}.csv" ) 



