# MuGAN and zMuGAN
## Machine unlearning techniques for zero-shot setups using GANs


The main script can be found in `execute_unlearning_algorithms.py`.  

It takes as an input the following parameters:
  - model
  - weight_path
  - dataset
  - dataset_path is already downloaded
  - classes --> number of classes in this dataset
  - target_class --> target forget class id
  - gan_output --> use gan output vs original dataset
  - gan_dataset_size
  - learning_rate
  - lipschitz_std  --> (only neeeded for JiT)
  - calc_ain ---> calculate Ain Score
  - method --> the intended unlearning technique:
    - the implemented options are:  ('retrain','finetune', 'SCRUM' ,'UNSIR', 'negative_gradiant' , 'lipschitz', 'randomize_label', 'original', 'emmn', 'experimental_method')
    
