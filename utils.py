import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix


from torch.utils.data.dataloader import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plot_accuracies(history, label = ""):
    accuracies = [x['val_acc'] for x in history]
    fig, ax = plt.subplots()
    ax.plot(accuracies, '-x')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_title(f'{label} Accuracy vs. No. of epochs')
    fig.savefig(f'outputs/accuracies {label}.png')
    
    
def plot_losses(history, label = ""):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    fig, ax = plt.subplots()
    ax.plot(train_losses, '-bx')
    ax.plot(val_losses, '-rx')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(['Training', 'Validation'])
    ax.set_title(f' Loss vs. No. of epochs {label}')
    fig.savefig(f'outputs/loss  {label} .png')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def calculate_class_accuracy(model, dataloader, num_classes = 10, target_class = 0,  print_details= True):
    model.eval()
    correct = {}
    total = {}
    
    overall_total = 0
    running_corrects = 0
    
    for i, batch in enumerate(dataloader):
        imgs, labels = batch
        imgs, labels = imgs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs, *_ = model(imgs)    
            _, preds = torch.max(outputs, 1)
        
        correct_temp = 0
        for number in range(0, num_classes):
            total[number] = total[number] + labels[labels == number].size(0)  if number in total else labels[labels == number].size(0)
            correct_temp = (preds[labels == number] == labels[labels == number]).sum().item()
            correct[number] = correct[number] + correct_temp  if number in correct else correct_temp
        
        overall_total += labels.shape[0]
        running_corrects += torch.sum(preds == labels.data)
    
    accuracies_Df = [ correct[class_id]/total[class_id] for class_id in range(num_classes) if class_id == target_class ]
    accuracies_Dr = [ correct[class_id]/total[class_id] for class_id in range(num_classes) if class_id != target_class ]

    Df_acc_mean = np.mean( accuracies_Df )
    Dr_acc_mean = np.mean( accuracies_Dr )
    
    if (print_details):
        print ( [ correct[class_id]/total[class_id] for class_id in range(num_classes) ]  )
    
    model.train()
    return round(Df_acc_mean, 3), round(Dr_acc_mean, 3) 



@torch.no_grad()
def generate_model_report(model, dataloader, model_name, dataset_name,labels_name, num_classes = 10):
    model.eval()
    correct = {}
    total = {}
    
    overall_total = 0
    
    y_true = []
    y_pred = []
    
    for i, batch in enumerate(dataloader):
        imgs, labels = batch
        imgs, labels = imgs.cuda(), labels.cuda()
        
        y_true.extend( labels.cpu().numpy() )
        

        with torch.no_grad():
            outputs, *_ = model(imgs)    
            _, preds = torch.max(outputs, 1)
            y_pred.extend( preds.cpu().numpy()  )
        
        correct_temp = 0
        for number in range(0, num_classes):
            total[number] = total[number] + labels[labels == number].size(0)  if number in total else labels[labels == number].size(0)
            correct_temp = (preds[labels == number] == labels[labels == number]).sum().item()
            correct[number] = correct[number] + correct_temp  if number in correct else correct_temp
        
        overall_total += labels.shape[0]
        
    classes_accuracy = [ round(correct[class_id]/total[class_id], 4)   for class_id in range(num_classes) ]
    # print (correct)
    
    for i in range(num_classes):
        print (i, end = ", ")
    print ( "\n")
    for acc in classes_accuracy:
        print (acc, end = ", ")
    print ( "\n")
    
    # conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    
    sn.heatmap(cmn, annot=True, yticklabels=labels_name, xticklabels=labels_name, fmt='.2f')
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.savefig(f"conf_matrix_{model_name}_{dataset_name}.png") 


def relearn_time(model, train_loader, valid_loader, reqAcc, lr):
    # measuring relearn time for gold standard model
    rltime = 0
    curr_Acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    # we will try the relearning step till 4 epochs.
    for epoch in range(10):
        
        for batch in train_loader:
            model.train()
            loss = model.training_step(batch)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            history = [evaluate(model, valid_loader)]
            curr_Acc = history[0]["val_acc"]*100
            print(curr_Acc, sep=',')
            
            rltime += 1
            if(curr_Acc >= reqAcc):
                break
                
        if(curr_Acc >= reqAcc):
            break
    return rltime

def ain(full_model, model, gold_model,
        train_data,
        val_retain, val_forget, 
        batch_size = 256,
        error_range = 0.05,
        lr = 0.001):
    # measuring performance of fully trained model on forget class
    forget_valid_dl = DataLoader(val_forget, batch_size)
    forget_valid_dl = DeviceDataLoader(forget_valid_dl, device) 
    history = [evaluate(full_model, forget_valid_dl)]
    AccForget = history[0]["val_acc"]*100
    
    print("Accuracy of fully trained model on forget set is: {}".format(AccForget))
    
    retain_valid_dl = DataLoader(val_retain, batch_size)
    retain_valid_dl = DeviceDataLoader(retain_valid_dl, device) 
    history = [evaluate(full_model, retain_valid_dl)]
    AccRetain = history[0]["val_acc"]*100
    
    print("Accuracy of fully trained model on retain set is: {}".format(AccRetain))
    
    history = [evaluate(model, forget_valid_dl)]
    AccForget_Fmodel = history[0]["val_acc"]*100
    
    print("Accuracy of forget model on forget set is: {}".format(AccForget_Fmodel))
    
    history = [evaluate(model, retain_valid_dl)]
    AccRetain_Fmodel = history[0]["val_acc"]*100
    
    print("Accuracy of forget model on retain set is: {}".format(AccRetain_Fmodel))
    
    history = [evaluate(gold_model, forget_valid_dl)]
    AccForget_Gmodel = history[0]["val_acc"]*100
    
    print("Accuracy of gold model on forget set is: {}".format(AccForget_Gmodel))
    
    history = [evaluate(gold_model, retain_valid_dl)]
    AccRetain_Gmodel = history[0]["val_acc"]*100
    
    print("Accuracy of gold model on retain set is: {}".format(AccRetain_Gmodel))
    
    reqAccF = (1-error_range)*AccForget
    
    print("Desired Accuracy for retrain time with error range {} is {}".format(error_range, reqAccF))
    
    train_loader = DataLoader(train_data, batch_size, shuffle = True)
    train_loader = DeviceDataLoader(train_loader, device) 
    
    valid_loader = DataLoader(val_forget, batch_size)
    valid_loader = DeviceDataLoader(valid_loader, device) 
    
    rltime_gold = relearn_time(model = gold_model,
                               train_loader = train_loader,
                               valid_loader = valid_loader, 
                               reqAcc = reqAccF,
                               lr = lr)
    
    print("Relearning time for Gold Standard Model is {}".format(rltime_gold))
    
    rltime_forget = relearn_time(model = model, train_loader = train_loader, valid_loader = valid_loader, 
                               reqAcc = reqAccF, lr = lr)
    
    print("Relearning time for Forget Model is {}".format(rltime_forget))
    
    rl_coeff = rltime_forget/rltime_gold
    print("AIN = {}".format(rl_coeff))
    return rl_coeff
    
    
def cifar100_to_cifar20( target):
        _dict = \
        {0: 4,
        1: 1,
        2: 14,
        3: 8,
        4: 0,
        5: 6,
        6: 7,
        7: 7,
        8: 18,
        9: 3,
        10: 3,
        11: 14,
        12: 9,
        13: 18,
        14: 7,
        15: 11,
        16: 3,
        17: 9,
        18: 7,
        19: 11,
        20: 6,
        21: 11,
        22: 5,
        23: 10,
        24: 7,
        25: 6,
        26: 13,
        27: 15,
        28: 3,
        29: 15,
        30: 0,
        31: 11,
        32: 1,
        33: 10,
        34: 12,
        35: 14,
        36: 16,
        37: 9,
        38: 11,
        39: 5,
        40: 5,
        41: 19,
        42: 8,
        43: 8,
        44: 15,
        45: 13,
        46: 14,
        47: 17,
        48: 18,
        49: 10,
        50: 16,
        51: 4,
        52: 17,
        53: 4,
        54: 2,
        55: 0,
        56: 17,
        57: 4,
        58: 18,
        59: 17,
        60: 10,
        61: 3,
        62: 2,
        63: 12,
        64: 12,
        65: 16,
        66: 12,
        67: 1,
        68: 9,
        69: 19,
        70: 2,
        71: 10,
        72: 0,
        73: 1,
        74: 16,
        75: 12,
        76: 9,
        77: 13,
        78: 15,
        79: 13,
        80: 16,
        81: 19,
        82: 2,
        83: 4,
        84: 6,
        85: 19,
        86: 5,
        87: 5,
        88: 8,
        89: 19,
        90: 18,
        91: 1,
        92: 2,
        93: 15,
        94: 6,
        95: 0,
        96: 17,
        97: 8,
        98: 14,
        99: 13}

        return _dict[target]