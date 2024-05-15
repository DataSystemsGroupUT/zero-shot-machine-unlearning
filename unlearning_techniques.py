from logger import Logger
from utils import * 

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn


from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UnlearningTechnique:

    def __init__(self, enable_logs):
        self.enable_logs = enable_logs
        self.logger = Logger( tag = "Unleanring Techniques", enabled = enable_logs )
        
        
    def experimental_method(  self, model, target_class, number_of_classes,  Dr, Df, opt_func,  learning_rate ):
        
        class DistillKL(nn.Module):
            def __init__(self, T):
                super(DistillKL, self).__init__()
                self.T = T

            def forward(self, y_s, y_t):
                p_s = F.log_softmax(y_s/self.T, dim=1)
                p_t = F.softmax(y_t/self.T, dim=1)
                loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
                return loss
        
        Df = DataLoader(Df, 32)
        Df = DeviceDataLoader(Df, device) 

        Dr = DataLoader(Dr, 64)
        Dr = DeviceDataLoader(Dr, device) 
        
        criterion = DistillKL(4.0)
        
        ref_model = deepcopy(model)
        ref_model.eval()

        model.train()
        torch.cuda.empty_cache()

        optimizer = opt_func(model.parameters(), lr = learning_rate)

        self.logger.log( "forget  step..." )
        for batch in  Df :
            images, labels = batch
            
            random_input = torch.rand( images.shape, device=device )
            
            out_r, *_ = ref_model(random_input)   
            out, *_ = model(images)                  
                         
            loss = criterion(out, out_r)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # optimizer = opt_func(model.parameters(), lr = learning_rate * 10)
        self.logger.log( "retain  step..." )
        for remember_batch in Dr:
            loss = model.training_step(remember_batch) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        return model
        
        
    def randomize_labels(self, model, target_class, number_of_classes,  Dr, Df, opt_func,  learning_rate ):

        def generate_random_labels( exclude_label, number_of_classes, size ):
            random_labels = []
            generated = 0
            
            while generated < size:
                random_label = random.randint(0 , number_of_classes - 1 )
                
                while random_label == exclude_label:
                    random_label = random.randint(0 , number_of_classes - 1)

                random_labels.append( random_label )
                generated += 1
            
            return  random_labels
        
        
        self.logger.log( "randomize_labels" )
        random_labels =  generate_random_labels(exclude_label=target_class, number_of_classes=number_of_classes, size= len(Df) )
        for index, df_sample in enumerate(Df):
            Df[index][1] = random_labels[index]
        
        self.logger.log( "Labels randomized !" )
        
        
        Df = DataLoader(Df, 32)
        Df = DeviceDataLoader(Df, device) 

        Dr = DataLoader(Dr, 64)
        Dr = DeviceDataLoader(Dr, device) 

        model.train()
        torch.cuda.empty_cache()

        optimizer = opt_func(model.parameters(), lr = learning_rate)

        self.logger.log( "forget  step..." )
        for batch in  Df :
            loss = model.training_step(batch) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # optimizer = opt_func(model.parameters(), lr = learning_rate * 10)
        self.logger.log( "retain  step..." )
        for remember_batch in Dr:
            loss = model.training_step(remember_batch) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                    
        return model
    
    def finetuning(self, target_forget_class, Dr , model, opt_func, learning_rate, number_of_epochs, plot_exp = False):
        self.logger.log( "finetuning" )
        
        val_size = len(Dr) // 20
        train_size = len(Dr) - val_size
        train_ds, val_ds = random_split(Dr, [train_size, val_size])
        
        train_dl = DataLoader(train_ds, 64)
        train_dl = DeviceDataLoader(train_dl, device) 

        val_dl = DataLoader(val_ds, 128)
        val_dl = DeviceDataLoader(val_dl, device) 
        
        # train_dl = DataLoader(train_ds, 64, shuffle=True, num_workers=4, pin_memory=True)
        # val_dl = DataLoader(val_ds, 128, num_workers=4, pin_memory=True)
        
        # train_dl = DeviceDataLoader(train_dl, device)    
        # val_dl = DeviceDataLoader(val_dl, device)    
        
        history = fit(number_of_epochs, learning_rate, model, train_dl, val_dl, opt_func)
        
        if plot_exp:
            plot_accuracies(history, label = f"Finetunning --  Df = class_{target_forget_class}")
            plot_losses(history, label = f" Finetunning -- Df = class_{target_forget_class}")
        
        return model

    def retraining(self, model_name , dataset_name,  target_forget_class,  Dr, model, opt_func, learning_rate, number_of_epochs, save_model= True, plot_exp = False ):
        self.logger.log( "retraining" )
        
        val_size = len(Dr) // 20
        train_size = len(Dr) - val_size
        train_ds, val_ds = random_split(Dr, [train_size, val_size])
        
        train_dl = DataLoader(train_ds, 64, shuffle=True )
        val_dl = DataLoader(val_ds, 128)
        
        train_dl = DeviceDataLoader(train_dl, device)    
        val_dl = DeviceDataLoader(val_dl, device)    
        
        history = fit(number_of_epochs, learning_rate, model, train_dl, val_dl, opt_func)
        
        if plot_exp:
            plot_accuracies(history, label = f"retraining from scratch without class_{target_forget_class}")
            plot_losses(history, label = f"retraining from scratch without class_{target_forget_class}")
        
        if save_model:
            torch.save(model.state_dict(), f'./weights/{model_name}_{dataset_name}_epochs={number_of_epochs}_lr={learning_rate}_without class_{target_forget_class}.pth')
        return model
    
    def UNSIR(self, model, Dr, Df, opt_func,  learning_rate,  num_impair_steps = 2, num_repair_steps = 1  ):
        self.logger.log( "UNSIR" )
        
        class Noise(nn.Module):
            def __init__(self, batch_size, *dim):
                super().__init__()
                self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

            def forward(self):
                return self.noise

        def impair(retain_dl, forget_dl):
            unlearned_model = model

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = opt_func(unlearned_model.parameters(), lr=learning_rate)

            num_epochs = num_impair_steps
            for epoch in range(num_epochs):
                running_loss = 0

                for batch_idx, ((x_retain, y_retain), (x_forget, y_forget)) in enumerate(zip(retain_dl, forget_dl)):
                    y_retain = y_retain.cuda()
                    batch_size_forget = y_forget.size(0)

                    if x_retain.size(0) != 64 or x_forget.size(0) != 64:
                        continue

                    # Initialize the noise.
                    noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
                    noise = Noise(batch_size_forget, *noise_dim).cuda()
                    noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.01)
                    noise_tensor = noise()[:batch_size_forget]

                    # Update the noise for increasing the loss value.
                    for _ in range(5):
                        outputs, *_ = unlearned_model(noise_tensor)
                        with torch.no_grad():
                            target_logits, *_ = unlearned_model(x_forget.cuda())
                        # Maximize the similarity between noise data and forget features.
                        loss_noise = -F.mse_loss(outputs, target_logits)

                        # Backpropagate to update the noise.
                        noise_optimizer.zero_grad()
                        loss_noise.backward(retain_graph=True)
                        noise_optimizer.step()

                    # Train the model with noise and retain image
                    noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().cuda()
                    outputs, *_  = unlearned_model(noise_tensor.cuda())
                    loss_1 = criterion(outputs, y_retain)

                    outputs, *_  = unlearned_model(x_retain.cuda())
                    loss_2 = criterion(outputs, y_retain)

                    joint_loss = loss_1 + loss_2

                    optimizer.zero_grad()
                    joint_loss.backward()
                    optimizer.step()
                    running_loss += joint_loss.item() * x_retain.size(0)

                self.logger.log(f"Epoch {epoch+1} completed.")
        
        def repair(retain_dl):
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = opt_func(model.parameters(), lr=learning_rate)

            num_epochs = num_repair_steps
            for epoch in range(num_epochs):
                running_loss = 0
                for batch_idx, (x_retain, y_retain) in enumerate(retain_dl):
                    y_retain = y_retain.cuda()
                    # Classification Loss
                    outputs_retain, *_ = model(x_retain.cuda())
                    classification_loss = criterion(outputs_retain, y_retain)

                    optimizer.zero_grad()
                    classification_loss.backward()
                    optimizer.step()

                    running_loss += classification_loss.item() * x_retain.size(0)

                self.logger.log(f"Epoch {epoch+1} completed.")
        
        
        retain_dataloader = DataLoader(Dr, 64)
        forget_dataloader = DataLoader(Df, 64)
        
        retain_dataloader = DeviceDataLoader(retain_dataloader, device)    
        forget_dataloader = DeviceDataLoader(forget_dataloader, device)    
        
        impair( retain_dataloader, forget_dataloader )
        repair( retain_dataloader )
        
        return model
        
    def negative_gradiant(self, model, Dr ,Df, opt_func, learning_rate, num_epochs  ):
        self.logger.log( "negative_gradiant" )
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = opt_func(model.parameters(), lr=learning_rate)
        
        retain_dataloader = DataLoader(Dr, 128)
        forget_dataloader = DataLoader(Df, 128)
        
        retain_dataloader = DeviceDataLoader(retain_dataloader, device)    
        forget_dataloader = DeviceDataLoader(forget_dataloader, device)    

        dataloader_iterator = iter(forget_dataloader)

        for epoch in range(num_epochs):
            running_loss = 0

            for batch_idx, (x_retain, y_retain) in enumerate(retain_dataloader):

                try:
                    (x_forget, y_forget) = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(forget_dataloader)
                    (x_forget, y_forget) = next(dataloader_iterator)

                if x_forget.size(0) != x_retain.size(0):
                    continue

                outputs_forget, *_ = model(x_forget)
                loss_ascent_forget = -criterion(outputs_forget, y_forget)

                optimizer.zero_grad()
                loss_ascent_forget.backward()
                optimizer.step()

                running_loss += loss_ascent_forget.item() * x_retain.size(0)
                
            self.logger.log(f"Epoch {epoch+1} completed.")

        return model
    
    
    def SCRUM(self, model, Dr, Df, learning_rate ):
        
        self.logger.log( "SCRUM" )
        
        class DistillKL(nn.Module):
            def __init__(self, T):
                super(DistillKL, self).__init__()
                self.T = T

            def forward(self, y_s, y_t):
                p_s = F.log_softmax(y_s/self.T, dim=1)
                p_t = F.softmax(y_t/self.T, dim=1)
                loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
                return loss
        
        # Function to compute accuracy.
        def compute_accuracy(outputs, labels):
            _, predicted = outputs.max(1)
            total = labels.size(0)
            correct = predicted.eq(labels).sum().item()
            return 100 * correct / total
        
        
        retain_dataloader = DataLoader(Dr, 64)
        forget_dataloader = DataLoader(Df, 64)
        
        retain_dataloader = DeviceDataLoader(retain_dataloader, device)    
        forget_dataloader = DeviceDataLoader(forget_dataloader, device)    

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(4.0)
        criterion_kd = DistillKL(4.0)
        
        teacher = model
        student = deepcopy(model)
        
        optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate)
        
        student.train()
        teacher.eval()
        
        total_loss_retain, total_accuracy_retain = 0, 0
        total_loss_forget, total_accuracy_forget = 0, 0

        # Training with retain data.
        for inputs_retain, labels_retain in retain_dataloader:
            inputs_retain, labels_retain = inputs_retain.cuda(), labels_retain.cuda()

            # Forward pass: Student
            outputs_retain_student, *_ = student(inputs_retain)

            # Forward pass: Teacher
            with torch.no_grad():
                outputs_retain_teacher, *_ = teacher(inputs_retain)

            # Loss computation
            loss_cls = criterion_cls(outputs_retain_student, labels_retain)
            loss_div_retain = criterion_div(outputs_retain_student, outputs_retain_teacher)

            loss = loss_cls + loss_div_retain

            # Update total loss and accuracy for retain data.
            total_loss_retain += loss.item()
            total_accuracy_retain += compute_accuracy(outputs_retain_student, labels_retain)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Training with forget data.
        for inputs_forget, labels_forget in forget_dataloader:
            inputs_forget, labels_forget = inputs_forget.cuda(), labels_forget.cuda()

            # Forward pass: Student
            outputs_forget_student, *_ = student(inputs_forget)

            # Forward pass: Teacher
            with torch.no_grad():
                outputs_forget_teacher, *_  = teacher(inputs_forget)

            # We want to maximize the divergence for the forget data.
            loss_div_forget = -criterion_div(outputs_forget_student, outputs_forget_teacher)

            # Update total loss and accuracy for forget data.
            total_loss_forget += loss_div_forget.item()
            total_accuracy_forget += compute_accuracy(outputs_forget_student, labels_forget)

            # Backward pass
            optimizer.zero_grad()
            loss_div_forget.backward()
            optimizer.step()

        # Print average loss and accuracy for the entire epoch
        avg_loss_retain = total_loss_retain / len(retain_dataloader)
        avg_accuracy_retain = total_accuracy_retain / len(retain_dataloader)

        avg_loss_forget = total_loss_forget / len(forget_dataloader)
        avg_accuracy_forget = total_accuracy_forget / len(forget_dataloader)

        self.logger.log(f'Epoch Retain: Avg Loss: {avg_loss_retain:.4f}, Avg Accuracy: {avg_accuracy_retain:.2f}%')
        self.logger.log(f'Epoch Forget: Avg Loss: {avg_loss_forget:.4f}, Avg Accuracy: {avg_accuracy_forget:.2f}%')
        
        return student

    def lipschitz(self, model , Df, opt_func, learning_rate, noise_std ):
        self.logger.log( "lipschitz" )
        
        class AddGaussianNoise(object):
            def __init__(self, mean=0., std=1., device='cuda'):
                self.std = std
                self.mean = mean
                self.device = device
                
            def __call__(self, tensor):
                _max = tensor.max()
                _min = tensor.min()
                tensor = tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
                tensor = torch.clamp(tensor, min=_min, max=_max)
                return tensor
            
            def __repr__(self):
                return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
        noise = AddGaussianNoise(std= noise_std)
        
        forget_dataloader = DataLoader(Df, 64)
        forget_dataloader = DeviceDataLoader(forget_dataloader, device)    
        
        optimizer = opt_func(model.parameters(), lr=learning_rate)   
        
        for sample in forget_dataloader:
            x = sample[0].to(device)
            image = x.unsqueeze(0) if x.dim() == 3 else x
            out, *_ = model(image)                            
            loss = torch.tensor(0.0, device=device)
            
            #Build comparison images
            for _ in range(100):   
                img2 = noise(deepcopy(x))

                image2 = img2.unsqueeze(0) if img2.dim() == 3 else img2
                
                with torch.no_grad():
                    out2, *_ = model(image2)
                
                #ignore batch dimension        
                flatimg, flatimg2 = image.view(image.size()[0], -1), image2.view(image2.size()[0], -1)

                in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)              
                out_norm = torch.linalg.vector_norm(out - out2, dim=1)
                #K = 0.001 * ((0.4- (out_norm / in_norm)).sum()).abs()#1*((0.08-
                K =  ((out_norm / in_norm).sum()).abs()#pow(2)#  0.1                                                
                loss += K
            
            loss /= 100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
    
    def gkt(self):
        pass
    
    def emmn(self, model, num_classes, forget_class, Dr, learning_rate,  batch_size=4):
        
        class UNSIR_noise(torch.nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

            def forward(self):
                return self.noise
        def UNSIR_noise_train(
            noise, model, forget_class_label, num_epochs, noise_batch_size, device="cuda"
        ):
            opt = torch.optim.Adam(noise.parameters(), lr=0.1)

            for epoch in range(num_epochs):
                total_loss = []
                inputs = noise()
                labels = torch.zeros(noise_batch_size).to(device) + forget_class_label
                outputs, *_ = model(inputs)
                loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(
                    torch.sum(inputs**2, [1, 2, 3])
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
                if epoch % 5 == 0:
                    self.logger.log("Loss: {}".format(np.mean(total_loss)))

            return noise

        def emmn_noise_train(noise, model, target_class_label, num_epochs, noise_batch_size, device="cuda"):
            opt = torch.optim.Adam(noise.parameters(), lr=0.1)

            for epoch in range(num_epochs):
                total_loss = []
                inputs = noise()
                labels = torch.zeros(noise_batch_size).to(device) + target_class_label
                outputs, *_ = model(inputs)
                loss = F.cross_entropy(outputs, labels.long())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
                if epoch % 5 == 0:
                    self.logger.log("Loss: {}".format(np.mean(total_loss)))

            return noise


        def emmc_create_noisy_lodaer(
            noise_dict,
            batch_size=64,
            num_noise_batches=80,
            device="cuda"
        ):
            noisy_data = []
            for i in range(num_noise_batches):
                for j, noise_ in noise_dict.items():
                    batch = noise_()
                    for k in range(batch[0].size(0)):
                        noisy_data.append(
                            (
                                batch[k].detach().cpu(),
                                torch.tensor(j),
                                torch.tensor(j)
                            )
                        )
            noisy_loader = DataLoader(noisy_data, batch_size=batch_size, shuffle=True)
            return noisy_loader
        
        val_size = len(Dr) // 20
        train_size = len(Dr) - val_size
        _, val_ds = random_split(Dr, [train_size, val_size])
        
        noise_batch_size = 4
        retain_valid_dl = DataLoader(val_ds, noise_batch_size)
        
        retain_valid_dl = DeviceDataLoader(retain_valid_dl, device)    
        
        forget_class_label = forget_class
        img_shape = next(iter(retain_valid_dl))[0].shape[-1]
        noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
        noise = UNSIR_noise_train(
            noise, model, forget_class_label, 25, noise_batch_size, device=device
        )

        retain_noises = {i_: UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
                        for i_ in [j for j in range(num_classes) if j!=forget_class]}
        for i_, n_ in retain_noises.items():
            retain_noises[i_] = emmn_noise_train(n_, model, i_, 30, noise_batch_size, device=device)

        noises = retain_noises
        noises[forget_class] = noise

        noisy_loader = emmc_create_noisy_lodaer(
            noises,
            batch_size=noise_batch_size,
            device=device,
        )        
        
        history = fit(epochs=2 , lr=learning_rate, model =  model,  train_loader= noisy_loader, val_loader= retain_valid_dl,opt_func=  torch.optim.Adam)

        return model
        
        
    
        
        
    