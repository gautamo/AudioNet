required_training = True
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import os
import time

from typing import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision

class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()

        ### self._body NOT USED
        self._body = nn.Sequential(
            
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5), # 1st cnn layer
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5), # 2nd cnn layer
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5), # 3rd cnn layer
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        )  
        
        self._head = nn.Sequential(
            
        nn.Linear(in_features=92, out_features=300), # 1st layer
        nn.ReLU(inplace=True),

        nn.Linear(in_features=300, out_features=600), # 2nd layer
        nn.ReLU(inplace=True),
            
        nn.Linear(in_features=600, out_features=300), # 3rd layer
        nn.ReLU(inplace=True),
        
        nn.Linear(in_features=300, out_features=35531) # output layer
        )
        
    def forward(self, x):
        #x = self._body(x)
        x = x.view(x.size()[0], -1)
        x = self._head(x)
        
        return x

def get_data(batch_size, data_root, num_workers=4, subset = True):
    
    preprocess = transforms.Compose([  # DataLoader loads images with 3 color channels by default
    transforms.Grayscale(num_output_channels=1), # convert to one channel image 
    transforms.ToTensor() # flatten to tensor for AudioNet input
    ])
    
    train_data_path = os.path.join(data_root, 'train', 'train') # train dataloader path
    validation_data_path = os.path.join(data_root, 'validation', 'validation') # test dataloader path
    
    train_data = datasets.ImageFolder(root=train_data_path, transform=preprocess)
    validation_data = datasets.ImageFolder(root=validation_data_path, transform=preprocess)
    
    #print("Training Data Length", len(train_data))
    #print("Validation Data Length", len(validation_data))
    
    if subset: # use a subset of the data
        subset_size = 1.0 # use 100% of the data
        print(f"USING {int(subset_size*100)}% OF THE DATA")
        train_data = torch.utils.data.Subset(train_data, np.arange(0, len(train_data), int(1/subset_size)))
        validation_data = torch.utils.data.Subset(validation_data, np.arange(0, len(validation_data), int(1/subset_size)))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, validation_loader

@dataclass
class SystemConfiguration:
    '''
    Describes the common system setting needed for reproducible training
    '''
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)

@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 16  # amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 20  # number of times the whole dataset will be passed through the network
    learning_rate: float = 0.1  # determines the speed of network's weights update
        
    log_interval: int = 1000  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "/home/gbanuru/notebooks/CS175/data_root"  # folder to read/save data
    num_workers: int = 10  # number of concurrent processes using to prepare data
    device: str = 'cuda'  # device to use for training.

def setup_system(system_config: SystemConfiguration) -> None:
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic

def train(
    train_config: TrainingConfiguration, model: nn.Module, optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader, epoch_idx: int
    ) -> None:
    
    # change model in training mood
    model.train()
    
    # to get batch loss
    batch_loss = np.array([])
    
    # to get batch accuracy
    batch_acc = np.array([])
        
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # clone target
        indx_target = target.clone()
        # send data to device (its is medatory if GPU has to be used)
        data = data.to(train_config.device)
        # send target to device
        target = target.to(train_config.device)

        # reset parameters gradient to zero
        optimizer.zero_grad()
        
        # forward pass to the model
        output = model(data)
        
        # cross entropy loss
        loss = F.cross_entropy(output, target)
        
        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gardients
        optimizer.step()
        
        batch_loss = np.append(batch_loss, [loss.item()])
        
        # Score to probability using softmax
        prob = F.softmax(output, dim=1)
            
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]  
                        
        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()
            
        # accuracy
        acc = float(correct) / float(len(data))
        
        batch_acc = np.append(batch_acc, [acc])

        if batch_idx % train_config.log_interval == 0 and batch_idx > 0:              
            print(
                'Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
                    epoch_idx, batch_idx * len(data), len(train_loader.dataset), loss.item(), acc
                )
            )
            
    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()
    return epoch_loss, epoch_acc

def validate(
    train_config: TrainingConfiguration,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    ) -> float:
    
    model.eval()
    test_loss = 0
    count_corect_predictions = 0
    
    for data, target in test_loader:
        
        indx_target = target.clone()
        data = data.to(train_config.device)
        target = target.to(train_config.device)
        output = model(data)
        
        test_loss += F.cross_entropy(output, target).item() # add loss for each mini batch
        prob = F.softmax(output, dim=1) # Score to probability using softmax
        pred = prob.data.max(dim=1)[1] # get the index of the max probability
        count_corect_predictions += pred.cpu().eq(indx_target).sum() # add correct prediction count

    test_loss = test_loss / len(test_loader) # average over number of mini-batches
    accuracy = 100. * count_corect_predictions / len(test_loader.dataset) # average over number of dataset
    
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, count_corect_predictions, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy/100.0

def save_model(model, device, model_dir='models', model_file_name='final_audionet_model.pt'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    
    if device == 'cpu': # make sure you transfer the model to cpu.
        model.to('cpu')

    torch.save(model.state_dict(), model_path) # save the state_dict
    
    if device == 'cuda':
        model.to('cuda')
    
    return

def main(
    system_configuration=SystemConfiguration(), 
    training_configuration=TrainingConfiguration(), 
    neuralModel = AudioNet(),
    batch_size=16,
    epochs_count=20,
    learning_rate=0.1
    ):
    
    training_configuration.batch_size = batch_size
    training_configuration.epochs_count = epochs_count
    training_configuration.learning_rate = learning_rate
        
    setup_system(system_configuration) # system configuration
    
    batch_size_to_set = training_configuration.batch_size # batch size
    num_workers_to_set = training_configuration.num_workers # num_workers
    epoch_num_to_set = training_configuration.epochs_count # epochs
        
    if torch.cuda.is_available(): # if GPU is available use training config
        device = "cuda"; print("CUDA AVAILABLE")
    else:
        device = "cpu"; print("CPU AVAILABLE")
        num_workers_to_set = 2
        
    # data loader
    train_loader, test_loader = get_data(
        batch_size=training_configuration.batch_size,
        data_root=training_configuration.data_root,
        num_workers=num_workers_to_set
    )
        
    # Update training configuration
    training_configuration.device = device
    training_configuration.num_workers = num_workers_to_set
    
    # initiate model
    model = neuralModel
            
    # send model to device (GPU/CPU)
    model.to(training_configuration.device)
    
    # optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=training_configuration.learning_rate
    )
    
    best_loss = torch.tensor(np.inf)
    
    # epoch train/test loss
    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])
    
    # epch train/test accuracy
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])
        
    # trainig time measurement
    t_begin = time.time()
    for epoch in range(training_configuration.epochs_count):
        
        train_loss, train_acc = train(training_configuration, model, optimizer, train_loader, epoch)
        epoch_train_loss = np.append(epoch_train_loss, [train_loss])
        epoch_train_acc = np.append(epoch_train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * training_configuration.epochs_count - elapsed_time
        
        print(
            "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapsed_time, speed_epoch, speed_batch, eta
            )
        )

        if epoch % training_configuration.test_interval == 0:
            current_loss, current_accuracy = validate(training_configuration, model, test_loader)
            epoch_test_loss = np.append(epoch_test_loss, [current_loss])
            epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])
            
            if current_loss < best_loss:
                best_loss = current_loss
                print('Loss decreases, saving the model.\n')
                save_model(model, device)
                
    elapsed = time.time() - t_begin
    print("Total time: {:.2f}, Best Loss: {:.3f}".format(elapsed, best_loss))
    
    return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc, elapsed
    
if __name__ == "__main__":
    if required_training:
        model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc, elapsed = main()