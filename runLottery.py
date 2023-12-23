from ResNet18 import ResNet18
from hyperParams import hyperParams
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

class Lottery():
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, model, dataloader, hyperparams, criterion, optimizer):
        model.train()
        losses = []
        for epoch in range(hyperparams.epochs):
            batchLoss = []
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}', unit='batch')
            for batch_idx, data in progress_bar:
                image, target = data
                image, target = image.to(self.device), target.to(self.device)
                
                outputs = model(image)
                
                preds = torch.argmax(outputs, dim = 1)
                
                loss = criterion(outputs, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batchLoss.append(loss.item())
            
            avgLoss = sum(batchLoss) / len(batchLoss)
            losses.append(avgLoss)
        return losses
            
                
    
    def createMask(self, model, dropout_p):
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name and ('conv' in name or 'fc' in name):
                threshold = param.data.flatten().quantile(dropout_p).item()
                masks[name] = torch.where(param.data >= threshold, 1, 0)
        return masks
    
    def applyMask(self, model, masks):
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name].to(self.device)
                param.data = param.data * mask

    def initResNet(self, model):
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain('relu'))
            else:
                torch.nn.init.constant_(param.data, 0.01)
    
    def run(self):
        resnet_model = ResNet18()
        hp = hyperParams()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(resnet_model.parameters(), lr = hp.learning_rate, momentum = hp.momentum, weight_decay = hp.weight_decay)

        
        if not os.path.exists("resnet_init.pth"):
            self.initResNet(resnet_model)
            print("Saving ...")
            # Save the learned representation for downstream tasks
            state = {'state_dict': resnet_model.state_dict(),
                     'lr': hp.learning_rate}
            torch.save(state, f'resnet_init.pth')
        state_dict = torch.load("resnet_init.pth", map_location=torch.device('cuda'))['state_dict']
        resnet_model.load_state_dict(state_dict)
        resnet_model = resnet_model.to(self.device)
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        
        root = "./data"
        dataset = CIFAR10(root = root, train = True, download = True, transform = transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) * 4 // 5, len(dataset) * 1 // 5])
        train_dataloader = DataLoader(train_dataset, batch_size = hp.batch_size, shuffle = True)
        test_dataset = CIFAR10(root = root, train = False, download = True, transform = transform)
        test_dataloader = DataLoader(test_dataset, batch_size = hp.batch_size, shuffle = True)
        
        trainLossDict = {}
        
        for iteration in range(hp.num_iterations):
            print(f"Iteration: {iteration + 1}")
            # at each iteration, continue to dropout at P ^ (1/N) %
            if iteration != 0:
                masks = self.createMask(resnet_model, hp.dropout_p ** (1/iteration))
                self.applyMask(resnet_model, masks)
            
            trainLoss = self.train(resnet_model, train_dataloader, hp, criterion, optimizer)
            trainLossDict[iteration] = trainLoss
            
            print(f"Loss: {min(trainLoss)}")
            
        plt.plot(trainLossDict)
        plt.savefig("result.png")
            
            

if __name__ == "__main__":
    hp = hyperParams()
    obj = Lottery(hp.num_iterations)
    obj.run()