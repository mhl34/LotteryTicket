import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from ResNet18 import ResNet18
from hyperParams import hyperParams
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class runResNet18():
    def __init__(self):
        pass

    def train(self, model, dataloader, hyperParams, optimizer, criterion):
        model.train()
        bestLoss = float('inf')
        for epoch in range(hyperParams.epochs):
            lossLst = []
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}', unit='batch')
            for batch_idx, data in progress_bar:
                image, target = data
                image, target = image.to(hyperParams.device), target.to(hyperParams.device)
                outputs = model(image)

                preds = torch.argmax(outputs, dim = 1)

                loss = criterion(outputs, target)
                lossLst.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avgLoss = sum(lossLst)/len(lossLst)
            print(f"Epoch {epoch + 1}   Average Training Loss: {avgLoss}")
            if avgLoss < bestLoss:
                print("Saving ...")
                # Save the learned representation for downstream tasks
                state = {'state_dict': model.state_dict(),
                         'epoch': epoch,
                         'lr': hyperParams.learning_rate}
                torch.save(state, f'resnet_model.pth')
                best_loss = avgLoss


    def run(self):
        model = ResNet18()
        hp = hyperParams()
        model = model.to(hp.device)
        optimizer = optim.SGD(model.parameters(), lr = hp.learning_rate, momentum = hp.momentum, weight_decay = hp.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # load data
        root = "./data"
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])
        dataset = CIFAR10(root = root, train = True, download = True, transform = transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) * 4 // 5, len(dataset) * 1 // 5])
        train_dataloader = DataLoader(train_dataset, batch_size = hp.batch_size, shuffle = True)

        self.train(model, train_dataloader, hp, optimizer, criterion)


if __name__ == "__main__":
    obj = runResNet18()
    obj.run()