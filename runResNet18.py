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
        for epoch in range(hyperParams.epochs):
            lossLst = []
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}', unit='batch')
            for batch_idx, data in progress_bar:
                image, target = data
                image, target = image.to(hyperParams.device), target.to(hyperParams.device)
                preds = model(image)

                loss = criterion(preds, target)
                lossLst.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}   Average Training Loss: {sum(lossLst)/len(lossLst)}")


    def run(self):
        model = ResNet18()
        hp = hyperParams()
        model = model.to(hp.device)
        optimizer = optim.SGD(model.parameters(), learning_rate = hp.learning_rate, momentum = hp.momentum, weight_decay = hp.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # load data
        root = "./data"
        transform = transforms.compose(
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        )
        dataset = CIFAR10(root = root, train = True, download = True, transform = transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) * 4 // 5, len(dataset) * 1 // 5])
        train_dataloader = DataLoader(train_dataset, batch_size = hp.batch_size, shuffle = True)

        self.train(model, train_dataloader, hp, optimizer, criterion)


if __name__ == "__main__":
    obj = runResNet18()
    obj.run()