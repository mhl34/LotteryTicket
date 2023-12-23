import torch

class hyperParams:
    def __init__(self):
        self.epochs = 1
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.batch_size = 128
        self.dropout_p = 0.3
        self.num_iterations = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'