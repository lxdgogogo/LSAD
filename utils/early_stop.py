import datetime
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, dataset=""):
        # dt = datetime.now()
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.filename = f'../model/{dataset}_{datetime.date.today()}.pth'

    def step(self, loss, model, acc=0):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(model)
            self.best_loss = loss
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    # def load_checkpoint(self):
    #     model = torch.load(self.filename)
    #     return model

