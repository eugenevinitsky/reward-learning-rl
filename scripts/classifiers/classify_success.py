import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
#from torchvision.datasets import MNIST
#from torchvision import transforms

import pytorch_lightning as pl

from data_loader.bottle_loader import BottleDataset
fail_dir = os.path.join(__file__, os.path.abspath('../../data/fail'))
success_dir = os.path.join(__file__, os.path.abspath('../../data/succ'))

class CoolSystem(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()
        self.make_square = nn.AvgPool2d((4, 3))
        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 3).float()
        x = self.make_square(x)
        x = self.pool1(x)
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)

    # def forward(self, x):
    #     import ipdb; ipdb.set_trace()
    #     # this is some nonsense where we unravel the whole thing
    #     return torch.squeeze(torch.relu(self.l1(x.float().view(x.size(0), -1))))
    #     #return torch.relu(self.l1(self.pool(x.view(x_trans.size(0), -1))))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        pos_weight = torch.ones([1])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(y_hat, y.float())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_idx):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'val_loss': F.cross_entropy(y_hat, y)}
    #
    # def validation_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    #
    # def test_step(self, batch, batch_idx):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}
    #
    # def test_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(BottleDataset(success_dir=success_dir, fail_dir=fail_dir), batch_size=32)

    # @pl.data_loader
    # def val_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
    #
    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
    #                       batch_size=32)

if __name__=='__main__':
    from pytorch_lightning import Trainer

    model = CoolSystem()

    # most basic trainer, uses good defaults
    trainer = Trainer(min_epochs=100)
    trainer.fit(model)
