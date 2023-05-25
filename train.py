import torch
from torch.utils.data import DataLoader

from src.preprocessing import set_seed, get_shuffle, get_transforms
from src.dataset import VINDataset
from src.model import get_model, train
from settings.config import Config

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    set_seed(Config.seed)

    datasets = {name: VINDataset(name, get_transforms(name)) for name in ['train', 'test']}
    dataloaders = {name: DataLoader(datasets[name], batch_size=Config.batch_size, 
                                    shuffle=get_shuffle(name), num_workers=Config.num_workers) 
                for name in ['train', 'test']}
    
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 10], gamma=0.84, verbose=True)
    callbackers = train(model, optimizer, criterion, dataloaders['train'], dataloaders['test'], lr_scheduler=lr_scheduler)
    
    print(f"Best model accuracy: {max(callbackers['valid_accuracy']):.3f}")
    