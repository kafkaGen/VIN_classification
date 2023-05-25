import numpy as np
import torch
import torchmetrics
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from settings.config import Config

def get_model(pretrained=False):
      """
      Returns an instance of EfficientNet-B3 model with customized modifications.

      Args:
            pretrained (bool): If True, loads weights from a pre-trained model.

      Returns:
            torch.nn.Module: An instance of EfficientNet-B3 model.
      """
      if pretrained:
            model = torch.load(Config.best_model_path, map_location=Config.device)
      else:
            model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            model.features[0][0] = torch.nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            model.classifier[1] = torch.nn.Linear(in_features=1536, out_features=Config.num_classes, bias=True)
      
      
      return model


def train(model, optimizer, criterion, train_loader, valid_loader, lr_scheduler=None, model_name='model',
          num_epochs=Config.epochs, valid_acc_max=0, device=Config.device):
      """
      Trains a PyTorch model using the specified optimizer, criterion, and data loaders.

      Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            criterion: The loss function to optimize.
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            valid_loader (torch.utils.data.DataLoader): The data loader for validation data.
            lr_scheduler (_LRScheduler, optional): The learning rate scheduler (default: None).
            model_name (str, optional): The name of the model used for saving checkpoints (default: 'model').
            num_epochs (int, optional): The number of training epochs (default: Config.epochs).
            valid_acc_max (float, optional): The maximum validation accuracy achieved (default: 0).
            device (str, optional): The device to use for training (default: Config.device).

      Returns:
            dict: A dictionary containing training and validation metrics.
      """
      callbackers = {}
      callbackers['train_loss'] = []
      callbackers['train_accuracy'] = []
      callbackers['train_avg_precision'] = []
      callbackers['valid_loss'] = []
      callbackers['valid_accuracy'] = []
      callbackers['valid_avg_precision'] = []
      callbackers['valid_predictions'] = []
      callbackers['valid_targets'] = []
      accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=Config.num_classes)
      avg_precision = torchmetrics.AveragePrecision(task='multiclass', num_classes=Config.num_classes)
      model.to(device)
      
      for e in np.arange(num_epochs):
            train_loss = 0.0
            valid_loss = 0.0
            train_predictions = []
            train_targets = []
            valid_predictions = []
            valid_targets = []
            model.train()
            
            for imgs, labels in train_loader:
                  imgs = imgs.to(device)
                  labels = labels.to(device)
                  
                  out = torch.softmax(model(imgs), dim=1)
                  loss = criterion(out, labels)
                  
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  
                  train_predictions.extend(out.detach().to('cpu').numpy())
                  train_targets.extend(labels.clone().to('cpu').numpy())
                  train_loss += loss.item()*imgs.shape[0]
                  
            model.eval()
            with torch.no_grad():
                  for imgs, labels in valid_loader:
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        
                        out = torch.softmax(model(imgs), dim=1)
                        loss = criterion(out, labels)
                        
                        valid_predictions.extend(out.to('cpu').numpy())
                        valid_targets.extend(labels.clone().to('cpu').numpy())
                        valid_loss += loss.item()*imgs.shape[0]
            
            if lr_scheduler:
                  lr_scheduler.step()
                  
            train_predictions = torch.from_numpy(np.array(train_predictions))
            train_targets = torch.from_numpy(np.array(train_targets))
            valid_predictions = torch.from_numpy(np.array(valid_predictions))
            valid_targets = torch.from_numpy(np.array(valid_targets))
            train_loss /= len(train_loader.sampler)
            train_acc = accuracy(train_predictions, train_targets)
            train_avg_pr = avg_precision(train_predictions, train_targets)
            valid_loss /= len(valid_loader.sampler)
            valid_acc = accuracy(valid_predictions, valid_targets)
            valid_avg_pr = avg_precision(valid_predictions, valid_targets)
            
            print(f'Epoch {e+1}/{num_epochs}: TrainLoss {train_loss:.3f} TrainAcc {train_acc:.3f} TrainAP {train_avg_pr:.3f} ValidLoss: {valid_loss:.3f} ValidAcc: {valid_acc:.3f} ValidAP {valid_avg_pr:.3f}')
            
            callbackers['train_loss'].append(train_loss)
            callbackers['train_accuracy'].append(train_acc)
            callbackers['train_avg_precision'].append(train_avg_pr)
            callbackers['valid_loss'].append(valid_loss)
            callbackers['valid_accuracy'].append(valid_acc)
            callbackers['valid_avg_precision'].append(valid_avg_pr)
            
            if valid_acc > valid_acc_max:
                  #script = model.to_torchscript()
                  #torch.jit.save(script, f"{Config.model_path}/{model_name}.pt")
                  torch.save(model, f"{Config.model_path}/{model_name}.pt")
                  valid_acc_max = valid_acc
                  
                  callbackers['valid_predictions'] = torch.argmax(valid_predictions, axis=1).numpy()
                  callbackers['valid_targets'] = valid_targets.numpy()
                  
      return callbackers