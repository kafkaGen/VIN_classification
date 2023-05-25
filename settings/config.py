import torch


class Config():
    # Training
    seed = 13
    epochs = 33
    learning_rate = 1e-4
    num_classes = 33
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Data preparation
    train_data_path = 'data/emnist-balanced-train.csv'
    test_data_path = 'data/emnist-balanced-test.csv'
    mapping_path = 'data/emnist-balanced-mapping.txt'
    model_path = 'models/'
    best_model_path = 'models/model.pt'
    batch_size = 128
    num_workers = 4
    
    # Data transformation
    resize_to = (28, 28)
    mean = (0.1307,)
    std = (0.3081,)