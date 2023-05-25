import os
from pathlib import Path
import filetype
import numpy as np
import pandas as pd
import cv2 as cv
import torch
from argparse import ArgumentParser

from src.model import get_model
from src.preprocessing import get_transforms
from settings.config import Config

import warnings
warnings.filterwarnings('ignore')


def is_valid_directory(path):
    """
    Check if the provided path is a valid directory.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path is a valid directory, False otherwise.
    """
    if not  os.path.exists(path):
        parser.error(f"Error: Directory '{path}' does not exist.")

    if not os.path.isdir(path):
        parser.error(f"Error: '{path}' is not a directory.")


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    
    is_valid_directory(args.input)
    
    mapping = pd.read_csv(Config.mapping_path, header=None, delimiter=' ', 
                              index_col=0, names=['ASCII'])
    mapping.index.name = 'label'
    mapping['char'] = mapping['ASCII'].apply(chr)
    # leave only digits and upper case letter labels except I, Q and O
    mapping = mapping[mapping['char'].str.contains(r'^[0-9A-HJ-NP-PR-Z]+$')]
    mapping['old_index'] = mapping.index
    mapping.index = range(mapping.shape[0])
    
    model = get_model(pretrained=True)
    model.eval()
    model.to('cpu')
    torch.set_grad_enabled(False)
    
    for el in os.listdir(args.input):
        fl_path = os.path.join(args.input, el)
        fl_path_posix = Path(args.input, el)
        if not filetype.is_image(fl_path):
            continue
        img = cv.imread(fl_path, 0).T[np.newaxis, :, :]
        img = torch.from_numpy(img).to(torch.float32)
        img = torch.unsqueeze(get_transforms('test')(img), 0)

        pred = model(img)
        pred = np.argmax(pred).item()
        pred = mapping.iloc[pred]['ASCII']
        
        print(str(pred).rjust(3, '0'), fl_path_posix.as_posix(), sep=', ')
        