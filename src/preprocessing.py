import numpy as np
import torch
from torchvision import transforms as T

from settings.config import Config


def set_seed(seed):
    """
    Set the random seed for NumPy and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
 

def get_shuffle(subset):
    """
    Determines whether to shuffle the data for the specified subset.

    Args:
        subset (str): The subset for which shuffling is to be determined. Should be one of 'train' or 'test'.

    Returns:
        bool: True if shuffling is required, False otherwise.
    """
    if subset == 'train':
        return True
    else:
        return False
    

class CustomPaddingRemove():
    def __call__(self, image):
        """
        Removes the white border around a squared image.

        Args:
            image (PIL.Image or torch.Tensor): The input image.

        Returns:
            PIL.Image or torch.Tensor: The image with the white border removed.
        """
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Calculate the threshold value for considering a pixel as part of the border
        threshold = 0.08 * image.shape[1]
        # Find the indices of the rows and columns to be removed
        x_indices = np.array(((image.sum(axis=1) / 255).round() == image.shape[1])).flatten()
        x_rem = [i for i, val in enumerate(x_indices) if val and (i < threshold or i > image.shape[0] - threshold)]     # round because jpg lose some info

        y_indices = np.array(((image.sum(axis=2) / 255).round() == image.shape[1])).flatten()
        y_rem = [i for i, val in enumerate(y_indices) if val and (i < threshold or i > image.shape[0] - threshold)]

        # Remove the border by setting the corresponding pixels to 0
        image[:, :, x_rem] = 0
        image[:, y_rem, :] = 0

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(torch.float32)

        return image
    
    
def get_transforms(subset):
    """
    Returns a composition of image transformations for the specified subset.

    Args:
        subset (str): The subset for which the transforms are required. Should be one of 'train' or 'test'.

    Returns:
        torchvision.transforms.Compose: A composition of image transformations.
    """
    if subset == 'train':
        transforms = T.Compose([
            CustomPaddingRemove(),
            T.Resize(Config.resize_to),
            T.RandomResizedCrop(Config.resize_to, scale=[0.9, 1.0]),
            T.RandomAffine(0, translate=(0.075, 0.075)),
            T.RandomRotation(10, fill=(0,)),
            T.Normalize(mean=Config.mean, std=Config.std)
        ])
    else:
        transforms = T.Compose([
            CustomPaddingRemove(),
            T.Resize(Config.resize_to),
            T.Normalize(mean=Config.mean, std=Config.std)
        ])
    
    return transforms
