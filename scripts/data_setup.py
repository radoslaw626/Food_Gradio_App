
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
  train_dir: str,
  test_dir: str,
  train_transform: transforms.Compose,
  test_transform: transforms.Compose,
  batch_size: int,
  num_workers: int=NUM_WORKERS
):

  """
  Creates DataLoader objects for training and testing datasets.

  This function initializes DataLoader objects for both training and testing datasets using the ImageFolder class
  from torchvision. It applies the specified transformations to the images and organizes them into batches.

  Args:
      train_dir (str): The directory path containing the training images organized in subdirectories per class.
      test_dir (str): The directory path containing the testing images organized in subdirectories per class.
      train_transform (transforms.Compose): A composition of transformations to apply to the training images.
      test_transform (transforms.Compose): A composition of transformations to apply to the testing images.
      batch_size (int): The number of images to process in each batch.
      num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to the number of CPUs available on the machine.

  Returns:
      tuple: A tuple containing the training DataLoader, testing DataLoader, and a list of class names derived from the training dataset.
  """

  train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform,
                                  target_transform=None)

  test_data = datasets.ImageFolder(root=test_dir,
                                 transform=test_transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             pin_memory=True)
  test_dataloader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True)

  return train_dataloader, test_dataloader, class_names
