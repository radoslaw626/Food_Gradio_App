
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  
  """
  Saves a PyTorch model to a specified directory with a given filename.

  Args:
  model (torch.nn.Module): The model to be saved.
  target_dir (str): The directory where the model will be saved.
  model_name (str): The name of the file for saving the model. Must end with '.pth' or '.pt'.

  Raises:
  AssertionError: If the model_name does not end with '.pth' or '.pt'.

  Prints:
  Information about the model saving path.

  Example:
  >>> model = torch.nn.Linear(10, 2)
  >>> save_model(model, 'models', 'linear_model.pth')
  [INFO] Saving model to: models/linear_model.pth
  """

  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)


  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)



def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:

    """
    Loads a PyTorch model from a specified file path.

    Args:
    model (torch.nn.Module): The model architecture into which the weights will be loaded.
    model_path (str): The path to the model file. Must end with '.pt' or '.pth'.

    Returns:
    torch.nn.Module: The model with loaded weights.

    Raises:
    AssertionError: If the model file does not exist or the file extension is not '.pt' or '.pth'.

    Example:
    >>> model = torch.nn.Linear(10, 2)
    >>> loaded_model = load_model(model, 'path/to/model.pth')
    [INFO] Loading model from: path/to/model.pth
    """

    model_path = Path(model_path)
    assert model_path.exists(), "Model file does not exist"
    assert model_path.suffix in ['.pt', '.pth'], "Model file should be a '.pt' or '.pth' file"

    print(f"[INFO] Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))

    return model


def plot_loss_curves(results: Dict[str, List[float]]):

  """
  Plots the loss and accuracy curves for training and testing data.

  This function takes a dictionary containing the loss and accuracy metrics
  for both training and testing data across epochs and generates two subplots:
  one for loss and one for accuracy.

  Args:
  results (Dict[str, List[float]]): A dictionary with keys 'train_loss', 'test_loss',
                                    'train_acc', and 'test_acc'. Each key should be
                                    associated with a list of floats representing the
                                    metric for each epoch.

  The function creates a figure with two subplots: the first subplot shows the
  training and testing loss curves, and the second subplot shows the training
  and testing accuracy curves.

  Example:
  >>> results = {
  ...     "train_loss": [0.1, 0.08, 0.06],
  ...     "test_loss": [0.1, 0.09, 0.08],
  ...     "train_acc": [0.8, 0.85, 0.9],
  ...     "test_acc": [0.7, 0.75, 0.8]
  ... }
  >>> plot_loss_curves(results)
  """

  loss = results["train_loss"]
  test_loss = results["test_loss"]

  accuracy = results["train_acc"]
  test_accuracy = results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="test_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend();
