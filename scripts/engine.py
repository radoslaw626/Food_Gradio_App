
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import torch


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
  
  """
  Performs a single training step including forward pass, loss computation, 
  backpropagation, and optimizer step.

  Args:
      model (torch.nn.Module): The neural network model to be trained.
      dataloader (torch.utils.data.DataLoader): The DataLoader that provides batches of data for training.
      loss_fn (torch.nn.Module): The loss function to measure the model's performance.
      optimizer (torch.optim.Optimizer): The optimization algorithm to update the model's parameters.
      device (torch.device): The device on which the model is being trained (e.g., 'cuda' or 'cpu').

  Returns:
      float: The average training loss for the epoch.
      float: The average training accuracy for the epoch.
  """

  model.train()

  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class==y).sum().item()/len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
  
  """
  Performs a single evaluation step to calculate the loss and accuracy of the model on the test dataset.

  This function sets the model to evaluation mode, disables gradient calculations, and iterates over the test
  dataloader to compute the model's performance metrics.

  Args:
      model (torch.nn.Module): The neural network model to be evaluated.
      dataloader (torch.utils.data.DataLoader): The DataLoader providing the test dataset.
      loss_fn (torch.nn.Module): The loss function used to evaluate the model's performance.
      device (torch.device): The device (CPU or GPU) on which the model is being evaluated.

  Returns:
      float: The average loss of the model on the test dataset.
      float: The average accuracy of the model on the test dataset.
  """

  model.eval()
  test_loss, test_acc = 0,  0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      test_pred_logits = model(X)

      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):

  """
  Trains and evaluates a neural network model using specified data loaders, optimizer, loss function, and device.

  This function orchestrates the training and testing process of a PyTorch model for a given number of epochs.
  It logs the training and testing loss and accuracy after each epoch and returns these metrics in a dictionary.

  Args:
      model (torch.nn.Module): The neural network model to be trained and evaluated.
      train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
      test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing data.
      optimizer (torch.optim.Optimizer): Optimizer to use for training the model.
      loss_fn (torch.nn.Module): Loss function to use for evaluating the model.
      epochs (int): Number of epochs to train the model.
      device (torch.device): Device on which to train the model (e.g., 'cuda' or 'cpu').

  Returns:
      dict: A dictionary containing lists of training losses, training accuracies, testing losses, and testing accuracies for each epoch.

  The function performs the following steps for each epoch:
  1. Calls `train_step` to train the model on the training data.
  2. Calls `test_step` to evaluate the model on the testing data.
  3. Logs the losses and accuracies.
  4. Appends the results to the respective lists in the results dictionary.
  """

  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in tqdm(range(epochs)):

    train_loss, train_acc = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device)

    test_loss, test_acc = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device)

    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
      )

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
