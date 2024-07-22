
"""
Trains a PyTorch image classification model using a configurable set of hyperparameters.

This script allows for the training of a TinyVGG model on a specified dataset with customizable training options.
Users can specify the number of training epochs, batch size, number of hidden units, learning rate, and data transformation method through command-line arguments.

Parameters:
    --num_epochs (int): Number of epochs to train the model. Default is 10.
    --batch_size (int): Number of samples per batch. Default is 32.
    --hidden_units (int): Number of hidden units in each hidden layer of the model. Default is 10.
    --learning_rate (float): Learning rate for the optimizer. Default is 0.001.
    --train_dir (str): Directory path to the training data in standard image classification format. Default is 'data/pizza_steak_sushi/train'.
    --test_dir (str): Directory path to the testing data in standard image classification format. Default is 'data/pizza_steak_sushi/test'.
    --transform (str): Type of data transformation to apply ('simple' or 'augmented'). Default is 'simple'.

Example usage:
    !python scripts/train.py --num_epochs 15 --batch_size 32 --hidden_units 10 --learning_rate 0.001 --transform "augmented"
"""

import argparse
import os
import json
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer


parser = argparse.ArgumentParser(description="Get some hyperparameters.")

parser.add_argument("--num_epochs", 
                     default=10, 
                     type=int, 
                     help="the number of epochs to train for")

parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")

parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")
parser.add_argument("--transform",
                    default="simple",
                    type=str,
                    help="defining what transform should be used on train data (simple or augumented)")
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
TRANSFORM_TYPE = args.transform
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")


train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"


simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
    ])

trivial_augumented_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
    ])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=trivial_augumented_transform if TRANSFORM_TYPE=="augumented" else simple_transform,
    test_transform=simple_transform,
    batch_size=BATCH_SIZE
)

model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

start_time = timer()

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       epochs=NUM_EPOCHS,
                       device=device)

if TRANSFORM_TYPE == "augumented":
  with open('results/augumented_data_model_results.json', 'w') as f:
    json.dump(results, f)
else:
  with open('results/simple_data_model_results.json', 'w') as f:
    json.dump(results, f)
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

utils.save_model(model=model,
                 target_dir="models",
                 model_name="augumented_data_tinyvgg_model.pt" if TRANSFORM_TYPE == "augumented" else "simple_data_tinyvgg_model.pt")
