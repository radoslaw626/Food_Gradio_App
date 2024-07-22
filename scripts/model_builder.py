
import torch

from torch import nn

class TinyVGG(nn.Module):

  """
  A simplified version of the VGG network architecture for image classification.

  This class implements a miniature version of the VGG (Visual Geometry Group) network,
  which is a type of convolutional neural network used for image recognition tasks.

  Attributes:
      conv_block_1 (nn.Sequential): The first convolutional block consisting of two convolutional
                                    layers followed by a ReLU activation and a max pooling layer.
      conv_block_2 (nn.Sequential): The second convolutional block, structured similarly to the
                                    first block, for further feature extraction.
      classifier (nn.Sequential): A classifier block that flattens the output of the last convolutional
                                  block and passes it through a linear layer to produce the final output.

  Args:
      input_shape (int): The number of input channels (e.g., 3 for RGB images).
      hidden_units (int): The number of channels produced by the convolutional layers.
      output_shape (int): The number of output units, corresponding to the number of classes.

  Methods:
      forward(x): Defines the computation performed at every call. It passes the input through two
                  convolutional blocks and then through the classifier to produce the output predictions.
  """


  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*13*13,
                  out_features=output_shape)
    )

  def forward(self, x):

    """
    Forward pass of the TinyVGG model.

    Args:
        x (torch.Tensor): The input tensor containing the image data.

    Returns:
        torch.Tensor: The output tensor containing the class probabilities.
    """

    return self.classifier(self.conv_block_2(self.conv_block_1(x)))
