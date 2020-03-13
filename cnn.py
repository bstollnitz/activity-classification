import torch.nn as nn

class CNN(nn.Module):

    def __init__(self) -> None:
        super(CNN, self).__init__()

        # Each batch has shape:
        # (batch_size, num_channels, height, width) = (batch_size, 9, 128, 128).

        self.conv1 = nn.Conv2d(
            in_channels = 9, # number of channels
            out_channels = 18, # number of filters
            kernel_size = 3,
            stride = 1,
            padding = 1 # this stride and padding don't change the (height, width) dims
            )
        self.maxpool = nn.MaxPool2d(kernel_size = 2) 
        self.conv2 = nn.Conv2d(
            in_channels = 18, # because we have 18 filters in conv1
            out_channels = 36, # number of filters
            kernel_size = 3,
            stride = 1,
            padding = 1
            )
        self.fc1 = nn.Linear(36*32*32, 20000)
        self.fc2 = nn.Linear(20000, 5000)
        self.fc3 = nn.Linear(5000, 6)

    def forward(self, x):
        # Each data point has dims 9 x 128 x 128 = 147,456.

        # Layer 1.
        x = nn.functional.relu(self.conv1(x))
        # Each data point has dims 18 x 128 x 128 = 294,912.
        x = self.maxpool(x)
        # Each data point has dims 18 x 64 x 64 = 73,738.

        # Layer 2.
        x = nn.functional.relu(self.conv2(x))
        # Each data point has dims 36 x 64 x 64 = 147,456.
        x = self.maxpool(x)
        # Each data point has dims 36 x 32 x 32 = 36,864.

        # Reshape data from (36, 32, 32) to (1, 36864).
        x = x.view(-1, 36*32*32)

        # Layer 3.
        x = nn.functional.relu(self.fc1(x))
        # Each data point has dims a 20,000 horizontal vector.

        # Layer 4.
        x = nn.functional.relu(self.fc2(x))
        # Each data point is now a 5,000 horizontal vector.

        # Layer 5.
        x = self.fc3(x)
        # Each data point is now a 6-long horizontal vector.
        # We don't call softmax because we'll use CrossEntropyLoss, which 
        # calls softmax already.

        return x
